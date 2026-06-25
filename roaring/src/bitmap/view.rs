use alloc::boxed::Box;
use alloc::vec::Vec;
use core::cmp::Ordering;
use core::fmt;

use crate::bitmap::container::{Container, ARRAY_LIMIT};
use crate::bitmap::store::{
    ArrayStore, BitmapStore, Interval, IntervalStore, Store, BITMAP_BYTES, BITMAP_LENGTH,
    RUN_ELEMENT_BYTES, RUN_NUM_BYTES,
};
use crate::bitmap::util;
use crate::bitmap::{
    DESCRIPTION_BYTES, NO_OFFSET_THRESHOLD, OFFSET_BYTES, SERIAL_COOKIE,
    SERIAL_COOKIE_NO_RUNCONTAINER,
};
use crate::RoaringBitmap;

/// A zero-copy view over portable-format [`RoaringBitmap`] bytes.
///
/// The view parses the portable-format header and container metadata up front,
/// but keeps container payloads borrowed from the original byte slice. Read-only
/// operations dispatch directly against those borrowed payloads.
#[derive(Clone, Debug)]
pub struct RoaringBitmapView<'a> {
    bytes: &'a [u8],
    containers: Vec<ContainerEntry>,
    /// Prefix sums of container cardinalities. `prefix_cardinalities[i]` is the
    /// total number of values in containers `0..i`, so the final entry equals
    /// `len`. Used for O(1) `rank` cross-container lookups.
    prefix_cardinalities: Vec<u64>,
    len: u64,
}

/// An iterator over values in a [`RoaringBitmapView`].
#[derive(Clone, Debug)]
pub struct RoaringBitmapViewIter<'view, 'bytes> {
    view: &'view RoaringBitmapView<'bytes>,
    container_index: usize,
    inner: Option<ViewContainerIter<'bytes>>,
    remaining: u64,
}

/// An error returned when serialized bytes cannot be parsed as a bitmap view.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParseError {
    /// The input ended before a required field or container payload.
    Truncated {
        /// Offset where the read was attempted.
        offset: usize,
        /// Number of bytes needed at `offset`.
        needed: usize,
        /// Number of bytes available at `offset`.
        got: usize,
    },
    /// The format cookie was not recognized.
    BadCookie {
        /// The cookie value found in the input.
        got: u32,
    },
    /// The serialized container count exceeds the maximum supported count.
    TooManyContainers {
        /// The serialized container count.
        count: u64,
        /// The maximum supported container count.
        max: u64,
    },
    /// A run container payload is malformed.
    InvalidRun {
        /// Offset of the invalid run container payload.
        offset: usize,
        /// Human-readable error detail.
        detail: &'static str,
    },
    /// A non-run container payload or shared bitmap metadata is malformed.
    InvalidContainer {
        /// Offset of the invalid data.
        offset: usize,
        /// Human-readable error detail.
        detail: &'static str,
    },
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct ContainerEntry {
    key: u16,
    cardinality: u32,
    kind: ContainerKind,
    payload_offset: usize,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum ContainerKind {
    Array,
    Bitmap,
    Run { runs: u16 },
}

#[derive(Clone, Debug)]
struct ViewContainerIter<'a> {
    key: u16,
    inner: ViewContainerValues<'a>,
}

#[derive(Clone, Debug)]
enum ViewContainerValues<'a> {
    Array { payload: &'a [u8], pos: usize, end: usize },
    Bitmap { payload: &'a [u8], next_word: usize, current_word: u64, current_base: u16 },
    Run { payload: &'a [u8], run_index: usize, run_count: usize, current: Option<(u16, u16)> },
}

impl<'a> RoaringBitmapView<'a> {
    /// Parse portable-format bitmap bytes into a zero-copy view.
    ///
    /// This validates container ordering, payload bounds, container shape, and
    /// cardinality consistency without copying the container payloads.
    pub fn try_new(bytes: &'a [u8]) -> Result<Self, ParseError> {
        let mut offset = 0;
        let cookie = read_u32(bytes, &mut offset)?;

        let (size, has_offsets, run_container_bitmap) = if cookie == SERIAL_COOKIE_NO_RUNCONTAINER {
            let count = read_u32(bytes, &mut offset)? as u64;
            validate_container_count(count)?;
            (count as usize, true, None)
        } else if (cookie as u16) == SERIAL_COOKIE {
            let count = u64::from(cookie >> 16) + 1;
            validate_container_count(count)?;
            let size = count as usize;
            let bitmap_len = size.div_ceil(8);
            let run_bitmap = take(bytes, &mut offset, bitmap_len)?;
            (size, size >= NO_OFFSET_THRESHOLD, Some(run_bitmap))
        } else {
            return Err(ParseError::BadCookie { got: cookie });
        };

        let descriptions_offset = offset;
        let descriptions_len = size
            .checked_mul(DESCRIPTION_BYTES)
            .ok_or(ParseError::TooManyContainers { count: size as u64, max: max_containers() })?;
        let descriptions = take(bytes, &mut offset, descriptions_len)?;

        let offsets_offset = offset;
        let serialized_offsets = if has_offsets {
            let offsets_len =
                size.checked_mul(OFFSET_BYTES).ok_or(ParseError::TooManyContainers {
                    count: size as u64,
                    max: max_containers(),
                })?;
            Some(take(bytes, &mut offset, offsets_len)?)
        } else {
            None
        };

        let payloads_offset = offset;
        let mut payload_offset = payloads_offset;
        let mut containers = Vec::with_capacity(size);
        let mut prefix_cardinalities = Vec::with_capacity(size + 1);
        prefix_cardinalities.push(0);
        let mut total_len = 0u64;
        let mut last_key = None;

        for index in 0..size {
            let description_offset = descriptions_offset + index * DESCRIPTION_BYTES;
            let key = read_u16_at(descriptions, index * DESCRIPTION_BYTES);
            if let Some(last) = last_key.replace(key) {
                if key <= last {
                    return Err(ParseError::InvalidContainer {
                        offset: description_offset,
                        detail: "container keys are not sorted",
                    });
                }
            }

            let cardinality =
                u32::from(read_u16_at(descriptions, index * DESCRIPTION_BYTES + 2)) + 1;
            let is_run = run_container_bitmap
                .is_some_and(|bitmap| bitmap[index / 8] & (1 << (index % 8)) != 0);

            let current_payload_offset = if let Some(offsets) = serialized_offsets {
                let start = index * OFFSET_BYTES;
                read_u32_at(offsets, start) as usize
            } else {
                payload_offset
            };

            if current_payload_offset < payloads_offset {
                return Err(ParseError::InvalidContainer {
                    offset: offsets_offset + index * OFFSET_BYTES,
                    detail: "container offset points into header",
                });
            }
            if current_payload_offset != payload_offset {
                return Err(ParseError::InvalidContainer {
                    offset: offsets_offset + index * OFFSET_BYTES,
                    detail: "container offset does not match payload layout",
                });
            }

            let (kind, byte_size) = if is_run {
                let runs = read_u16_at_abs(bytes, current_payload_offset)?;
                if runs == 0 {
                    return Err(ParseError::InvalidRun {
                        offset: current_payload_offset,
                        detail: "run container with zero runs",
                    });
                }
                let byte_size = RUN_NUM_BYTES + usize::from(runs) * RUN_ELEMENT_BYTES;
                validate_run(bytes, current_payload_offset, runs, cardinality)?;
                (ContainerKind::Run { runs }, byte_size)
            } else if u64::from(cardinality) <= ARRAY_LIMIT {
                let byte_size = usize::try_from(cardinality).unwrap() * 2;
                validate_array(bytes, current_payload_offset, cardinality)?;
                (ContainerKind::Array, byte_size)
            } else {
                validate_bitmap(bytes, current_payload_offset, cardinality)?;
                (ContainerKind::Bitmap, BITMAP_BYTES)
            };

            checked_payload(bytes, current_payload_offset, byte_size)?;
            payload_offset = current_payload_offset.checked_add(byte_size).ok_or(
                ParseError::InvalidContainer {
                    offset: current_payload_offset,
                    detail: "container payload offset overflow",
                },
            )?;

            total_len += u64::from(cardinality);
            prefix_cardinalities.push(total_len);
            containers.push(ContainerEntry {
                key,
                cardinality,
                kind,
                payload_offset: current_payload_offset,
            });
        }

        Ok(Self { bytes, containers, prefix_cardinalities, len: total_len })
    }

    /// Returns the number of values in the bitmap.
    pub fn len(&self) -> u64 {
        self.len
    }

    /// Returns `true` if the bitmap contains no values.
    pub fn is_empty(&self) -> bool {
        self.containers.is_empty()
    }

    /// Returns `true` if the bitmap contains `value`.
    pub fn contains(&self, value: u32) -> bool {
        let (key, index) = util::split(value);
        self.containers
            .binary_search_by_key(&key, |container| container.key)
            .map(|pos| self.containers[pos].contains(self.bytes, index))
            .unwrap_or(false)
    }

    /// Returns the smallest value in the bitmap.
    pub fn min(&self) -> Option<u32> {
        let container = self.containers.first()?;
        Some(util::join(container.key, container.min(self.bytes)))
    }

    /// Returns the largest value in the bitmap.
    pub fn max(&self) -> Option<u32> {
        let container = self.containers.last()?;
        Some(util::join(container.key, container.max(self.bytes)))
    }

    /// Returns the number of values less than or equal to `value`.
    pub fn rank(&self, value: u32) -> u64 {
        let (key, index) = util::split(value);
        match self.containers.binary_search_by_key(&key, |container| container.key) {
            Ok(pos) => {
                self.containers[pos].rank(self.bytes, index) + self.prefix_cardinalities[pos]
            }
            Err(pos) => self.prefix_cardinalities[pos],
        }
    }

    /// Computes the len of the intersection with another view without materializing a bitmap.
    pub fn intersection_len(&self, other: &Self) -> u64 {
        let mut lhs = 0;
        let mut rhs = 0;
        let mut total = 0;

        while let (Some(left), Some(right)) = (self.containers.get(lhs), other.containers.get(rhs))
        {
            match left.key.cmp(&right.key) {
                Ordering::Less => lhs += 1,
                Ordering::Greater => rhs += 1,
                Ordering::Equal => {
                    total += left.intersection_len(self.bytes, right, other.bytes);
                    lhs += 1;
                    rhs += 1;
                }
            }
        }

        total
    }

    /// Computes the len of the union with another view without materializing a bitmap.
    pub fn union_len(&self, other: &Self) -> u64 {
        self.len() + other.len() - self.intersection_len(other)
    }

    /// Computes the len of the difference with another view without materializing a bitmap.
    pub fn difference_len(&self, other: &Self) -> u64 {
        self.len() - self.intersection_len(other)
    }

    /// Computes the len of the symmetric difference with another view without materializing a
    /// bitmap.
    pub fn symmetric_difference_len(&self, other: &Self) -> u64 {
        let intersection_len = self.intersection_len(other);
        self.len() + other.len() - intersection_len - intersection_len
    }

    /// Computes the len of the symmetric difference with another view without materializing a
    /// bitmap.
    pub fn xor_len(&self, other: &Self) -> u64 {
        self.symmetric_difference_len(other)
    }

    /// Returns `true` if every value in this view is also contained in `other`.
    ///
    /// Matches the set-equality semantics of [`RoaringBitmap::is_subset`]: a
    /// `Bitmap`-shaped container is never reported as a subset of an
    /// `Array`-shaped container, regardless of cardinality.
    pub fn is_subset(&self, other: &Self) -> bool {
        // Total-cardinality short-circuit: a subset can never have more values
        // than its superset. Both lengths are cached on the view so this is one
        // u64 compare with no payload reads.
        if self.len > other.len {
            return false;
        }

        let mut lhs = 0;
        let mut rhs = 0;
        while lhs < self.containers.len() {
            let left = &self.containers[lhs];
            let Some(right) = other.containers.get(rhs) else {
                return false;
            };
            match left.key.cmp(&right.key) {
                Ordering::Less => return false,
                Ordering::Greater => rhs += 1,
                Ordering::Equal => {
                    // Per-container cardinality precheck. Eliminates Bitmap ⊆ Array
                    // and Run ⊆ Array in the common canonical layout before any
                    // payload bytes are touched.
                    if left.cardinality > right.cardinality {
                        return false;
                    }
                    if !left.is_subset(self.bytes, right, other.bytes) {
                        return false;
                    }
                    lhs += 1;
                    rhs += 1;
                }
            }
        }
        true
    }

    /// Returns `true` if this view and `other` share at least one value.
    ///
    /// Equivalent to `!`[`RoaringBitmap::is_disjoint`] but short-circuits on the
    /// first overlapping element.
    pub fn intersects(&self, other: &Self) -> bool {
        let (Some(self_first), Some(self_last)) = (self.containers.first(), self.containers.last())
        else {
            return false;
        };
        let (Some(other_first), Some(other_last)) =
            (other.containers.first(), other.containers.last())
        else {
            return false;
        };
        // Key-range disjoint precheck: two u16 compares against cached endpoints,
        // no payload reads. Big win for queries against bitmaps with well-separated
        // key spaces (e.g. user-id buckets that never collide).
        if self_last.key < other_first.key || other_last.key < self_first.key {
            return false;
        }

        let mut lhs = 0;
        let mut rhs = 0;
        while lhs < self.containers.len() && rhs < other.containers.len() {
            let left = &self.containers[lhs];
            let right = &other.containers[rhs];
            match left.key.cmp(&right.key) {
                Ordering::Less => lhs += 1,
                Ordering::Greater => rhs += 1,
                Ordering::Equal => {
                    if left.intersects(self.bytes, right, other.bytes) {
                        return true;
                    }
                    lhs += 1;
                    rhs += 1;
                }
            }
        }
        false
    }

    /// Iterates over all values in ascending order.
    pub fn iter(&self) -> RoaringBitmapViewIter<'_, 'a> {
        RoaringBitmapViewIter { view: self, container_index: 0, inner: None, remaining: self.len }
    }

    /// Materializes the view into an owned [`RoaringBitmap`].
    pub fn to_owned(&self) -> RoaringBitmap {
        let containers = self
            .containers
            .iter()
            .map(|entry| Container { key: entry.key, store: entry.to_store(self.bytes) })
            .collect();
        RoaringBitmap { containers }
    }
}

impl ContainerEntry {
    fn payload<'a>(&self, bytes: &'a [u8], len: usize) -> &'a [u8] {
        &bytes[self.payload_offset..self.payload_offset + len]
    }

    fn array_payload<'a>(&self, bytes: &'a [u8]) -> &'a [u8] {
        self.payload(bytes, self.cardinality as usize * 2)
    }

    fn bitmap_payload<'a>(&self, bytes: &'a [u8]) -> &'a [u8] {
        self.payload(bytes, BITMAP_BYTES)
    }

    fn run_payload<'a>(&self, bytes: &'a [u8], runs: u16) -> &'a [u8] {
        self.payload(bytes, RUN_NUM_BYTES + usize::from(runs) * RUN_ELEMENT_BYTES)
    }

    fn contains(&self, bytes: &[u8], index: u16) -> bool {
        match self.kind {
            ContainerKind::Array => array_contains(self.array_payload(bytes), index),
            ContainerKind::Bitmap => bitmap_contains(self.bitmap_payload(bytes), index),
            ContainerKind::Run { runs } => run_contains(self.run_payload(bytes, runs), runs, index),
        }
    }

    fn min(&self, bytes: &[u8]) -> u16 {
        match self.kind {
            ContainerKind::Array => read_u16_at(self.array_payload(bytes), 0),
            ContainerKind::Bitmap => bitmap_min(self.bitmap_payload(bytes)),
            ContainerKind::Run { runs } => {
                let payload = self.run_payload(bytes, runs);
                read_u16_at(payload, RUN_NUM_BYTES)
            }
        }
    }

    fn max(&self, bytes: &[u8]) -> u16 {
        match self.kind {
            ContainerKind::Array => {
                read_u16_at(self.array_payload(bytes), (self.cardinality as usize - 1) * 2)
            }
            ContainerKind::Bitmap => bitmap_max(self.bitmap_payload(bytes)),
            ContainerKind::Run { runs } => {
                let offset = RUN_NUM_BYTES + (usize::from(runs) - 1) * RUN_ELEMENT_BYTES;
                let payload = self.run_payload(bytes, runs);
                let start = read_u16_at(payload, offset);
                let len = read_u16_at(payload, offset + 2);
                start + len
            }
        }
    }

    fn rank(&self, bytes: &[u8], index: u16) -> u64 {
        match self.kind {
            ContainerKind::Array => array_rank(self.array_payload(bytes), index),
            ContainerKind::Bitmap => bitmap_rank(self.bitmap_payload(bytes), index),
            ContainerKind::Run { runs } => run_rank(self.run_payload(bytes, runs), runs, index),
        }
    }

    fn intersection_len(&self, bytes: &[u8], other: &Self, other_bytes: &[u8]) -> u64 {
        match (self.kind, other.kind) {
            (ContainerKind::Array, ContainerKind::Array) => array_intersection_len_array(
                self.array_payload(bytes),
                other.array_payload(other_bytes),
            ),
            (ContainerKind::Array, ContainerKind::Bitmap) => array_intersection_len_bitmap(
                self.array_payload(bytes),
                other.bitmap_payload(other_bytes),
            ),
            (ContainerKind::Bitmap, ContainerKind::Array) => array_intersection_len_bitmap(
                other.array_payload(other_bytes),
                self.bitmap_payload(bytes),
            ),
            (ContainerKind::Bitmap, ContainerKind::Bitmap) => bitmap_intersection_len_bitmap(
                self.bitmap_payload(bytes),
                other.bitmap_payload(other_bytes),
            ),
            (ContainerKind::Run { runs }, ContainerKind::Run { runs: other_runs }) => {
                run_intersection_len_run(
                    self.run_payload(bytes, runs),
                    runs,
                    other.run_payload(other_bytes, other_runs),
                    other_runs,
                )
            }
            (ContainerKind::Run { runs }, ContainerKind::Array) => array_intersection_len_run(
                other.array_payload(other_bytes),
                self.run_payload(bytes, runs),
                runs,
            ),
            (ContainerKind::Array, ContainerKind::Run { runs }) => array_intersection_len_run(
                self.array_payload(bytes),
                other.run_payload(other_bytes, runs),
                runs,
            ),
            (ContainerKind::Run { runs }, ContainerKind::Bitmap) => run_intersection_len_bitmap(
                self.run_payload(bytes, runs),
                runs,
                other.bitmap_payload(other_bytes),
            ),
            (ContainerKind::Bitmap, ContainerKind::Run { runs }) => run_intersection_len_bitmap(
                other.run_payload(other_bytes, runs),
                runs,
                self.bitmap_payload(bytes),
            ),
        }
    }

    fn is_subset(&self, bytes: &[u8], other: &Self, other_bytes: &[u8]) -> bool {
        match (self.kind, other.kind) {
            (ContainerKind::Array, ContainerKind::Array) => array_is_subset_of_array(
                self.array_payload(bytes),
                other.array_payload(other_bytes),
            ),
            (ContainerKind::Array, ContainerKind::Bitmap) => array_is_subset_of_bitmap(
                self.array_payload(bytes),
                other.bitmap_payload(other_bytes),
            ),
            (ContainerKind::Array, ContainerKind::Run { runs }) => array_is_subset_of_run(
                self.array_payload(bytes),
                other.run_payload(other_bytes, runs),
                runs,
            ),
            // Mirror RoaringBitmap::is_subset: a Bitmap-shaped container is never
            // a subset of an Array-shaped container, regardless of cardinality.
            // Keeps view/owned semantics in lock-step so `view == view.to_owned()`.
            (ContainerKind::Bitmap, ContainerKind::Array) => false,
            (ContainerKind::Bitmap, ContainerKind::Bitmap) => bitmap_is_subset_of_bitmap(
                self.bitmap_payload(bytes),
                other.bitmap_payload(other_bytes),
            ),
            (ContainerKind::Bitmap, ContainerKind::Run { runs }) => bitmap_is_subset_of_run(
                self.bitmap_payload(bytes),
                other.run_payload(other_bytes, runs),
                runs,
            ),
            (ContainerKind::Run { runs }, ContainerKind::Array) => run_is_subset_of_array(
                self.run_payload(bytes, runs),
                runs,
                other.array_payload(other_bytes),
            ),
            (ContainerKind::Run { runs }, ContainerKind::Bitmap) => run_is_subset_of_bitmap(
                self.run_payload(bytes, runs),
                runs,
                other.bitmap_payload(other_bytes),
            ),
            (ContainerKind::Run { runs: left_runs }, ContainerKind::Run { runs: right_runs }) => {
                run_is_subset_of_run(
                    self.run_payload(bytes, left_runs),
                    left_runs,
                    other.run_payload(other_bytes, right_runs),
                    right_runs,
                )
            }
        }
    }

    fn intersects(&self, bytes: &[u8], other: &Self, other_bytes: &[u8]) -> bool {
        match (self.kind, other.kind) {
            (ContainerKind::Array, ContainerKind::Array) => {
                array_intersects_array(self.array_payload(bytes), other.array_payload(other_bytes))
            }
            (ContainerKind::Array, ContainerKind::Bitmap) => array_intersects_bitmap(
                self.array_payload(bytes),
                other.bitmap_payload(other_bytes),
            ),
            (ContainerKind::Bitmap, ContainerKind::Array) => array_intersects_bitmap(
                other.array_payload(other_bytes),
                self.bitmap_payload(bytes),
            ),
            (ContainerKind::Bitmap, ContainerKind::Bitmap) => bitmap_intersects_bitmap(
                self.bitmap_payload(bytes),
                other.bitmap_payload(other_bytes),
            ),
            (ContainerKind::Array, ContainerKind::Run { runs }) => array_intersects_run(
                self.array_payload(bytes),
                other.run_payload(other_bytes, runs),
                runs,
            ),
            (ContainerKind::Run { runs }, ContainerKind::Array) => array_intersects_run(
                other.array_payload(other_bytes),
                self.run_payload(bytes, runs),
                runs,
            ),
            (ContainerKind::Bitmap, ContainerKind::Run { runs }) => bitmap_intersects_run(
                self.bitmap_payload(bytes),
                other.run_payload(other_bytes, runs),
                runs,
            ),
            (ContainerKind::Run { runs }, ContainerKind::Bitmap) => bitmap_intersects_run(
                other.bitmap_payload(other_bytes),
                self.run_payload(bytes, runs),
                runs,
            ),
            (ContainerKind::Run { runs: left_runs }, ContainerKind::Run { runs: right_runs }) => {
                run_intersects_run(
                    self.run_payload(bytes, left_runs),
                    left_runs,
                    other.run_payload(other_bytes, right_runs),
                    right_runs,
                )
            }
        }
    }

    fn payload_bytes<'a>(&self, bytes: &'a [u8]) -> &'a [u8] {
        match self.kind {
            ContainerKind::Array => self.array_payload(bytes),
            ContainerKind::Bitmap => self.bitmap_payload(bytes),
            ContainerKind::Run { runs } => self.run_payload(bytes, runs),
        }
    }

    fn iter<'a>(&self, bytes: &'a [u8]) -> ViewContainerIter<'a> {
        let inner = match self.kind {
            ContainerKind::Array => ViewContainerValues::Array {
                payload: self.array_payload(bytes),
                pos: 0,
                end: self.cardinality as usize,
            },
            ContainerKind::Bitmap => ViewContainerValues::Bitmap {
                payload: self.bitmap_payload(bytes),
                next_word: 0,
                current_word: 0,
                current_base: 0,
            },
            ContainerKind::Run { runs } => ViewContainerValues::Run {
                payload: self.run_payload(bytes, runs),
                run_index: 0,
                run_count: runs as usize,
                current: None,
            },
        };
        ViewContainerIter { key: self.key, inner }
    }

    fn to_store(self, bytes: &[u8]) -> Store {
        match self.kind {
            ContainerKind::Array => {
                let payload = self.array_payload(bytes);
                let values = (0..self.cardinality as usize)
                    .map(|index| read_u16_at(payload, index * 2))
                    .collect();
                Store::Array(ArrayStore::from_vec_unchecked(values))
            }
            ContainerKind::Bitmap => {
                let payload = self.bitmap_payload(bytes);
                let mut bits = Box::new([0u64; BITMAP_LENGTH]);
                debug_assert_eq!(payload.len(), BITMAP_BYTES);
                // SAFETY: validate_bitmap has ensured payload.len() == BITMAP_BYTES, which
                // equals size_of::<[u64; BITMAP_LENGTH]>(). The destination is a freshly
                // allocated Box so the regions cannot overlap. A &mut [u64] is always at
                // least 8-byte aligned, satisfying the alignment requirement on its u8 view.
                unsafe {
                    core::ptr::copy_nonoverlapping(
                        payload.as_ptr(),
                        bits.as_mut_ptr().cast::<u8>(),
                        BITMAP_BYTES,
                    );
                }
                #[cfg(target_endian = "big")]
                for word in bits.iter_mut() {
                    *word = word.swap_bytes();
                }
                Store::Bitmap(BitmapStore::from_unchecked(u64::from(self.cardinality), bits))
            }
            ContainerKind::Run { runs } => {
                let payload = self.run_payload(bytes, runs);
                let intervals = (0..usize::from(runs))
                    .map(|index| {
                        let offset = RUN_NUM_BYTES + index * RUN_ELEMENT_BYTES;
                        let start = read_u16_at(payload, offset);
                        let len = read_u16_at(payload, offset + 2);
                        Interval::new_unchecked(start, start + len)
                    })
                    .collect();
                Store::Run(IntervalStore::from_vec_unchecked(intervals))
            }
        }
    }
}

impl PartialEq for RoaringBitmapView<'_> {
    fn eq(&self, other: &Self) -> bool {
        if self.len != other.len {
            return false;
        }
        if self.containers.len() != other.containers.len() {
            return false;
        }
        // Fast path: when every container has the same key/cardinality/kind,
        // each payload has a canonical byte layout and set equality reduces
        // to byte equality of payloads. Both bytea inputs to PG `rb_equals`
        // typically come from the same serializer with the same optimize()
        // pass, so this path hits in the common case and reduces the whole
        // comparison to one memcmp per container.
        let same_layout = self.containers.iter().zip(&other.containers).all(|(left, right)| {
            left.key == right.key
                && left.cardinality == right.cardinality
                && left.kind == right.kind
        });
        if same_layout {
            return self.containers.iter().zip(&other.containers).all(|(left, right)| {
                left.payload_bytes(self.bytes) == right.payload_bytes(other.bytes)
            });
        }
        // Slow path: cross-source / cross-shape comparison falls back to set
        // equality. With lengths already equal, `self.is_subset(other)` is iff
        // set equality.
        self.is_subset(other)
    }
}

impl Eq for RoaringBitmapView<'_> {}

impl Iterator for RoaringBitmapViewIter<'_, '_> {
    type Item = u32;

    fn next(&mut self) -> Option<u32> {
        loop {
            if let Some(inner) = &mut self.inner {
                if let Some(value) = inner.next() {
                    self.remaining -= 1;
                    return Some(value);
                }
                self.inner = None;
            }

            let entry = self.view.containers.get(self.container_index)?;
            self.container_index += 1;
            self.inner = Some(entry.iter(self.view.bytes));
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = usize::try_from(self.remaining).unwrap_or(usize::MAX);
        (remaining, Some(remaining))
    }
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            ParseError::Truncated { offset, needed, got } => {
                write!(f, "input too short: needed {needed} bytes at offset {offset}, got {got}")
            }
            ParseError::BadCookie { got } => {
                write!(
                    f,
                    "bad cookie: expected {:#x} or {:#x}, got {got:#x}",
                    SERIAL_COOKIE_NO_RUNCONTAINER, SERIAL_COOKIE
                )
            }
            ParseError::TooManyContainers { count, max } => {
                write!(f, "container count overflow: {count} containers exceeds maximum {max}")
            }
            ParseError::InvalidRun { offset, detail } => {
                write!(f, "invalid run container at offset {offset}: {detail}")
            }
            ParseError::InvalidContainer { offset, detail } => {
                write!(f, "invalid container at offset {offset}: {detail}")
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for ParseError {}

impl Iterator for ViewContainerIter<'_> {
    type Item = u32;

    fn next(&mut self) -> Option<u32> {
        self.inner.next().map(|index| util::join(self.key, index))
    }
}

impl Iterator for ViewContainerValues<'_> {
    type Item = u16;

    fn next(&mut self) -> Option<u16> {
        match self {
            ViewContainerValues::Array { payload, pos, end } => {
                if *pos == *end {
                    return None;
                }
                let value = read_u16_at(payload, *pos * 2);
                *pos += 1;
                Some(value)
            }
            ViewContainerValues::Bitmap { payload, next_word, current_word, current_base } => {
                loop {
                    if *current_word != 0 {
                        let bit = current_word.trailing_zeros() as u16;
                        *current_word &= *current_word - 1;
                        return Some(*current_base + bit);
                    }
                    if *next_word == BITMAP_LENGTH {
                        return None;
                    }
                    *current_base = (*next_word as u16) * 64;
                    *current_word = read_u64_at(payload, *next_word * 8);
                    *next_word += 1;
                }
            }
            ViewContainerValues::Run { payload, run_index, run_count, current } => loop {
                if let Some((value, end)) = *current {
                    if value == end {
                        *current = None;
                    } else {
                        *current = Some((value + 1, end));
                    }
                    return Some(value);
                }
                if *run_index == *run_count {
                    return None;
                }
                let offset = RUN_NUM_BYTES + *run_index * RUN_ELEMENT_BYTES;
                let start = read_u16_at(payload, offset);
                let len = read_u16_at(payload, offset + 2);
                *run_index += 1;
                *current = Some((start, start + len));
            },
        }
    }
}

fn validate_container_count(count: u64) -> Result<(), ParseError> {
    let max = max_containers();
    if count > max {
        Err(ParseError::TooManyContainers { count, max })
    } else {
        Ok(())
    }
}

fn max_containers() -> u64 {
    u64::from(u16::MAX) + 1
}

fn checked_payload(bytes: &[u8], offset: usize, needed: usize) -> Result<(), ParseError> {
    let got = bytes.len().saturating_sub(offset);
    if got < needed {
        Err(ParseError::Truncated { offset, needed, got })
    } else {
        Ok(())
    }
}

fn take<'a>(bytes: &'a [u8], offset: &mut usize, needed: usize) -> Result<&'a [u8], ParseError> {
    checked_payload(bytes, *offset, needed)?;
    let start = *offset;
    *offset += needed;
    Ok(&bytes[start..*offset])
}

fn read_u32(bytes: &[u8], offset: &mut usize) -> Result<u32, ParseError> {
    let raw = take(bytes, offset, 4)?;
    Ok(u32::from_le_bytes(raw.try_into().unwrap()))
}

fn read_u16_at_abs(bytes: &[u8], offset: usize) -> Result<u16, ParseError> {
    checked_payload(bytes, offset, 2)?;
    Ok(read_u16_at(bytes, offset))
}

fn read_u16_at(bytes: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes(bytes[offset..offset + 2].try_into().unwrap())
}

fn read_u32_at(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
}

fn read_u64_at(bytes: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap())
}

fn validate_array(bytes: &[u8], offset: usize, cardinality: u32) -> Result<(), ParseError> {
    let len = cardinality as usize * 2;
    checked_payload(bytes, offset, len)?;

    let mut last = None;
    for index in 0..cardinality as usize {
        let value = read_u16_at(bytes, offset + index * 2);
        if last.replace(value).is_some_and(|last| value <= last) {
            return Err(ParseError::InvalidContainer {
                offset: offset + index * 2,
                detail: "array container values are not sorted and unique",
            });
        }
    }
    Ok(())
}

fn validate_bitmap(bytes: &[u8], offset: usize, cardinality: u32) -> Result<(), ParseError> {
    checked_payload(bytes, offset, BITMAP_BYTES)?;
    // count_ones is endian-invariant, so from_ne_bytes (free on all targets) is
    // sufficient. chunks_exact lets LLVM hoist the slice bounds invariant.
    let actual = bytes[offset..offset + BITMAP_BYTES]
        .chunks_exact(8)
        .map(|chunk| u64::from_ne_bytes(chunk.try_into().unwrap()).count_ones())
        .sum::<u32>();
    if actual == cardinality {
        Ok(())
    } else {
        Err(ParseError::InvalidContainer {
            offset,
            detail: "bitmap container cardinality does not match payload",
        })
    }
}

fn validate_run(
    bytes: &[u8],
    offset: usize,
    runs: u16,
    cardinality: u32,
) -> Result<(), ParseError> {
    let byte_size = RUN_NUM_BYTES + usize::from(runs) * RUN_ELEMENT_BYTES;
    checked_payload(bytes, offset, byte_size)?;

    let mut last_end = None;
    let mut actual_cardinality = 0u32;
    for index in 0..usize::from(runs) {
        let run_offset = offset + RUN_NUM_BYTES + index * RUN_ELEMENT_BYTES;
        let start = read_u16_at(bytes, run_offset);
        let len = read_u16_at(bytes, run_offset + 2);
        let end = start.checked_add(len).ok_or(ParseError::InvalidRun {
            offset: run_offset,
            detail: "run length overflows container",
        })?;
        if let Some(last_end) = last_end.replace(end) {
            if start <= last_end.saturating_add(1) {
                return Err(ParseError::InvalidRun {
                    offset: run_offset,
                    detail: "runs overlap or are contiguous",
                });
            }
        }
        actual_cardinality += u32::from(len) + 1;
    }

    if actual_cardinality == cardinality {
        Ok(())
    } else {
        Err(ParseError::InvalidRun {
            offset,
            detail: "run container cardinality does not match payload",
        })
    }
}

fn array_contains(payload: &[u8], index: u16) -> bool {
    array_binary_search(payload, index).is_ok()
}

fn array_rank(payload: &[u8], index: u16) -> u64 {
    match array_binary_search(payload, index) {
        Ok(pos) => pos as u64 + 1,
        Err(pos) => pos as u64,
    }
}

fn array_binary_search(payload: &[u8], index: u16) -> Result<usize, usize> {
    let len = payload.len() / 2;
    let mut size = len;
    let mut left = 0;
    while size > 0 {
        let half = size / 2;
        let mid = left + half;
        match read_u16_at(payload, mid * 2).cmp(&index) {
            Ordering::Less => {
                left = mid + 1;
                size -= half + 1;
            }
            Ordering::Equal => return Ok(mid),
            Ordering::Greater => size = half,
        }
    }
    Err(left)
}

fn array_intersection_len_array(left: &[u8], right: &[u8]) -> u64 {
    let mut left_index = 0;
    let mut right_index = 0;
    let left_len = left.len() / 2;
    let right_len = right.len() / 2;
    let mut total = 0;

    while left_index < left_len && right_index < right_len {
        let left_value = read_u16_at(left, left_index * 2);
        let right_value = read_u16_at(right, right_index * 2);
        match left_value.cmp(&right_value) {
            Ordering::Less => left_index += 1,
            Ordering::Greater => right_index += 1,
            Ordering::Equal => {
                total += 1;
                left_index += 1;
                right_index += 1;
            }
        }
    }

    total
}

fn array_intersection_len_bitmap(array: &[u8], bitmap: &[u8]) -> u64 {
    let len = array.len() / 2;
    (0..len).map(|index| u64::from(bitmap_contains(bitmap, read_u16_at(array, index * 2)))).sum()
}

fn array_intersection_len_run(array: &[u8], run: &[u8], runs: u16) -> u64 {
    let len = array.len() / 2;
    (0..len).map(|index| u64::from(run_contains(run, runs, read_u16_at(array, index * 2)))).sum()
}

fn bitmap_contains(payload: &[u8], index: u16) -> bool {
    let word_index = usize::from(index / 64);
    let bit = index % 64;
    read_u64_at(payload, word_index * 8) & (1 << bit) != 0
}

fn bitmap_min(payload: &[u8]) -> u16 {
    for index in 0..BITMAP_LENGTH {
        let word = read_u64_at(payload, index * 8);
        if word != 0 {
            return index as u16 * 64 + word.trailing_zeros() as u16;
        }
    }
    unreachable!("bitmap containers are validated as non-empty")
}

fn bitmap_max(payload: &[u8]) -> u16 {
    for index in (0..BITMAP_LENGTH).rev() {
        let word = read_u64_at(payload, index * 8);
        if word != 0 {
            return index as u16 * 64 + (u64::BITS - 1 - word.leading_zeros()) as u16;
        }
    }
    unreachable!("bitmap containers are validated as non-empty")
}

fn bitmap_rank(payload: &[u8], index: u16) -> u64 {
    let word_index = usize::from(index / 64);
    let bit = index % 64;
    // One up-front slice bounds-check + chunks_exact(8) gives LLVM the invariant it
    // needs to autovectorize the popcount loop (PSADBW on x86_64, popcount lanes on
    // AArch64). `u64::from_le_bytes` is a no-op on little-endian targets.
    let full_words: u64 = payload[..word_index * 8]
        .chunks_exact(8)
        .map(|chunk| u64::from_le_bytes(chunk.try_into().unwrap()).count_ones() as u64)
        .sum();
    let mask = if bit == 63 { u64::MAX } else { (1u64 << (bit + 1)) - 1 };
    full_words + u64::from((read_u64_at(payload, word_index * 8) & mask).count_ones())
}

fn bitmap_intersection_len_bitmap(left: &[u8], right: &[u8]) -> u64 {
    left[..BITMAP_BYTES]
        .chunks_exact(8)
        .zip(right[..BITMAP_BYTES].chunks_exact(8))
        .map(|(l, r)| {
            let l = u64::from_le_bytes(l.try_into().unwrap());
            let r = u64::from_le_bytes(r.try_into().unwrap());
            (l & r).count_ones() as u64
        })
        .sum()
}

fn bitmap_intersection_len_range(payload: &[u8], start: u16, end: u16) -> u64 {
    let start_word = usize::from(start / 64);
    let end_word = usize::from(end / 64);
    let start_bit = start % 64;
    let end_bit = end % 64;

    if start_word == end_word {
        let mask = range_mask(start_bit, end_bit);
        return u64::from((read_u64_at(payload, start_word * 8) & mask).count_ones());
    }

    let first_mask = u64::MAX << start_bit;
    let last_mask = if end_bit == 63 { u64::MAX } else { (1u64 << (end_bit + 1)) - 1 };
    let mut total = u64::from((read_u64_at(payload, start_word * 8) & first_mask).count_ones());

    // Same chunks_exact pattern as bitmap_rank for the middle stretch of full words.
    if end_word > start_word + 1 {
        let mid_start = (start_word + 1) * 8;
        let mid_end = end_word * 8;
        total += payload[mid_start..mid_end]
            .chunks_exact(8)
            .map(|chunk| u64::from_le_bytes(chunk.try_into().unwrap()).count_ones() as u64)
            .sum::<u64>();
    }

    total + u64::from((read_u64_at(payload, end_word * 8) & last_mask).count_ones())
}

fn range_mask(start_bit: u16, end_bit: u16) -> u64 {
    let start_mask = u64::MAX << start_bit;
    let end_mask = if end_bit == 63 { u64::MAX } else { (1u64 << (end_bit + 1)) - 1 };
    start_mask & end_mask
}

fn run_contains(payload: &[u8], runs: u16, index: u16) -> bool {
    run_search(payload, runs, index).is_ok()
}

fn run_rank(payload: &[u8], runs: u16, index: u16) -> u64 {
    let mut rank = 0;
    for run_index in 0..usize::from(runs) {
        let offset = RUN_NUM_BYTES + run_index * RUN_ELEMENT_BYTES;
        let start = read_u16_at(payload, offset);
        let len = read_u16_at(payload, offset + 2);
        let end = start + len;
        if end <= index {
            rank += u64::from(len) + 1;
        } else if start <= index {
            rank += u64::from(index - start) + 1;
            break;
        } else {
            break;
        }
    }
    rank
}

fn run_intersection_len_run(left: &[u8], left_runs: u16, right: &[u8], right_runs: u16) -> u64 {
    let mut left_index = 0;
    let mut right_index = 0;
    let mut total = 0;

    while left_index < usize::from(left_runs) && right_index < usize::from(right_runs) {
        let (left_start, left_end) = run_bounds(left, left_index);
        let (right_start, right_end) = run_bounds(right, right_index);

        let start = left_start.max(right_start);
        let end = left_end.min(right_end);
        if start <= end {
            total += u64::from(end - start) + 1;
        }

        match left_end.cmp(&right_end) {
            Ordering::Less => left_index += 1,
            Ordering::Greater => right_index += 1,
            Ordering::Equal => {
                left_index += 1;
                right_index += 1;
            }
        }
    }

    total
}

fn run_intersection_len_bitmap(run: &[u8], runs: u16, bitmap: &[u8]) -> u64 {
    (0..usize::from(runs))
        .map(|index| {
            let (start, end) = run_bounds(run, index);
            bitmap_intersection_len_range(bitmap, start, end)
        })
        .sum()
}

// ---------------------------------------------------------------------------
// Predicate kernels: is_subset / intersects
//
// Naming mirrors the cardinality-kernel convention above. `is_subset_of`
// kernels are asymmetric in argument order: left ⊆ right. `intersects`
// kernels are symmetric; the dispatcher above passes the smaller container
// first when symmetry helps the inner loop, but all kernels are correct for
// either ordering.
// ---------------------------------------------------------------------------

fn array_is_subset_of_array(left: &[u8], right: &[u8]) -> bool {
    let left_len = left.len() / 2;
    let right_len = right.len() / 2;
    let mut li = 0;
    let mut ri = 0;
    while li < left_len {
        // Pigeonhole: not enough right values remain to cover all left values.
        if right_len - ri < left_len - li {
            return false;
        }
        let lv = read_u16_at(left, li * 2);
        loop {
            if ri == right_len {
                return false;
            }
            match read_u16_at(right, ri * 2).cmp(&lv) {
                Ordering::Less => ri += 1,
                Ordering::Equal => {
                    ri += 1;
                    break;
                }
                Ordering::Greater => return false,
            }
        }
        li += 1;
    }
    true
}

fn array_is_subset_of_bitmap(array: &[u8], bitmap: &[u8]) -> bool {
    let len = array.len() / 2;
    for index in 0..len {
        if !bitmap_contains(bitmap, read_u16_at(array, index * 2)) {
            return false;
        }
    }
    true
}

fn array_is_subset_of_run(array: &[u8], run: &[u8], runs: u16) -> bool {
    let len = array.len() / 2;
    let run_count = usize::from(runs);
    let mut run_index = 0usize;
    for index in 0..len {
        let value = read_u16_at(array, index * 2);
        // Monotone interval cursor: advance past runs strictly before `value`.
        while run_index < run_count {
            let (start, end) = run_bounds(run, run_index);
            if value <= end {
                if value < start {
                    return false;
                }
                break;
            }
            run_index += 1;
        }
        if run_index == run_count {
            return false;
        }
    }
    true
}

fn bitmap_is_subset_of_bitmap(left: &[u8], right: &[u8]) -> bool {
    // chunks_exact(8) gives LLVM the constant-length invariant needed to
    // autovectorize the AND-NOT loop (AVX2: 4 u64s/iter; AVX-512: 8). `.all()`
    // short-circuits at the iterator boundary.
    left[..BITMAP_BYTES].chunks_exact(8).zip(right[..BITMAP_BYTES].chunks_exact(8)).all(|(l, r)| {
        let l = u64::from_le_bytes(l.try_into().unwrap());
        let r = u64::from_le_bytes(r.try_into().unwrap());
        l & !r == 0
    })
}

fn bitmap_is_subset_of_run(bitmap: &[u8], run: &[u8], runs: u16) -> bool {
    // Walk the bitmap's set bits, advancing a monotone run cursor. Short-circuit
    // on the first set bit that falls outside every run.
    let run_count = usize::from(runs);
    let mut run_index = 0usize;
    for word_index in 0..BITMAP_LENGTH {
        let mut word = read_u64_at(bitmap, word_index * 8);
        while word != 0 {
            let bit = word.trailing_zeros() as u16;
            let value = (word_index as u16) * 64 + bit;
            while run_index < run_count {
                let (start, end) = run_bounds(run, run_index);
                if value <= end {
                    if value < start {
                        return false;
                    }
                    break;
                }
                run_index += 1;
            }
            if run_index == run_count {
                return false;
            }
            word &= word - 1;
        }
    }
    true
}

fn run_is_subset_of_array(run: &[u8], runs: u16, array: &[u8]) -> bool {
    // Algebraic trick: for run [s, e] of length L = e - s, binary-search `s` in
    // the sorted-unique array. If `array[pos] == s` and `array[pos + L] == e`
    // then array[pos..=pos+L] are L+1 distinct sorted u16 values in [s, e],
    // which must be exactly s, s+1, ..., e (the interval is fully covered).
    // Avoids touching the L-1 interior values entirely.
    let array_len = array.len() / 2;
    let run_count = usize::from(runs);
    for run_index in 0..run_count {
        let (start, end) = run_bounds(run, run_index);
        let len = usize::from(end - start);
        let pos = match array_binary_search(array, start) {
            Ok(p) => p,
            Err(_) => return false,
        };
        if pos + len >= array_len {
            return false;
        }
        if read_u16_at(array, (pos + len) * 2) != end {
            return false;
        }
    }
    true
}

fn run_is_subset_of_bitmap(run: &[u8], runs: u16, bitmap: &[u8]) -> bool {
    let run_count = usize::from(runs);
    for run_index in 0..run_count {
        let (start, end) = run_bounds(run, run_index);
        if !bitmap_contains_range(bitmap, start, end) {
            return false;
        }
    }
    true
}

fn run_is_subset_of_run(left: &[u8], left_runs: u16, right: &[u8], right_runs: u16) -> bool {
    // Two-pointer interval merge: for each left run, advance the right cursor
    // until the right run's end reaches or exceeds the left run's end. If the
    // right run also starts at or before the left run's start, it covers; else
    // not-subset. Mirrors `interval_store::is_subset`.
    let left_count = usize::from(left_runs);
    let right_count = usize::from(right_runs);
    let mut ri = 0usize;
    for li in 0..left_count {
        let (ls, le) = run_bounds(left, li);
        while ri < right_count && run_bounds(right, ri).1 < le {
            ri += 1;
        }
        if ri == right_count {
            return false;
        }
        if run_bounds(right, ri).0 > ls {
            return false;
        }
    }
    true
}

fn array_intersects_array(left: &[u8], right: &[u8]) -> bool {
    let mut li = 0;
    let mut ri = 0;
    let left_len = left.len() / 2;
    let right_len = right.len() / 2;
    while li < left_len && ri < right_len {
        match read_u16_at(left, li * 2).cmp(&read_u16_at(right, ri * 2)) {
            Ordering::Less => li += 1,
            Ordering::Greater => ri += 1,
            Ordering::Equal => return true,
        }
    }
    false
}

fn array_intersects_bitmap(array: &[u8], bitmap: &[u8]) -> bool {
    let len = array.len() / 2;
    for index in 0..len {
        if bitmap_contains(bitmap, read_u16_at(array, index * 2)) {
            return true;
        }
    }
    false
}

fn array_intersects_run(array: &[u8], run: &[u8], runs: u16) -> bool {
    let len = array.len() / 2;
    let run_count = usize::from(runs);
    let mut run_index = 0usize;
    for index in 0..len {
        let value = read_u16_at(array, index * 2);
        while run_index < run_count {
            let (start, end) = run_bounds(run, run_index);
            if value < start {
                break;
            }
            if value <= end {
                return true;
            }
            run_index += 1;
        }
        if run_index == run_count {
            return false;
        }
    }
    false
}

fn bitmap_intersects_bitmap(left: &[u8], right: &[u8]) -> bool {
    left[..BITMAP_BYTES].chunks_exact(8).zip(right[..BITMAP_BYTES].chunks_exact(8)).any(|(l, r)| {
        let l = u64::from_le_bytes(l.try_into().unwrap());
        let r = u64::from_le_bytes(r.try_into().unwrap());
        l & r != 0
    })
}

fn bitmap_intersects_run(bitmap: &[u8], run: &[u8], runs: u16) -> bool {
    let run_count = usize::from(runs);
    for run_index in 0..run_count {
        let (start, end) = run_bounds(run, run_index);
        if bitmap_intersects_range(bitmap, start, end) {
            return true;
        }
    }
    false
}

fn run_intersects_run(left: &[u8], left_runs: u16, right: &[u8], right_runs: u16) -> bool {
    let left_count = usize::from(left_runs);
    let right_count = usize::from(right_runs);
    let mut li = 0;
    let mut ri = 0;
    while li < left_count && ri < right_count {
        let (ls, le) = run_bounds(left, li);
        let (rs, re) = run_bounds(right, ri);
        if le < rs {
            li += 1;
        } else if re < ls {
            ri += 1;
        } else {
            return true;
        }
    }
    false
}

fn bitmap_contains_range(payload: &[u8], start: u16, end: u16) -> bool {
    let start_word = usize::from(start / 64);
    let end_word = usize::from(end / 64);
    let start_bit = start % 64;
    let end_bit = end % 64;

    if start_word == end_word {
        let mask = range_mask(start_bit, end_bit);
        return read_u64_at(payload, start_word * 8) & mask == mask;
    }

    let first_mask = u64::MAX << start_bit;
    if read_u64_at(payload, start_word * 8) & first_mask != first_mask {
        return false;
    }
    if end_word > start_word + 1 {
        let mid_start = (start_word + 1) * 8;
        let mid_end = end_word * 8;
        let all_full = payload[mid_start..mid_end]
            .chunks_exact(8)
            .all(|chunk| u64::from_le_bytes(chunk.try_into().unwrap()) == u64::MAX);
        if !all_full {
            return false;
        }
    }
    let last_mask = if end_bit == 63 { u64::MAX } else { (1u64 << (end_bit + 1)) - 1 };
    read_u64_at(payload, end_word * 8) & last_mask == last_mask
}

fn bitmap_intersects_range(payload: &[u8], start: u16, end: u16) -> bool {
    let start_word = usize::from(start / 64);
    let end_word = usize::from(end / 64);
    let start_bit = start % 64;
    let end_bit = end % 64;

    if start_word == end_word {
        let mask = range_mask(start_bit, end_bit);
        return read_u64_at(payload, start_word * 8) & mask != 0;
    }

    let first_mask = u64::MAX << start_bit;
    if read_u64_at(payload, start_word * 8) & first_mask != 0 {
        return true;
    }
    if end_word > start_word + 1 {
        let mid_start = (start_word + 1) * 8;
        let mid_end = end_word * 8;
        let any_nonzero = payload[mid_start..mid_end]
            .chunks_exact(8)
            .any(|chunk| u64::from_le_bytes(chunk.try_into().unwrap()) != 0);
        if any_nonzero {
            return true;
        }
    }
    let last_mask = if end_bit == 63 { u64::MAX } else { (1u64 << (end_bit + 1)) - 1 };
    read_u64_at(payload, end_word * 8) & last_mask != 0
}

fn run_bounds(payload: &[u8], index: usize) -> (u16, u16) {
    let offset = RUN_NUM_BYTES + index * RUN_ELEMENT_BYTES;
    let start = read_u16_at(payload, offset);
    let end = start + read_u16_at(payload, offset + 2);
    (start, end)
}

fn run_search(payload: &[u8], runs: u16, index: u16) -> Result<usize, usize> {
    let mut size = usize::from(runs);
    let mut left = 0;
    while size > 0 {
        let half = size / 2;
        let mid = left + half;
        let offset = RUN_NUM_BYTES + mid * RUN_ELEMENT_BYTES;
        let start = read_u16_at(payload, offset);
        let end = start + read_u16_at(payload, offset + 2);
        if index < start {
            size = half;
        } else if index > end {
            left = mid + 1;
            size -= half + 1;
        } else {
            return Ok(mid);
        }
    }
    Err(left)
}

#[cfg(all(test, feature = "std"))]
mod test {
    use super::*;
    use proptest::collection::{btree_set, vec};
    use proptest::prelude::*;

    fn view_bytes(bitmap: &RoaringBitmap) -> Vec<u8> {
        let mut bytes = Vec::new();
        bitmap.serialize_into(&mut bytes).unwrap();
        bytes
    }

    fn assert_cardinality_ops(left: &RoaringBitmap, right: &RoaringBitmap) {
        let left_bytes = view_bytes(left);
        let right_bytes = view_bytes(right);
        let left_view = RoaringBitmapView::try_new(&left_bytes).unwrap();
        let right_view = RoaringBitmapView::try_new(&right_bytes).unwrap();

        assert_eq!(left_view.intersection_len(&right_view), (left & right).len());
        assert_eq!(left_view.union_len(&right_view), (left | right).len());
        assert_eq!(left_view.difference_len(&right_view), (left - right).len());
        assert_eq!(left_view.symmetric_difference_len(&right_view), (left ^ right).len());
        assert_eq!(left_view.xor_len(&right_view), (left ^ right).len());
    }

    fn assert_predicate_ops(left: &RoaringBitmap, right: &RoaringBitmap) {
        let left_bytes = view_bytes(left);
        let right_bytes = view_bytes(right);
        let left_view = RoaringBitmapView::try_new(&left_bytes).unwrap();
        let right_view = RoaringBitmapView::try_new(&right_bytes).unwrap();

        assert_eq!(left_view.is_subset(&right_view), left.is_subset(right));
        assert_eq!(right_view.is_subset(&left_view), right.is_subset(left));
        assert_eq!(left_view.intersects(&right_view), !left.is_disjoint(right));
        assert_eq!(right_view.intersects(&left_view), !right.is_disjoint(left));
        assert_eq!(left_view == right_view, left == right);
    }

    #[test]
    fn reads_array_bitmap_and_run_containers() {
        let mut bitmap = RoaringBitmap::from_iter([1, 3, 7, 65_535, 70_000]);
        bitmap.insert_range(200_000..210_000);
        bitmap.insert_range(300_000..301_000);
        bitmap.optimize();

        let bytes = view_bytes(&bitmap);
        let view = RoaringBitmapView::try_new(&bytes).unwrap();

        assert_eq!(view.len(), bitmap.len());
        assert_eq!(view.is_empty(), bitmap.is_empty());
        assert_eq!(view.min(), bitmap.min());
        assert_eq!(view.max(), bitmap.max());
        assert_eq!(view.to_owned(), bitmap);
        assert_eq!(view.iter().collect::<Vec<_>>(), bitmap.iter().collect::<Vec<_>>());

        for value in [0, 1, 7, 65_535, 70_000, 200_000, 209_999, 210_000, u32::MAX] {
            assert_eq!(view.contains(value), bitmap.contains(value));
            assert_eq!(view.rank(value), bitmap.rank(value));
        }
    }

    #[test]
    fn cardinality_ops_match_owned_bitmap_container_pairs() {
        let array = RoaringBitmap::from_iter([1, 3, 7, 65, 4_095, 70_001, 70_003]);
        let mut bitmap = RoaringBitmap::from_iter(2_000..7_500);
        bitmap.insert_range(70_000..75_000);

        let mut run = RoaringBitmap::new();
        run.insert_range(3_000..8_000);
        run.insert_range(70_002..70_010);
        run.optimize();

        let mut disjoint_key = RoaringBitmap::new();
        disjoint_key.insert_range(200_000..205_000);

        for (left, right) in [
            (&array, &array),
            (&array, &bitmap),
            (&bitmap, &array),
            (&bitmap, &bitmap),
            (&array, &run),
            (&run, &array),
            (&bitmap, &run),
            (&run, &bitmap),
            (&run, &run),
            (&run, &disjoint_key),
        ] {
            assert_cardinality_ops(left, right);
            assert_predicate_ops(left, right);
        }
    }

    #[test]
    fn eq_is_set_equality_across_layouts() {
        // Same set in two different on-disk shapes: Bitmap (4_500 contiguous
        // values keeps it in BitmapStore) and Run (after optimize). PartialEq
        // must report equal via the set-equality slow path; the byte-compare
        // fast path is intentionally bypassed by the layout mismatch.
        let canonical = RoaringBitmap::from_iter(2_000..6_500);
        let mut runs = canonical.clone();
        runs.optimize();

        let canonical_bytes = view_bytes(&canonical);
        let runs_bytes = view_bytes(&runs);
        let canonical_view = RoaringBitmapView::try_new(&canonical_bytes).unwrap();
        let runs_view = RoaringBitmapView::try_new(&runs_bytes).unwrap();

        assert_ne!(canonical_bytes, runs_bytes); // sanity: the bytes really differ
        assert_eq!(canonical_view, runs_view);

        // Identical bytes hit the layout-match memcmp fast path.
        let canonical_view_again = RoaringBitmapView::try_new(&canonical_bytes).unwrap();
        assert_eq!(canonical_view, canonical_view_again);

        // Length short-circuit: differ by a single element.
        let smaller = RoaringBitmap::from_iter(2_001..6_500);
        let smaller_bytes = view_bytes(&smaller);
        let smaller_view = RoaringBitmapView::try_new(&smaller_bytes).unwrap();
        assert_ne!(canonical_view, smaller_view);
    }

    #[test]
    fn rejects_bad_cookie() {
        assert!(matches!(
            RoaringBitmapView::try_new(&[1, 2, 3, 4]),
            Err(ParseError::BadCookie { got: 0x0403_0201 })
        ));
    }

    #[test]
    fn rejects_truncated_payload() {
        let bitmap = RoaringBitmap::from_iter(0..5000);
        let mut bytes = view_bytes(&bitmap);
        bytes.truncate(bytes.len() - 1);
        assert!(matches!(RoaringBitmapView::try_new(&bytes), Err(ParseError::Truncated { .. })));
    }

    proptest! {
        #[test]
        fn matches_owned_bitmap(
            values in btree_set(0u32..=300_000, 0usize..=20_000),
            other_values in btree_set(0u32..=300_000, 0usize..=20_000),
            checks in vec(0u32..=300_000, 0usize..=256),
            optimize in any::<bool>(),
            other_optimize in any::<bool>(),
        ) {
            let mut bitmap = RoaringBitmap::from_sorted_iter(values.iter().copied()).unwrap();
            if optimize {
                bitmap.optimize();
            }
            let mut other = RoaringBitmap::from_sorted_iter(other_values.iter().copied()).unwrap();
            if other_optimize {
                other.optimize();
            }
            let bytes = view_bytes(&bitmap);
            let view = RoaringBitmapView::try_new(&bytes).unwrap();
            let other_bytes = view_bytes(&other);
            let other_view = RoaringBitmapView::try_new(&other_bytes).unwrap();

            prop_assert_eq!(view.len(), bitmap.len());
            prop_assert_eq!(view.is_empty(), bitmap.is_empty());
            prop_assert_eq!(view.min(), bitmap.min());
            prop_assert_eq!(view.max(), bitmap.max());
            prop_assert_eq!(view.iter().collect::<Vec<_>>(), bitmap.iter().collect::<Vec<_>>());
            prop_assert_eq!(view.to_owned(), bitmap.clone());

            for value in checks {
                prop_assert_eq!(view.contains(value), bitmap.contains(value));
                prop_assert_eq!(view.rank(value), bitmap.rank(value));
            }

            prop_assert_eq!(view.intersection_len(&other_view), bitmap.intersection_len(&other));
            prop_assert_eq!(view.union_len(&other_view), bitmap.union_len(&other));
            prop_assert_eq!(view.difference_len(&other_view), bitmap.difference_len(&other));
            prop_assert_eq!(
                view.symmetric_difference_len(&other_view),
                bitmap.symmetric_difference_len(&other)
            );
            prop_assert_eq!(view.xor_len(&other_view), bitmap.symmetric_difference_len(&other));

            prop_assert_eq!(view.is_subset(&other_view), bitmap.is_subset(&other));
            prop_assert_eq!(other_view.is_subset(&view), other.is_subset(&bitmap));
            prop_assert_eq!(view.intersects(&other_view), !bitmap.is_disjoint(&other));
            prop_assert_eq!(other_view.intersects(&view), !other.is_disjoint(&bitmap));
            prop_assert_eq!(view == other_view, bitmap == other);
        }
    }
}
