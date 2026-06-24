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
    let full_words =
        (0..word_index).map(|word| read_u64_at(payload, word * 8).count_ones() as u64).sum::<u64>();
    let mask = if bit == 63 { u64::MAX } else { (1u64 << (bit + 1)) - 1 };
    full_words + u64::from((read_u64_at(payload, word_index * 8) & mask).count_ones())
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
            checks in vec(0u32..=300_000, 0usize..=256),
            optimize in any::<bool>(),
        ) {
            let mut bitmap = RoaringBitmap::from_sorted_iter(values.iter().copied()).unwrap();
            if optimize {
                bitmap.optimize();
            }
            let bytes = view_bytes(&bitmap);
            let view = RoaringBitmapView::try_new(&bytes).unwrap();

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
        }
    }
}
