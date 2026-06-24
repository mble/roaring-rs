mod arbitrary;
mod container;
mod fmt;
mod multiops;
mod proptests;
mod statistics;
mod store;
mod util;

// Order of these modules matters as it determines the `impl` blocks order in
// the docs
mod cmp;
mod inherent;
mod iter;
mod ops;
#[cfg(feature = "std")]
mod ops_with_serialized;
#[cfg(feature = "serde")]
mod serde;
#[cfg(feature = "std")]
mod serialization;
mod view;

use self::cmp::Pairs;
pub use self::iter::IntoIter;
pub use self::iter::Iter;
pub use self::statistics::Statistics;
pub use self::view::{ParseError, RoaringBitmapView, RoaringBitmapViewIter};

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

/// A compressed bitmap using the [Roaring bitmap compression scheme](https://roaringbitmap.org/).
///
/// # Examples
///
/// ```rust
/// use roaring::RoaringBitmap;
///
/// let mut rb = RoaringBitmap::new();
///
/// // insert all primes less than 10
/// rb.insert(2);
/// rb.insert(3);
/// rb.insert(5);
/// rb.insert(7);
/// println!("total bits set to true: {}", rb.len());
/// ```
#[derive(PartialEq, Eq)]
pub struct RoaringBitmap {
    containers: Vec<container::Container>,
}

pub(crate) const SERIAL_COOKIE_NO_RUNCONTAINER: u32 = 12346;
pub(crate) const SERIAL_COOKIE: u16 = 12347;
pub(crate) const NO_OFFSET_THRESHOLD: usize = 4;

// Sizes of header structures
#[cfg(feature = "std")]
pub(crate) const COOKIE_BYTES: usize = 4;
#[cfg(feature = "std")]
pub(crate) const SIZE_BYTES: usize = 4;
pub(crate) const DESCRIPTION_BYTES: usize = 4;
pub(crate) const OFFSET_BYTES: usize = 4;
