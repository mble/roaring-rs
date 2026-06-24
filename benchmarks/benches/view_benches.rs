//! Performance guardrails for `RoaringBitmapView`.
//!
//! Three Criterion groups (`view_to_owned`, `view_rank`, `view_parse`) measure
//! the hot paths against two complementary fixture sources:
//!
//! - **Synthetic** (`*/synthetic/*`): deterministic fixtures with no network
//!   dependency, so the bench also runs under `cargo test -p benchmarks
//!   --benches` in CI without needing the dataset clone.
//! - **Real** (`*/<dataset>`): the `real-roaring-datasets` corpus shared with
//!   `benches/lib.rs`, providing workload-representative cardinality
//!   distributions.
//!
//! Local regression check (run on base, switch branches, run on PR):
//! ```
//! cargo bench --bench view_benches -- --save-baseline pre
//! # ...apply changes...
//! cargo bench --bench view_benches -- --baseline pre
//! ```
//! Criterion prints the percentage delta per benchmark. The audit set a 10%
//! regression budget for `view_to_owned/synthetic/dense_bitmap`,
//! `view_rank/synthetic/*`, and `view_parse/synthetic/bitmap_heavy`.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

use roaring::{RoaringBitmap, RoaringBitmapView};

use crate::datasets::Datasets;

mod datasets;

const CONTAINER_BASE: u32 = 1 << 16;

/// Bitmap whose containers all materialize as bitmap-kind (`cardinality > 4096`).
fn build_bitmap_heavy(num_containers: u16) -> RoaringBitmap {
    let mut bitmap = RoaringBitmap::new();
    for key in 0..num_containers {
        let base = u32::from(key) * CONTAINER_BASE;
        // Every 3rd value in the [0, 65536) sub-range gives ~21845 entries — well over
        // ARRAY_LIMIT (4096), forcing a bitmap container.
        for offset in (0..u32::from(u16::MAX) + 1).step_by(3) {
            bitmap.insert(base + offset);
        }
    }
    bitmap
}

/// Bitmap whose containers all materialize as array-kind (`cardinality <= 4096`).
fn build_array_heavy(num_containers: u16) -> RoaringBitmap {
    let mut bitmap = RoaringBitmap::new();
    for key in 0..num_containers {
        let base = u32::from(key) * CONTAINER_BASE;
        // 1000 values per container, spaced — stays well under ARRAY_LIMIT.
        for offset in (0..10_000u32).step_by(10) {
            bitmap.insert(base + offset);
        }
    }
    bitmap
}

/// Bitmap optimized into run-kind containers.
fn build_run_heavy(num_containers: u16) -> RoaringBitmap {
    let mut bitmap = RoaringBitmap::new();
    for key in 0..num_containers {
        let base = u32::from(key) * CONTAINER_BASE;
        bitmap.insert_range(base..base + 30_000);
        bitmap.insert_range(base + 35_000..base + 60_000);
    }
    bitmap.optimize();
    bitmap
}

/// Bitmap with a deterministic mix of all three container kinds.
fn build_mixed(num_containers: u16) -> RoaringBitmap {
    let mut bitmap = RoaringBitmap::new();
    for key in 0..num_containers {
        let base = u32::from(key) * CONTAINER_BASE;
        match key % 3 {
            0 => {
                for offset in (0..1000u32).step_by(10) {
                    bitmap.insert(base + offset);
                }
            }
            1 => {
                for offset in (0..65_000u32).step_by(3) {
                    bitmap.insert(base + offset);
                }
            }
            _ => {
                bitmap.insert_range(base..base + 50_000);
            }
        }
    }
    bitmap.optimize();
    bitmap
}

fn serialize(bitmap: &RoaringBitmap) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(bitmap.serialized_size());
    bitmap.serialize_into(&mut bytes).unwrap();
    bytes
}

fn view_to_owned(c: &mut Criterion) {
    let mut group = c.benchmark_group("view_to_owned");

    let dense_bytes = serialize(&build_bitmap_heavy(64));
    let array_bytes = serialize(&build_array_heavy(64));
    let run_bytes = serialize(&build_run_heavy(64));

    let dense_view = RoaringBitmapView::try_new(&dense_bytes).unwrap();
    let array_view = RoaringBitmapView::try_new(&array_bytes).unwrap();
    let run_view = RoaringBitmapView::try_new(&run_bytes).unwrap();

    group.bench_function(BenchmarkId::new("synthetic", "dense_bitmap"), |b| {
        b.iter(|| black_box(dense_view.to_owned()))
    });
    group.bench_function(BenchmarkId::new("synthetic", "sparse_array"), |b| {
        b.iter(|| black_box(array_view.to_owned()))
    });
    group.bench_function(BenchmarkId::new("synthetic", "runs"), |b| {
        b.iter(|| black_box(run_view.to_owned()))
    });

    group.finish();
}

fn view_rank(c: &mut Criterion) {
    let mut group = c.benchmark_group("view_rank");

    for &num_containers in &[16u16, 1024, 4096] {
        let bitmap = build_mixed(num_containers);
        let bytes = serialize(&bitmap);
        let view = RoaringBitmapView::try_new(&bytes).unwrap();

        // 1024 probes spread across the populated key space — exercises the binary
        // search over containers and the prefix-cardinality lookup uniformly.
        let max = view.max().unwrap_or(0);
        let stride = (max / 1024).max(1);
        let probes: Vec<u32> = (0..1024u32).map(|i| i.saturating_mul(stride)).collect();

        let id = BenchmarkId::new("synthetic", format!("{num_containers}_containers"));
        group.bench_function(id, |b| {
            b.iter(|| {
                for &probe in &probes {
                    black_box(view.rank(probe));
                }
            });
        });
    }

    group.finish();
}

fn view_parse(c: &mut Criterion) {
    let mut group = c.benchmark_group("view_parse");

    let dense_bytes = serialize(&build_bitmap_heavy(64));
    let array_bytes = serialize(&build_array_heavy(64));
    let run_bytes = serialize(&build_run_heavy(64));

    group.bench_function(BenchmarkId::new("synthetic", "bitmap_heavy"), |b| {
        b.iter(|| black_box(RoaringBitmapView::try_new(&dense_bytes).unwrap()));
    });
    group.bench_function(BenchmarkId::new("synthetic", "array_heavy"), |b| {
        b.iter(|| black_box(RoaringBitmapView::try_new(&array_bytes).unwrap()));
    });
    group.bench_function(BenchmarkId::new("synthetic", "run_heavy"), |b| {
        b.iter(|| black_box(RoaringBitmapView::try_new(&run_bytes).unwrap()));
    });

    group.finish();
}

/// Serialize every bitmap in a dataset into a vec of bytes (one per bitmap).
fn dataset_bytes(dataset: &datasets::Dataset) -> Vec<Vec<u8>> {
    dataset
        .bitmaps
        .iter()
        .map(|bitmap| {
            let mut buf = Vec::with_capacity(bitmap.serialized_size());
            bitmap.serialize_into(&mut buf).unwrap();
            buf
        })
        .collect()
}

fn view_to_owned_real(c: &mut Criterion) {
    let mut group = c.benchmark_group("view_to_owned");

    for dataset in Datasets {
        let inputs = dataset_bytes(dataset);
        let views: Vec<RoaringBitmapView<'_>> =
            inputs.iter().map(|buf| RoaringBitmapView::try_new(buf).unwrap()).collect();

        group.throughput(Throughput::Elements(dataset.bitmaps.iter().map(|rb| rb.len()).sum()));
        group.bench_function(BenchmarkId::new("to_owned", &dataset.name), |b| {
            b.iter(|| {
                for view in &views {
                    black_box(view.to_owned());
                }
            });
        });
    }

    group.finish();
}

fn view_rank_real(c: &mut Criterion) {
    let mut group = c.benchmark_group("view_rank");

    for dataset in Datasets {
        let inputs = dataset_bytes(dataset);
        let views: Vec<RoaringBitmapView<'_>> =
            inputs.iter().map(|buf| RoaringBitmapView::try_new(buf).unwrap()).collect();

        // Match lib.rs::rank — probe every 100th value below the cardinality.
        // Capping probes at len() (rather than max()) avoids degenerating into
        // a len() benchmark for any sparse bitmap.
        let probes: Vec<Vec<u32>> = views
            .iter()
            .map(|view| (0..view.len() as u32).step_by(100).collect::<Vec<u32>>())
            .collect();

        group.throughput(Throughput::Elements(probes.iter().map(|p| p.len() as u64).sum()));
        group.bench_function(BenchmarkId::new("rank", &dataset.name), |b| {
            b.iter(|| {
                for (view, probes) in views.iter().zip(probes.iter()) {
                    for &probe in probes {
                        black_box(view.rank(probe));
                    }
                }
            });
        });
    }

    group.finish();
}

fn view_parse_real(c: &mut Criterion) {
    let mut group = c.benchmark_group("view_parse");

    for dataset in Datasets {
        let inputs = dataset_bytes(dataset);

        group.throughput(Throughput::Bytes(inputs.iter().map(|b| b.len() as u64).sum()));
        group.bench_function(BenchmarkId::new("try_new", &dataset.name), |b| {
            b.iter(|| {
                for buf in &inputs {
                    black_box(RoaringBitmapView::try_new(buf).unwrap());
                }
            });
        });
    }

    group.finish();
}

criterion_group!(
    view_benches,
    view_to_owned,
    view_rank,
    view_parse,
    view_to_owned_real,
    view_rank_real,
    view_parse_real,
);
criterion_main!(view_benches);
