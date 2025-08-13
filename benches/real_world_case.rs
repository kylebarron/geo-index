use bytemuck::cast_slice;
use criterion::{criterion_group, criterion_main, Criterion};
use geo_index::rtree::sort::{HilbertSort, STRSort};
use geo_index::rtree::util::f64_box_to_f32;
use geo_index::rtree::{RTree, RTreeBuilder, RTreeIndex};
use geo_index::IndexableNum;
use once_cell::sync::OnceCell;
use rand::seq::SliceRandom;
use rand::thread_rng;
use std::fs::read;
use std::time::Duration;

/// Dataset metadata
struct Dataset {
    name: &'static str,
    file_path: &'static str,
}

/// Available datasets
const BUILDINGS: Dataset = Dataset {
    name: "buildings",
    file_path: "benches/data/buildings-boxes.raw",
};

const NODES: Dataset = Dataset {
    name: "nodes",
    file_path: "benches/data/osm-nodes-boxes.raw",
};

const POSTAL_CODES: Dataset = Dataset {
    name: "postal-codes",
    file_path: "benches/data/postal-codes-boxes.raw",
};

const POSTAL_CODES_CA: Dataset = Dataset {
    name: "postal-codes-ca",
    file_path: "benches/data/postal-codes-ca-boxes.raw",
};

/// Benchmark configurations: (index_dataset, query_dataset, name_suffix)
const BENCHMARK_CONFIGS: &[BenchmarkConfig] = &[
    BenchmarkConfig {
        index_dataset: &BUILDINGS,
        query_dataset: &NODES,
        name_suffix: "buildings_nodes",
        queries_per_iter: 1000,
    },
    BenchmarkConfig {
        index_dataset: &POSTAL_CODES,
        query_dataset: &BUILDINGS,
        name_suffix: "postal_codes_buildings",
        queries_per_iter: 1000,
    },
    BenchmarkConfig {
        index_dataset: &BUILDINGS,
        query_dataset: &POSTAL_CODES_CA,
        name_suffix: "buildings_postal_codes",
        queries_per_iter: 1000,
    },
];

struct BenchmarkConfig {
    index_dataset: &'static Dataset,
    query_dataset: &'static Dataset,
    name_suffix: &'static str,
    queries_per_iter: usize,
}

struct WrapAroundQueryBoxIter<'a> {
    query_boxes: &'a [(f64, f64, f64, f64)],
    current_idx: usize,
}

impl<'a> WrapAroundQueryBoxIter<'a> {
    fn new(query_boxes: &'a [(f64, f64, f64, f64)]) -> Self {
        assert!(!query_boxes.is_empty());
        Self {
            query_boxes,
            current_idx: 0,
        }
    }

    fn next(&mut self) -> (f64, f64, f64, f64) {
        let result = self.query_boxes[self.current_idx];
        self.current_idx += 1;
        if self.current_idx >= self.query_boxes.len() {
            self.current_idx = 0;
        }
        result
    }
}

/// Load bounding box data from a raw file
fn load_dataset(dataset: &Dataset) -> Vec<f64> {
    let buf =
        read(dataset.file_path).unwrap_or_else(|_| panic!("Failed to read {}", dataset.file_path));
    cast_slice(&buf).to_vec()
}

/// Construct an RTree with Hilbert sorting
fn construct_rtree_hilbert<N: IndexableNum>(boxes_buf: &[N]) -> RTree<N> {
    let mut builder = RTreeBuilder::new((boxes_buf.len() / 4) as _);
    for box_ in boxes_buf.chunks(4) {
        let min_x = box_[0];
        let min_y = box_[1];
        let max_x = box_[2];
        let max_y = box_[3];
        builder.add(min_x, min_y, max_x, max_y);
    }
    builder.finish::<HilbertSort>()
}

/// Construct an RTree with STR sorting
fn construct_rtree_str<N: IndexableNum>(boxes_buf: &[N]) -> RTree<N> {
    let mut builder = RTreeBuilder::new((boxes_buf.len() / 4) as _);
    for box_ in boxes_buf.chunks(4) {
        let min_x = box_[0];
        let min_y = box_[1];
        let max_x = box_[2];
        let max_y = box_[3];
        builder.add(min_x, min_y, max_x, max_y);
    }
    builder.finish::<STRSort>()
}

/// Construct an RTree with f32 precision using Hilbert sorting
fn construct_rtree_hilbert_f32_with_cast(boxes_buf: &[f64]) -> RTree<f32> {
    let mut builder = RTreeBuilder::new((boxes_buf.len() / 4) as _);
    for box_ in boxes_buf.chunks(4) {
        let min_x = box_[0];
        let min_y = box_[1];
        let max_x = box_[2];
        let max_y = box_[3];
        let (min_x, min_y, max_x, max_y) = f64_box_to_f32(min_x, min_y, max_x, max_y);
        builder.add(min_x, min_y, max_x, max_y);
    }
    builder.finish::<HilbertSort>()
}

/// Construct an RTree with f32 precision using STR sorting
fn construct_rtree_str_f32_with_cast(boxes_buf: &[f64]) -> RTree<f32> {
    let mut builder = RTreeBuilder::new((boxes_buf.len() / 4) as _);
    for box_ in boxes_buf.chunks(4) {
        let min_x = box_[0];
        let min_y = box_[1];
        let max_x = box_[2];
        let max_y = box_[3];
        let (min_x, min_y, max_x, max_y) = f64_box_to_f32(min_x, min_y, max_x, max_y);
        builder.add(min_x, min_y, max_x, max_y);
    }
    builder.finish::<STRSort>()
}

/// Convert dataset to query boxes (use all boxes)
fn dataset_to_query_boxes(boxes_buf: &[f64]) -> Vec<(f64, f64, f64, f64)> {
    boxes_buf
        .chunks(4)
        .map(|box_| (box_[0], box_[1], box_[2], box_[3]))
        .collect()
}

pub fn benchmark_construction(c: &mut Criterion) {
    for config in BENCHMARK_CONFIGS {
        let index_dataset = config.index_dataset;
        let index_data = load_dataset(index_dataset);

        // Construction benchmarks
        c.bench_function(
            &format!("construction_{}_hilbert_f64", index_dataset.name),
            |b| b.iter(|| construct_rtree_hilbert(&index_data)),
        );

        c.bench_function(
            &format!("construction_{}_str_f64", index_dataset.name),
            |b| b.iter(|| construct_rtree_str(&index_data)),
        );

        c.bench_function(
            &format!("construction_{}_hilbert_f32", index_dataset.name),
            |b| b.iter(|| construct_rtree_hilbert_f32_with_cast(&index_data)),
        );

        c.bench_function(
            &format!("construction_{}_str_f32", index_dataset.name),
            |b| b.iter(|| construct_rtree_str_f32_with_cast(&index_data)),
        );
    }
}

pub fn benchmark_search(c: &mut Criterion) {
    for config in BENCHMARK_CONFIGS {
        let index_dataset = config.index_dataset;
        let query_dataset = config.query_dataset;
        let name_suffix = config.name_suffix;
        let queries_per_iter = config.queries_per_iter;

        let index_data = load_dataset(index_dataset);
        let query_data = load_dataset(query_dataset);
        let mut query_boxes = dataset_to_query_boxes(&query_data);
        let mut rng = thread_rng();
        query_boxes.shuffle(&mut rng);
        let mut query_boxes_iter = WrapAroundQueryBoxIter::new(&query_boxes);

        // Search benchmarks with OnceCell for lazy, one-time index construction
        let hilbert_f64_index: OnceCell<RTree<f64>> = OnceCell::new();
        c.bench_function(&format!("search_{name_suffix}_hilbert_f64"), |b| {
            b.iter(|| {
                let index = hilbert_f64_index.get_or_init(|| construct_rtree_hilbert(&index_data));
                for _ in 0..queries_per_iter {
                    let (min_x, min_y, max_x, max_y) = query_boxes_iter.next();
                    index.search(min_x, min_y, max_x, max_y);
                }
            })
        });

        let str_f64_index: OnceCell<RTree<f64>> = OnceCell::new();
        c.bench_function(&format!("search_{name_suffix}_str_f64"), |b| {
            b.iter(|| {
                let index = str_f64_index.get_or_init(|| construct_rtree_str(&index_data));
                for _ in 0..queries_per_iter {
                    let (min_x, min_y, max_x, max_y) = query_boxes_iter.next();
                    index.search(min_x, min_y, max_x, max_y);
                }
            })
        });

        let hilbert_f32_index: OnceCell<RTree<f32>> = OnceCell::new();
        c.bench_function(&format!("search_{name_suffix}_hilbert_f32"), |b| {
            b.iter(|| {
                let index = hilbert_f32_index
                    .get_or_init(|| construct_rtree_hilbert_f32_with_cast(&index_data));
                for _ in 0..queries_per_iter {
                    let (min_x, min_y, max_x, max_y) = query_boxes_iter.next();
                    let (min_x_f32, min_y_f32, max_x_f32, max_y_f32) =
                        f64_box_to_f32(min_x, min_y, max_x, max_y);
                    index.search(min_x_f32, min_y_f32, max_x_f32, max_y_f32);
                }
            })
        });

        let str_f32_index: OnceCell<RTree<f32>> = OnceCell::new();
        c.bench_function(&format!("search_{name_suffix}_str_f32"), |b| {
            b.iter(|| {
                let index =
                    str_f32_index.get_or_init(|| construct_rtree_str_f32_with_cast(&index_data));
                for _ in 0..queries_per_iter {
                    let (min_x, min_y, max_x, max_y) = query_boxes_iter.next();
                    let (min_x_f32, min_y_f32, max_x_f32, max_y_f32) =
                        f64_box_to_f32(min_x, min_y, max_x, max_y);
                    index.search(min_x_f32, min_y_f32, max_x_f32, max_y_f32);
                }
            })
        });
    }
}

criterion_group! {
    name = benches_search;
    config = Criterion::default()
        .measurement_time(Duration::from_secs(10))
        .warm_up_time(Duration::from_secs(3));
    targets = benchmark_search
}

criterion_group! {
    name = benches_construction;
    config = Criterion::default()
        .measurement_time(Duration::from_secs(10))
        .sample_size(10)
        .warm_up_time(Duration::from_secs(3));
    targets = benchmark_construction
}

criterion_main!(benches_search, benches_construction);
