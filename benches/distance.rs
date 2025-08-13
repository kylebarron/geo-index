use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use geo::{Geometry, LineString, Point, Polygon};
use geo_index::rtree::distance::{
    DistanceMetric, EuclideanDistance, HaversineDistance, SpheroidDistance,
};
use geo_index::rtree::sort::HilbertSort;
use geo_index::rtree::{RTreeBuilder, RTreeIndex};
use geo_types::coord;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

fn generate_test_data(n: usize) -> (Vec<Point<f64>>, Vec<Geometry<f64>>) {
    let mut rng = StdRng::seed_from_u64(42);
    let mut points = Vec::with_capacity(n);
    let mut geometries = Vec::with_capacity(n);

    for i in 0..n {
        let x = rng.gen_range(-180.0..180.0);
        let y = rng.gen_range(-90.0..90.0);
        let point = Point::new(x, y);
        points.push(point);

        // Create a mix of geometry types
        let geom = match i % 3 {
            0 => Geometry::Point(point),
            1 => {
                // Create a small line around the point
                let offset = 0.01;
                Geometry::LineString(LineString::new(vec![
                    coord! { x: x, y: y },
                    coord! { x: x + offset, y: y + offset },
                ]))
            }
            2 => {
                // Create a small square around the point
                let offset = 0.005;
                Geometry::Polygon(Polygon::new(
                    LineString::new(vec![
                        coord! { x: x - offset, y: y - offset },
                        coord! { x: x + offset, y: y - offset },
                        coord! { x: x + offset, y: y + offset },
                        coord! { x: x - offset, y: y + offset },
                        coord! { x: x - offset, y: y - offset },
                    ]),
                    vec![],
                ))
            }
            _ => unreachable!(),
        };
        geometries.push(geom);
    }

    (points, geometries)
}

fn build_rtree(points: &[Point<f64>]) -> geo_index::rtree::RTree<f64> {
    let mut builder = RTreeBuilder::new(points.len() as u32);
    for point in points {
        builder.add(point.x(), point.y(), point.x(), point.y());
    }
    builder.finish::<HilbertSort>()
}

fn benchmark_distance_metrics(c: &mut Criterion) {
    let sizes = vec![100, 1000];

    for size in sizes {
        let (points, geometries) = generate_test_data(size);
        let tree = build_rtree(&points);
        let query_point = Point::new(0.0, 0.0);

        let euclidean = EuclideanDistance;
        let haversine = HaversineDistance::default();
        let spheroid = SpheroidDistance::default();

        // Benchmark neighbors_with_distance with different metrics
        let mut group = c.benchmark_group("neighbors_with_distance");

        group.bench_with_input(BenchmarkId::new("euclidean", size), &size, |b, _| {
            b.iter(|| {
                tree.neighbors_with_distance(
                    query_point.x(),
                    query_point.y(),
                    Some(10),
                    None,
                    &euclidean,
                )
            })
        });

        group.bench_with_input(BenchmarkId::new("haversine", size), &size, |b, _| {
            b.iter(|| {
                tree.neighbors_with_distance(
                    query_point.x(),
                    query_point.y(),
                    Some(10),
                    None,
                    &haversine,
                )
            })
        });

        group.bench_with_input(BenchmarkId::new("spheroid", size), &size, |b, _| {
            b.iter(|| {
                tree.neighbors_with_distance(
                    query_point.x(),
                    query_point.y(),
                    Some(10),
                    None,
                    &spheroid,
                )
            })
        });

        group.finish();

        // Benchmark neighbors_geometry with different metrics
        let mut geom_group = c.benchmark_group("neighbors_geometry");
        let query_geometry = Geometry::Point(query_point);

        geom_group.bench_with_input(BenchmarkId::new("euclidean", size), &size, |b, _| {
            b.iter(|| {
                tree.neighbors_geometry(&query_geometry, Some(10), None, &euclidean, &geometries)
            })
        });

        geom_group.bench_with_input(BenchmarkId::new("haversine", size), &size, |b, _| {
            b.iter(|| {
                tree.neighbors_geometry(&query_geometry, Some(10), None, &haversine, &geometries)
            })
        });

        geom_group.bench_with_input(BenchmarkId::new("spheroid", size), &size, |b, _| {
            b.iter(|| {
                tree.neighbors_geometry(&query_geometry, Some(10), None, &spheroid, &geometries)
            })
        });

        geom_group.finish();
    }
}

fn benchmark_distance_calculations(c: &mut Criterion) {
    let euclidean = EuclideanDistance;
    let haversine = HaversineDistance::default();
    let spheroid = SpheroidDistance::default();

    let p1 = Point::new(-74.0, 40.7); // New York
    let p2 = Point::new(-0.1, 51.5); // London
    let geom1 = Geometry::Point(p1);
    let geom2 = Geometry::Point(p2);

    // Benchmark raw distance calculations
    let mut group = c.benchmark_group("distance_calculation");

    group.bench_function("euclidean_point_to_point", |b| {
        b.iter(|| euclidean.distance(p1.x(), p1.y(), p2.x(), p2.y()))
    });

    group.bench_function("haversine_point_to_point", |b| {
        b.iter(|| haversine.distance(p1.x(), p1.y(), p2.x(), p2.y()))
    });

    group.bench_function("spheroid_point_to_point", |b| {
        b.iter(|| spheroid.distance(p1.x(), p1.y(), p2.x(), p2.y()))
    });

    group.bench_function("euclidean_geometry_to_geometry", |b| {
        b.iter(|| {
            let _: f64 = euclidean.geometry_to_geometry_distance(&geom1, &geom2);
        })
    });

    group.bench_function("haversine_geometry_to_geometry", |b| {
        b.iter(|| {
            let _: f64 = haversine.geometry_to_geometry_distance(&geom1, &geom2);
        })
    });

    group.bench_function("spheroid_geometry_to_geometry", |b| {
        b.iter(|| {
            let _: f64 = spheroid.geometry_to_geometry_distance(&geom1, &geom2);
        })
    });

    // Benchmark bbox distance calculations
    let bbox = (-75.0, 40.0, -73.0, 41.0); // Bbox around New York

    group.bench_function("euclidean_distance_to_bbox", |b| {
        b.iter(|| euclidean.distance_to_bbox(p2.x(), p2.y(), bbox.0, bbox.1, bbox.2, bbox.3))
    });

    group.bench_function("haversine_distance_to_bbox", |b| {
        b.iter(|| haversine.distance_to_bbox(p2.x(), p2.y(), bbox.0, bbox.1, bbox.2, bbox.3))
    });

    group.bench_function("spheroid_distance_to_bbox", |b| {
        b.iter(|| spheroid.distance_to_bbox(p2.x(), p2.y(), bbox.0, bbox.1, bbox.2, bbox.3))
    });

    group.finish();
}

fn benchmark_comparison_with_baseline(c: &mut Criterion) {
    let (points, _) = generate_test_data(1000);
    let tree = build_rtree(&points);
    let query_point = Point::new(0.0, 0.0);

    let euclidean = EuclideanDistance;

    let mut group = c.benchmark_group("baseline_comparison");

    // Original neighbors method (baseline)
    group.bench_function("original_neighbors", |b| {
        b.iter(|| tree.neighbors(query_point.x(), query_point.y(), Some(10), None))
    });

    // New neighbors_with_distance method
    group.bench_function("neighbors_with_distance_euclidean", |b| {
        b.iter(|| {
            tree.neighbors_with_distance(
                query_point.x(),
                query_point.y(),
                Some(10),
                None,
                &euclidean,
            )
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    benchmark_distance_metrics,
    benchmark_distance_calculations,
    benchmark_comparison_with_baseline
);
criterion_main!(benches);
