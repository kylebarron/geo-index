//! Example demonstrating different distance metrics for spatial queries.
//!
//! This example shows how to use Euclidean, Haversine, and Spheroid distance metrics
//! for finding nearest neighbors in spatial datasets.

use geo_index::rtree::distance::{EuclideanDistance, HaversineDistance, SpheroidDistance};
use geo_index::rtree::sort::HilbertSort;
use geo_index::rtree::{RTreeBuilder, RTreeIndex};

fn main() {
    println!("=== Distance Metrics Example ===\n");

    // Example 1: Euclidean Distance (for planar coordinates)
    println!("1. Euclidean Distance (planar coordinates):");
    euclidean_distance_example();

    // Example 2: Haversine Distance (for geographic coordinates)
    println!("\n2. Haversine Distance (geographic coordinates):");
    haversine_distance_example();

    // Example 3: Spheroid Distance (for high-precision geographic coordinates)
    println!("\n3. Spheroid Distance (high-precision geographic coordinates):");
    spheroid_distance_example();

    // Example 4: Comparison of different metrics
    println!("\n4. Comparison of different distance metrics:");
    comparison_example();
}

fn euclidean_distance_example() {
    let mut builder = RTreeBuilder::<f64>::new(4);

    // Add some points in a planar coordinate system
    builder.add(0., 0., 1., 1.); // Point A
    builder.add(3., 4., 4., 5.); // Point B
    builder.add(6., 8., 7., 9.); // Point C
    builder.add(1., 1., 2., 2.); // Point D

    let tree = builder.finish::<HilbertSort>();

    let euclidean = EuclideanDistance;
    let query_point = (0., 0.);

    println!("  Query point: {:?}", query_point);
    let results =
        tree.neighbors_with_distance(query_point.0, query_point.1, Some(3), None, &euclidean);

    println!("  Nearest neighbors (by insertion order): {:?}", results);
    println!("  Distance metric: Euclidean (straight-line distance)");
}

fn haversine_distance_example() {
    let mut builder = RTreeBuilder::<f64>::new(5);

    // Add some major cities (longitude, latitude)
    builder.add(-74.0, 40.7, -74.0, 40.7); // New York
    builder.add(-0.1, 51.5, -0.1, 51.5); // London
    builder.add(139.7, 35.7, 139.7, 35.7); // Tokyo
    builder.add(-118.2, 34.1, -118.2, 34.1); // Los Angeles
    builder.add(2.3, 48.9, 2.3, 48.9); // Paris

    let tree = builder.finish::<HilbertSort>();

    let haversine = HaversineDistance::default();
    let query_point = (-74.0, 40.7); // New York

    println!("  Query point: New York {:?}", query_point);
    let results =
        tree.neighbors_with_distance(query_point.0, query_point.1, Some(3), None, &haversine);

    println!("  Nearest neighbors (by insertion order): {:?}", results);
    println!("  Distance metric: Haversine (great-circle distance on sphere)");
    println!("  Earth radius: {} meters", haversine.earth_radius);
}

fn spheroid_distance_example() {
    let mut builder = RTreeBuilder::<f64>::new(5);

    // Add some major cities (longitude, latitude)
    builder.add(-74.0, 40.7, -74.0, 40.7); // New York
    builder.add(-0.1, 51.5, -0.1, 51.5); // London
    builder.add(139.7, 35.7, 139.7, 35.7); // Tokyo
    builder.add(-118.2, 34.1, -118.2, 34.1); // Los Angeles
    builder.add(2.3, 48.9, 2.3, 48.9); // Paris

    let tree = builder.finish::<HilbertSort>();

    let spheroid = SpheroidDistance::default(); // WGS84 ellipsoid
    let query_point = (-74.0, 40.7); // New York

    println!("  Query point: New York {:?}", query_point);
    let results =
        tree.neighbors_with_distance(query_point.0, query_point.1, Some(3), None, &spheroid);

    println!("  Nearest neighbors (by insertion order): {:?}", results);
    println!("  Distance metric: Spheroid (distance on ellipsoid)");
    println!("  Semi-major axis: {} meters", spheroid.semi_major_axis);
    println!("  Semi-minor axis: {} meters", spheroid.semi_minor_axis);
}

fn comparison_example() {
    let mut builder = RTreeBuilder::<f64>::new(3);

    // Add points with different characteristics
    builder.add(0., 0., 1., 1.); // Origin
    builder.add(1., 1., 2., 2.); // Close point
    builder.add(10., 10., 11., 11.); // Distant point

    let tree = builder.finish::<HilbertSort>();

    let query_point = (0., 0.);

    // Test with different distance metrics
    let euclidean = EuclideanDistance;
    let haversine = HaversineDistance::default();
    let spheroid = SpheroidDistance::default();

    println!("  Query point: {:?}", query_point);

    let euclidean_results =
        tree.neighbors_with_distance(query_point.0, query_point.1, Some(2), None, &euclidean);
    println!("  Euclidean results: {:?}", euclidean_results);

    let haversine_results =
        tree.neighbors_with_distance(query_point.0, query_point.1, Some(2), None, &haversine);
    println!("  Haversine results: {:?}", haversine_results);

    let spheroid_results =
        tree.neighbors_with_distance(query_point.0, query_point.1, Some(2), None, &spheroid);
    println!("  Spheroid results: {:?}", spheroid_results);

    // Test backward compatibility
    let original_results = tree.neighbors(query_point.0, query_point.1, Some(2), None);
    println!("  Original method results: {:?}", original_results);

    println!("  Note: For small distances, all metrics should give similar ordering.");
}
