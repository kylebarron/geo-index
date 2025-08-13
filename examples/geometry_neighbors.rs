//! Example demonstrating geometry-based neighbor search.
//!
//! This example shows how to use the neighbors_geometry method to find
//! nearest neighbors based on actual geometry distances rather than just
//! bounding box distances.

use geo::algorithm::BoundingRect;
use geo::{Geometry, LineString, Point, Polygon};
use geo_index::rtree::distance::EuclideanDistance;
use geo_index::rtree::sort::HilbertSort;
use geo_index::rtree::{RTreeBuilder, RTreeIndex};
use geo_types::coord;

fn main() {
    println!("=== Geometry-based Neighbor Search Example ===\n");

    // Example 1: Point geometries
    println!("1. Finding nearest point geometries:");
    point_geometries_example();

    // Example 2: Mixed geometry types
    println!("\n2. Finding nearest geometries with mixed types:");
    mixed_geometries_example();

    // Example 3: LineString geometries
    println!("\n3. Finding nearest line geometries:");
    linestring_geometries_example();
}

fn point_geometries_example() {
    let mut builder = RTreeBuilder::<f64>::new(5);

    // Create bounding boxes for the points
    let points = vec![
        Point::new(1.0, 1.0),
        Point::new(4.0, 2.0),
        Point::new(2.0, 5.0),
        Point::new(7.0, 3.0),
        Point::new(5.0, 6.0),
    ];

    // Add bounding boxes to the spatial index
    for point in points.iter() {
        builder.add(point.x(), point.y(), point.x(), point.y());
    }
    let tree = builder.finish::<HilbertSort>();

    // Create geometries array
    let geometries: Vec<Geometry<f64>> = points.into_iter().map(|p| Geometry::Point(p)).collect();

    // Query with a point geometry
    let query_geom = Geometry::Point(Point::new(3.0, 3.0));
    let euclidean = EuclideanDistance;
    let results = tree.neighbors_geometry(&query_geom, Some(3), None, &euclidean, &geometries);

    println!("  Query point: (3.0, 3.0)");
    println!("  Nearest 3 points (indices): {:?}", results);

    for (rank, &idx) in results.iter().enumerate() {
        if let Geometry::Point(p) = &geometries[idx as usize] {
            println!("    {}. Point at ({}, {})", rank + 1, p.x(), p.y());
        }
    }
}

fn mixed_geometries_example() {
    let mut builder = RTreeBuilder::<f64>::new(4);

    // Create mixed geometries and their bounding boxes
    let geometries = vec![
        // Point at origin
        Geometry::Point(Point::new(0.0, 0.0)),
        // Horizontal line
        Geometry::LineString(LineString::new(vec![
            coord! { x: 2.0, y: 2.0 },
            coord! { x: 6.0, y: 2.0 },
        ])),
        // Square polygon
        Geometry::Polygon(Polygon::new(
            LineString::new(vec![
                coord! { x: 5.0, y: 5.0 },
                coord! { x: 8.0, y: 5.0 },
                coord! { x: 8.0, y: 8.0 },
                coord! { x: 5.0, y: 8.0 },
                coord! { x: 5.0, y: 5.0 },
            ]),
            vec![],
        )),
        // Another point
        Geometry::Point(Point::new(10.0, 1.0)),
    ];

    // Add bounding boxes to the spatial index
    for geom in &geometries {
        if let Some(rect) = geom.bounding_rect() {
            let min = rect.min();
            let max = rect.max();
            builder.add(min.x, min.y, max.x, max.y);
        }
    }
    let tree = builder.finish::<HilbertSort>();

    // Query with a point near the line
    let query_geom = Geometry::Point(Point::new(4.0, 3.0));
    let euclidean = EuclideanDistance;
    let results = tree.neighbors_geometry(&query_geom, None, None, &euclidean, &geometries);

    println!("  Query point: (4.0, 3.0)");
    println!("  Nearest geometries (indices): {:?}", results);

    for (rank, &idx) in results.iter().enumerate() {
        match &geometries[idx as usize] {
            Geometry::Point(p) => println!("    {}. Point at ({}, {})", rank + 1, p.x(), p.y()),
            Geometry::LineString(_) => println!("    {}. LineString", rank + 1),
            Geometry::Polygon(_) => println!("    {}. Polygon", rank + 1),
            _ => println!("    {}. Other geometry", rank + 1),
        }
    }
}

fn linestring_geometries_example() {
    let mut builder = RTreeBuilder::<f64>::new(3);

    // Create line geometries
    let lines = vec![
        // Horizontal line at y=0
        LineString::new(vec![coord! { x: 0.0, y: 0.0 }, coord! { x: 10.0, y: 0.0 }]),
        // Vertical line at x=5
        LineString::new(vec![coord! { x: 5.0, y: 0.0 }, coord! { x: 5.0, y: 10.0 }]),
        // Diagonal line
        LineString::new(vec![coord! { x: 0.0, y: 0.0 }, coord! { x: 10.0, y: 10.0 }]),
    ];

    // Add bounding boxes to the spatial index
    for line in &lines {
        if let Some(rect) = line.bounding_rect() {
            let min = rect.min();
            let max = rect.max();
            builder.add(min.x, min.y, max.x, max.y);
        }
    }
    let tree = builder.finish::<HilbertSort>();

    // Create geometries array
    let geometries: Vec<Geometry<f64>> =
        lines.into_iter().map(|l| Geometry::LineString(l)).collect();

    // Query with a point
    let query_geom = Geometry::Point(Point::new(3.0, 1.0));
    let euclidean = EuclideanDistance;
    let results = tree.neighbors_geometry(&query_geom, None, None, &euclidean, &geometries);

    println!("  Query point: (3.0, 1.0)");
    println!("  Nearest lines (indices): {:?}", results);
    println!("  Line 0: Horizontal line at y=0 (closest to query point)");
    println!("  Line 1: Vertical line at x=5");
    println!("  Line 2: Diagonal line");
}
