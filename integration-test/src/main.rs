use geo::Polygon;
use geo::{BoundingRect, Geometry};
use geo_index::rtree::sort::HilbertSort;
use geo_index::rtree::{RTreeBuilder, RTreeIndex};
use geozero::geo_types::GeoWriter;
use geozero::geojson::read_geojson_fc;
use rstar::primitives::GeomWithData;
use rstar::{primitives::Rectangle, AABB};

// Find tree self-intersection canddiates using rstar
fn geo_contiguity(geom: &[Polygon]) {
    let to_insert = geom
        .iter()
        .enumerate()
        .map(|(i, gi)| {
            let rect = gi.bounding_rect().unwrap();
            let aabb =
                AABB::from_corners([rect.min().x, rect.min().y], [rect.max().x, rect.max().y]);

            GeomWithData::new(Rectangle::from_aabb(aabb), i)
        })
        .collect::<Vec<_>>();

    let tree = rstar::RTree::bulk_load(to_insert);
    let candidates = tree.intersection_candidates_with_other_tree(&tree);

    println!("rstar candidates:");
    for (left_idx, right_idx) in candidates {
        println!("{:?} {:?}", left_idx.data, right_idx.data);
    }
}

// Find tree self-intersection canddiates using geo-index
fn geo_index_contiguity(geoms: &Vec<Polygon>, node_size: usize) {
    let mut tree_builder = RTreeBuilder::new_with_node_size(geoms.len(), node_size);
    for geom in geoms {
        let rect = geom.bounding_rect().unwrap();
        tree_builder.add(rect.min().x, rect.min().y, rect.max().x, rect.max().y);
    }
    let tree = tree_builder.finish::<HilbertSort>();

    let candidates = tree.intersection_candidates_with_other_tree(&tree);

    println!("geo-index candidates:");
    for (left_idx, right_idx) in candidates {
        println!("{:?} {:?}", left_idx, right_idx);
    }
}

fn main() {
    let file = std::fs::File::open("src/guerry.geojson").unwrap();
    let reader = std::io::BufReader::new(file);

    let mut geo_writer = GeoWriter::new();
    read_geojson_fc(reader, &mut geo_writer).unwrap();

    let geoms = match geo_writer.take_geometry().unwrap() {
        Geometry::GeometryCollection(gc) => gc.0,
        _ => panic!(),
    };

    let mut polys = vec![];
    for geom in geoms {
        let poly = match geom {
            Geometry::Polygon(poly) => poly,
            _ => panic!(),
        };
        polys.push(poly);
    }

    println!("There are {} polygons", polys.len());
    geo_index_contiguity(&polys, 10);

    geo_contiguity(&polys);
}
