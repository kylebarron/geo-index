# Changelog

## [0.2.0] - 2025-01-06

### New Features

- Support for nearest neighbor searching on RTrees with [`neighbors`](https://kylebarron.dev/geo-index/v0.2.0/api/rtree/#geoindex_rs.rtree.neighbors).
- Join two RTrees together with [`tree_join`](https://kylebarron.dev/geo-index/v0.2.0/api/rtree/#geoindex_rs.rtree.tree_join), finding their overlapping elements. This is the first part of a spatial join: to find which elements from two different data sources potentially intersect.
- Extract partitioning structure from the underlying RTree with [`partitions`](https://kylebarron.dev/geo-index/v0.2.0/api/rtree/#geoindex_rs.rtree.partitions) and see the partition geometries with [`partition_boxes`](https://kylebarron.dev/geo-index/v0.2.0/api/rtree/#geoindex_rs.rtree.partition_boxes).
- Expose [`RTreeMetadata`](https://kylebarron.dev/geo-index/v0.2.0/api/rtree/#geoindex_rs.rtree.RTreeMetadata) and [`KDTreeMetadata`](https://kylebarron.dev/geo-index/v0.2.0/api/kdtree/#geoindex_rs.kdtree.KDTreeMetadata). These allow you to infer the memory usage a tree would incur.
- Access the internal boxes within the RTree for inspecting the tree internals with `boxes_at_level`.
- Implement the buffer protocol on `RTree` and `KDTree`. This means you can copy the underlying buffer to Python with `bytes(tree)`.

### Breaking

- **Move RTree and KDTree query functions to standalone global functions**. This
  makes it easier to persist index buffers and reuse them later, because the
  query functions work on any object supporting the buffer protocol.
- **Create "builder" classes**: `RTreeBuilder` and `KDTreeBuilder`. Having these as separate classes allows for iteratively adding the coordinates for an RTree or KDTree. This is useful when the source geometries are larger than fits in memory.

### Documentation

- New documentation website for Python bindings.

## [0.1.0] - 2024-03-26

- Initial public release.
