# Changelog

**This is the changelog for the core Rust library**. There's a [separate changelog](./python/CHANGELOG.md) for the Python bindings.

## [0.3.1] - 2025-06-20

## Bug fixes

- Fix hanging when building RTree with 0 items by @kontinuation in https://github.com/kylebarron/geo-index/pull/129
- Fix panic when building STR tree with certain number of items by @kontinuation in https://github.com/kylebarron/geo-index/pull/129

## [0.3.0] - 2025-06-13

### Breaking

- Upgrade geo-traits to 0.3 by @kontinuation in https://github.com/kylebarron/geo-index/pull/127
- Update geo to 0.30.0 by @kontinuation in https://github.com/kylebarron/geo-index/pull/127

## [0.2.0] - 2025-01-06

### Breaking

- Use u32 and u16 in public API for num_items and node_size by @kylebarron in https://github.com/kylebarron/geo-index/pull/69
- Rename `OwnedRTree` to `RTree` and `OwnedKDTree` to `KDTree` by @kylebarron in https://github.com/kylebarron/geo-index/pull/81

### Bug fixes

- Fix `intersection_candidates_with_other_tree` by @kylebarron in https://github.com/kylebarron/geo-index/pull/51
- Improve precision in f64 to f32 box cast by @kylebarron in https://github.com/kylebarron/geo-index/pull/76
- Avoid panic for rtree with one item by @kylebarron in https://github.com/kylebarron/geo-index/pull/91

### New Features

- Implement nearest neighbor searches on RTree by @kylebarron in https://github.com/kylebarron/geo-index/pull/79
- Add geo-traits integration by @kylebarron in https://github.com/kylebarron/geo-index/pull/71
  - Implement RectTrait for Node by @kylebarron in https://github.com/kylebarron/geo-index/pull/75
- KDTree traversal by @kylebarron in https://github.com/kylebarron/geo-index/pull/96
- Expose RTreeMetadata & KDTreeMetadata (allowing you to infer the memory usage a tree would incur) by @kylebarron in https://github.com/kylebarron/geo-index/pull/77

### Performance

- Remove unnecessary `Cow` in kdtree trait by @kylebarron in https://github.com/kylebarron/geo-index/pull/72

### Documentation

- Use "immutable" over "static" wording in docs by @kylebarron in https://github.com/kylebarron/geo-index/pull/70
- improved rtree & kdtree docs by @kylebarron in https://github.com/kylebarron/geo-index/pull/93

### What's Changed

- Don't panic for accessing level out of bounds by @kylebarron in https://github.com/kylebarron/geo-index/pull/49

### New Contributors

- @H-Plus-Time made their first contribution in https://github.com/kylebarron/geo-index/pull/55

**Full Changelog**: https://github.com/kylebarron/geo-index/compare/v0.1.1...v0.2.0

## [0.1.1] - 2024-01-14

- Updated benchmarks in documentation by @kylebarron in #27

## [0.1.0] - 2024-01-14

- Initial public release.
