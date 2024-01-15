# geo-index

A Rust crate for packed, static, ABI-stable spatial indexes.

## Features

- **An R-tree and k-d tree written in safe rust.**
- **Fast.** Because of optimizations available by using static indexes, tends to be faster than dynamic implementations like [`rstar`](https://github.com/georust/rstar).
- **Memory-efficient.** The index is fully _packed_, meaning that all nodes are at full capacity (except for the last node at each tree level). This means the RTree uses less memory. And because the index is backed by a single buffer, it exhibits excellent memory locality. For any number of input geometries, the peak memory use both to build the index and to store the index can be pre-computed.
- **Multiple R-tree sorting methods:** hilbert or sort-tile-recursive (STR), and extensible to other spatial sorting algorithms in the future, like overlap-minimizing top-down (OMT).
- **ABI-stable:** the index is contained in a single `Vec<u8>`, compatible with the [`flatbush`](https://github.com/mourner/flatbush) and [`kdbush`](https://github.com/mourner/kdbush) JavaScript libraries. Being ABI-stable means that the spatial index can be shared zero-copy between Rust and another program like Python.
- **Generic over a set of coordinate types:** `i8`, `u8`, `i16`, `u16`, `i32`, `u32`, `f32`, `f64`.
- **Efficient bulk loading.** As a static index, _only_ bulk loading is supported.
- Optional `rayon` feature for parallelizing part of the sort in the sort-tile-recursive (`STRSort`) method.

## Drawbacks

- Trees are _static_. After creating the index, items can no longer be added or removed.
- Only two-dimensional data is supported. Can still be used with higher-dimensional input if you're ok with only indexing two of the dimensions.
- Only the set of coordinate types that exist in JavaScript are allowed, to maintain FFI compatibility with the reference JavaScript implementations. This does not and probably will not support other types like `u64`.
- Queries return positional indexes into the input set, so you must manage your own collections.

## Alternatives

- [`rstar`](https://github.com/georust/rstar): a dynamic RTree implementation.
- [`kdtree`](https://github.com/mrhooray/kdtree-rs): a dynamic KDTree implementation.
- [`kdbush`](https://github.com/pka/rust-kdbush): a port of `kdbush` but does not strive for FFI-compatibility.
- [`static_aabb2d_index`](https://github.com/jbuckmccready/static_aabb2d_index): a port of `flatbush` but does not strive for FFI-compatibility.
- [`static-bushes`](https://github.com/apendleton/static-bushes): a port of `flatbush` and `kdbush` but does not strive for FFI-compatibility.

## Applications

[`geoarrow-rs`](https://github.com/geoarrow/geoarrow-rs) uses this library to speed up boolean operations and spatial joins. In the future, there may be more interesting FFI-focused use cases in conjunction with Python and JavaScript.

## Future work

- geographic queries

## Inspiration



## Benchmarks

```ignore
cargo bench --bench rtree --features rayon
```

```ignore
construction (flatbush) time:   [77.642 ms 77.880 ms 78.153 ms]
construction (flatbush f64 to f32, including casting)
                        time:   [86.559 ms 87.194 ms 88.119 ms]
construction (flatbush f32)
                        time:   [79.957 ms 80.450 ms 81.125 ms]
construction (rstar bulk)
                        time:   [154.73 ms 155.12 ms 155.57 ms]

search() results in 34384 items
search() on f32 results in 34391 items

flatbush buffer size: 41533064 bytes
flatbush f32 buffer size: 23073928 bytes

search (flatbush)       time:   [98.864 µs 98.967 µs 99.084 µs]
search (flatbush f32)   time:   [104.81 µs 105.86 µs 107.02 µs]
search (rstar)          time:   [149.09 µs 149.37 µs 149.64 µs]
```
