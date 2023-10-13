pub trait IndexableNum {
    /// The type index to match the array order of `ARRAY_TYPES` in flatbush JS
    const TYPE_INDEX: u8;
}

impl IndexableNum for i8 {
    const TYPE_INDEX: u8 = 0;
}

impl IndexableNum for u8 {
    const TYPE_INDEX: u8 = 1;
}

impl IndexableNum for i16 {
    const TYPE_INDEX: u8 = 3;
}

impl IndexableNum for u16 {
    const TYPE_INDEX: u8 = 4;
}

impl IndexableNum for i32 {
    const TYPE_INDEX: u8 = 5;
}

impl IndexableNum for u32 {
    const TYPE_INDEX: u8 = 6;
}

impl IndexableNum for f32 {
    const TYPE_INDEX: u8 = 7;
}

impl IndexableNum for f64 {
    const TYPE_INDEX: u8 = 8;
}
