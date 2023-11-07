pub struct OwnedKdbush {
    pub(crate) buffer: Vec<u8>,
    pub(crate) node_size: usize,
    pub(crate) num_items: usize,
}

impl OwnedKdbush {
    pub fn into_inner(self) -> Vec<u8> {
        self.buffer
    }
}

impl AsRef<[u8]> for OwnedKdbush {
    fn as_ref(&self) -> &[u8] {
        &self.buffer
    }
}
