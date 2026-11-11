#[derive(Clone, Copy)]
pub(crate) struct Profile {
    pub(crate) id: &'static str,
    pub(crate) points: &'static [[u64; 2]],
    pub(crate) cap: u64,
}

#[derive(Clone, Copy)]
pub(crate) struct Case {
    pub(crate) owner: usize,
    pub(crate) operand: u64,
    pub(crate) perimeter: [u64; 2],
    pub(crate) radius: Option<[u64; 2]>,
    pub(crate) depth: Option<[u64; 2]>,
}
