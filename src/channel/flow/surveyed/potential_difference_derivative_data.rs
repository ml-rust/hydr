pub(crate) struct DerivativePoint {
    pub area: [u64; 2],
    pub squared_width: [u64; 2],
    pub delta: [u64; 2],
    pub slope: [u64; 2],
    pub convex_margin: [u64; 2],
    pub reference: [u64; 2],
}
pub(crate) struct DerivativePanel {
    pub id: &'static str,
    pub area: [u64; 2],
    pub squared_width: [u64; 2],
    pub absolute_delta: [u64; 2],
    pub points: &'static [DerivativePoint],
}
