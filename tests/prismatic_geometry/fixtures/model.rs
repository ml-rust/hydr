#[derive(Clone, Copy, Debug)]
pub(crate) struct Truth {
    pub lower_bits: u64,
    pub upper_bits: u64,
    pub maximum_width_bits: u64,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct ProfileCase {
    pub id: &'static str,
    pub points: &'static [[u64; 2]],
    pub maximum_stage_bits: u64,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct ForwardCase {
    pub profile: usize,
    pub depth_bits: u64,
    pub area: Truth,
    pub top_width: Option<Truth>,
    pub integral: Truth,
    pub hydraulic_depth: Option<Truth>,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct InverseCase {
    pub profile: usize,
    pub area_bits: u64,
    pub original_depth_bits: Option<u64>,
    pub depth: Truth,
    pub forward_area: Truth,
    pub top_width: Option<Truth>,
    pub integral: Truth,
    pub hydraulic_depth: Option<Truth>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Quantity {
    Depth,
    Area,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct QueryRejection {
    pub profile: usize,
    pub quantity: Quantity,
    pub bits: u64,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct GeometryRejection {
    pub profile: ProfileCase,
    pub reason: &'static str,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct UnresolvedInverse {
    pub profile: ProfileCase,
    pub area_bits: u64,
}

pub(crate) const fn truth(lower_bits: u64, upper_bits: u64, maximum_width_bits: u64) -> Truth {
    Truth {
        lower_bits,
        upper_bits,
        maximum_width_bits,
    }
}

pub(crate) const fn forward(
    profile: usize,
    depth_bits: u64,
    area: Truth,
    top_width: Option<Truth>,
    integral: Truth,
    hydraulic_depth: Option<Truth>,
) -> ForwardCase {
    ForwardCase {
        profile,
        depth_bits,
        area,
        top_width,
        integral,
        hydraulic_depth,
    }
}

pub(crate) const fn inverse(
    profile: usize,
    operand: (u64, Option<u64>),
    depth: Truth,
    forward_area: Truth,
    top_width: Option<Truth>,
    integral: Truth,
    hydraulic_depth: Option<Truth>,
) -> InverseCase {
    InverseCase {
        profile,
        area_bits: operand.0,
        original_depth_bits: operand.1,
        depth,
        forward_area,
        top_width,
        integral,
        hydraulic_depth,
    }
}
