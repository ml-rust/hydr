#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ProfileRef {
    Query(usize),
    OriginalRectangle,
    LargeRectangle,
}
#[derive(Clone, Copy, Debug)]
pub(crate) enum Input {
    Baseline {
        binding_index: usize,
        original_parameter_endpoint: usize,
    },
    SourceRelative {
        profile: ProfileRef,
        gravity_bits: u64,
        source: [u64; 2],
        delta: [u64; 2],
    },
}
#[derive(Clone, Copy, Debug)]
pub(crate) enum Reference {
    ExactZero,
    Local { bounds: [u64; 2], allowance: u64 },
}
#[derive(Clone, Copy, Debug)]
pub(crate) enum Expected {
    Dry,
    Equality { limit: Reference },
    Rare { curve: Reference },
    Shock { curve: Reference, offset: Reference },
    NumericalRange,
}
#[derive(Clone, Copy, Debug)]
pub(crate) struct Case {
    pub id: &'static str,
    pub input: Input,
    pub expected: Expected,
}
#[derive(Clone, Copy, Debug)]
pub(crate) struct Profile {
    pub coordinates: &'static [[u64; 2]],
    pub cap_stage: u64,
}
#[derive(Clone, Copy, Debug)]
pub(crate) struct UncertainSign {
    pub case_index: usize,
    pub delta_center: [u64; 2],
    pub delta_radius: [u64; 2],
    pub actual_shock_offset_unavailable: bool,
    pub overlap_proves_neither_equality_nor_shock: bool,
}
#[derive(Clone, Copy, Debug)]
pub(crate) struct ContractionUnderflow {
    pub case_index: usize,
    pub nonzero_physical_truth: bool,
    pub zero_center_requires_nonzero_radius: bool,
}

pub(super) const fn local(lower: u64, upper: u64, allowance: u64) -> Reference {
    Reference::Local {
        bounds: [lower, upper],
        allowance,
    }
}
pub(super) const fn baseline(
    id: &'static str,
    binding_index: usize,
    original_parameter_endpoint: usize,
    expected: Expected,
) -> Case {
    Case {
        id,
        input: Input::Baseline {
            binding_index,
            original_parameter_endpoint,
        },
        expected,
    }
}
pub(super) const fn relative(
    id: &'static str,
    profile: ProfileRef,
    gravity_bits: u64,
    source: [u64; 2],
    delta: [u64; 2],
    expected: Expected,
) -> Case {
    Case {
        id,
        input: Input::SourceRelative {
            profile,
            gravity_bits,
            source,
            delta,
        },
        expected,
    }
}
