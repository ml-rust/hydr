//! Precision, arithmetic requirements and runtime support.

/// Floating-point format of a component's state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Precision {
    /// IEEE 754 binary64.
    F64,
}

impl Precision {
    /// Returns the stable kebab-case id. Persisted decisions store it, so it never changes.
    pub fn id(&self) -> &'static str {
        match self {
            Self::F64 => "f64",
        }
    }
}

/// Arithmetic property that a runtime must provide for a component.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ArithmeticRequirement {
    /// IEEE 754 binary64 add, subtract, multiply and divide.
    F64Ieee754Basic,
    /// Round-to-nearest-even with gradual underflow to subnormals.
    NearestEvenGradualUnderflow,
    /// Square root rounded to nearest.
    RoundToNearestSqrt,
    /// Fused multiply-add with one rounding.
    FusedMulAdd,
    /// F64 `exp`, `log`, `log1p`, `pow` and `acos` through numr `CorrectlyRoundedOps`.
    /// `exp`, `log` and `log1p` are correctly rounded to nearest.
    /// `pow` is correctly rounded modulo the unknown hardest cases.
    /// `acos` is correctly rounded on `|x| >= 2^-4`, proven by an exhaustive upstream search.
    /// Below `2^-4` no proof covers it, and numr's MPFR checks find no misrounded input.
    /// Interval `acos` pads one ulp outward, which stays valid for any faithful result.
    /// All five are deterministic on every runtime that meets the contracts.
    /// Needs IEEE basic arithmetic and fused multiply-add.
    /// Evidence: numr `tests/correctly_rounded.rs` checks every function against MPFR, and
    /// `tests/backend_parity/correctly_rounded.rs` checks CUDA against CPU bit for bit.
    CorrectlyRoundedElementary,
}

impl ArithmeticRequirement {
    /// Returns the stable kebab-case id. Persisted decisions store it, so it never changes.
    pub fn id(&self) -> &'static str {
        match self {
            Self::F64Ieee754Basic => "f64-ieee754-basic",
            Self::NearestEvenGradualUnderflow => "nearest-even-gradual-underflow",
            Self::RoundToNearestSqrt => "round-to-nearest-sqrt",
            Self::FusedMulAdd => "fused-mul-add",
            Self::CorrectlyRoundedElementary => "correctly-rounded-elementary",
        }
    }
}

/// Numr runtime family.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum RuntimeKind {
    /// CPU runtime.
    Cpu,
    /// CUDA runtime.
    Cuda,
    /// WebGPU runtime.
    Wgpu,
}

impl RuntimeKind {
    /// Returns the stable kebab-case id. Persisted decisions store it, so it never changes.
    pub fn id(&self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Cuda => "cuda",
            Self::Wgpu => "wgpu",
        }
    }
}

/// Standing of a component on one runtime.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum RuntimeStatus {
    /// Reference profile. Other runtimes are checked against it.
    Reference,
    /// A test runs the component on this runtime.
    Executed,
    /// Admission accepts the runtime, but no test runs the component on it.
    AdmittedUnexercised,
    /// Admission refuses the runtime because it lacks F64.
    RejectedLacksF64,
}

impl RuntimeStatus {
    /// Returns the stable kebab-case id. Persisted decisions store it, so it never changes.
    pub fn id(&self) -> &'static str {
        match self {
            Self::Reference => "reference",
            Self::Executed => "executed",
            Self::AdmittedUnexercised => "admitted-unexercised",
            Self::RejectedLacksF64 => "rejected-lacks-f64",
        }
    }
}

/// Standing of a component on one runtime, with its evidence.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RuntimeSupport {
    /// Runtime family.
    pub runtime: RuntimeKind,
    /// Standing on that runtime.
    pub status: RuntimeStatus,
    /// Name of the test function that shows the status.
    pub evidence: &'static str,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capability::id_check::assert_ids;

    #[test]
    fn precision_ids_are_unique_kebab_case() {
        assert_ids(&[Precision::F64.id()]);
    }

    #[test]
    fn arithmetic_ids_are_unique_kebab_case() {
        let all = [
            ArithmeticRequirement::F64Ieee754Basic,
            ArithmeticRequirement::NearestEvenGradualUnderflow,
            ArithmeticRequirement::RoundToNearestSqrt,
            ArithmeticRequirement::FusedMulAdd,
            ArithmeticRequirement::CorrectlyRoundedElementary,
        ];
        assert_ids(&all.map(|a| a.id()));
    }

    #[test]
    fn runtime_ids_are_unique_kebab_case() {
        let all = [RuntimeKind::Cpu, RuntimeKind::Cuda, RuntimeKind::Wgpu];
        assert_ids(&all.map(|r| r.id()));
    }

    #[test]
    fn runtime_status_ids_are_unique_kebab_case() {
        let all = [
            RuntimeStatus::Reference,
            RuntimeStatus::Executed,
            RuntimeStatus::AdmittedUnexercised,
            RuntimeStatus::RejectedLacksF64,
        ];
        assert_ids(&all.map(|s| s.id()));
    }
}
