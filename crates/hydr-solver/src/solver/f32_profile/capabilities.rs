//! Capability declaration of the opt-in `F32` 2D shallow-water surface solver.

use super::{SURFACE_SOLVER_F32_CONTRACT_VERSION, SURFACE_SOLVER_F32_SNAPSHOT_SCHEMA};
use hydr_core::capability::{
    ArithmeticRequirement, BoundaryKind, ComponentCapabilities, ComponentId, Endpoint, EndpointId,
    Limit, Precision, Regime, RegimeScope, RegimeSupport, RuntimeKind, RuntimeStatus,
    RuntimeSupport, SourceKind, Summary, Unsupported, UnsupportedReason,
};

const F32_DOC: &str = "docs/surface-f32.md";

static WALL_ONLY: &[BoundaryKind] = &[BoundaryKind::Wall];

static SURFACE_SOLVER_F32: ComponentCapabilities = ComponentCapabilities {
    component: ComponentId::SurfaceSolverF32,
    contract: Some(SURFACE_SOLVER_F32_CONTRACT_VERSION),
    snapshot_schema: Some(SURFACE_SOLVER_F32_SNAPSHOT_SCHEMA),
    precision: &[Precision::F32],
    arithmetic: &[ArithmeticRequirement::F32Ieee754Basic],
    runtimes: &[
        RuntimeSupport {
            runtime: RuntimeKind::Cpu,
            status: RuntimeStatus::Reference,
            evidence: "surface_solver_f32_cpu_lake_at_rest_rain_and_dambreak",
        },
        RuntimeSupport {
            runtime: RuntimeKind::Cuda,
            status: RuntimeStatus::Executed,
            evidence: "selected_cuda_client_executes_surface_solver_f32",
        },
        RuntimeSupport {
            runtime: RuntimeKind::Wgpu,
            status: RuntimeStatus::Executed,
            evidence: "selected_wgpu_client_executes_surface_solver_f32",
        },
    ],
    endpoints: &[Endpoint {
        id: EndpointId::ExteriorFace,
        admitted: WALL_ONLY,
    }],
    regimes: &[
        RegimeSupport {
            scope: RegimeScope::Interior,
            regime: Regime::WetSubcritical,
        },
        RegimeSupport {
            scope: RegimeScope::Interior,
            regime: Regime::Supercritical,
        },
        RegimeSupport {
            scope: RegimeScope::Interior,
            regime: Regime::Dry,
        },
    ],
    sources: &[
        SourceKind::Rain,
        SourceKind::ImperviousLand,
        SourceKind::Evaporation,
        SourceKind::Withdrawal,
    ],
    options: &[],
    nested: &[],
    attachable_to: &[],
    unsupported: &[
        Unsupported {
            id: "certified-f64-contract",
            reason: UnsupportedReason::OutsideModel,
            summary: "Runs in binary32 (F32) for throughput and WebGPU support; use `Solver` (`F64`) for certified reference accuracy.",
            doc: F32_DOC,
        },
        Unsupported {
            id: "invalid-f32-config",
            reason: UnsupportedReason::RejectedByAdmission,
            summary: "Non-finite or out-of-range gravity, CFL, dry depth, or step limits reject at construction.",
            doc: F32_DOC,
        },
        Unsupported {
            id: "tampered-f32-snapshot",
            reason: UnsupportedReason::RejectedByAdmission,
            summary: "Snapshot restore rejects mismatched schema, contract, cell count, or non-finite/negative depths.",
            doc: F32_DOC,
        },
    ],
    limits: &[
        Limit {
            id: "f32-rounding-envelope",
            summary: "Single-precision rounding (~1e-7 relative) governs lake-at-rest and mass conservation; no TwoSum or correctly-rounded F64 contract applies.",
            doc: F32_DOC,
        },
        Limit {
            id: "closed-wall-boundaries",
            summary: "Current F32 profile admits closed impermeable wall boundaries and distributed rain/evaporation/withdrawal sources.",
            doc: F32_DOC,
        },
    ],
    summary: Summary {
        equations: "2D shallow-water finite-volume equations with hydrostatic bed reconstruction, HLL fluxes, and split Manning friction in binary32 (F32).",
        method: "Scatter-free 2D tensor sweeps running natively in DType::F32 on CpuRuntime, CudaRuntime, and WgpuRuntime.",
        assessment: "Opt-in fast/WebGPU profile verified against CPU F32 and certified F64 `Solver` on lake-at-rest, rain-on-basin, and 2D dam-break benchmarks.",
    },
};

/// Returns the capability declaration of the opt-in `F32` 2D shallow-water solver.
pub fn surface_solver_f32_capabilities() -> &'static ComponentCapabilities {
    &SURFACE_SOLVER_F32
}
