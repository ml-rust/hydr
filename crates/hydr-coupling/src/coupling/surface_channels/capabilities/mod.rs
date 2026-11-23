//! Capability declaration of the 2D surface coupled to channel reaches and level pools through
//! bank weirs, structure links, reach-end ports, and pool ends.

mod unsupported;

use super::{SURFACE_CHANNEL_CONTRACT_VERSION, SURFACE_CHANNEL_SNAPSHOT_SCHEMA};
use hydr_core::capability::{
    ArithmeticRequirement, BoundaryKind, ComponentCapabilities, ComponentId, Endpoint, EndpointId,
    Limit, Precision, Regime, RegimeScope, RegimeSupport, RuntimeKind, RuntimeStatus,
    RuntimeSupport, SourceKind, Summary,
};

/// Path of the guide that defines links, ports, exchange, limits and scope.
const SURFACE_CHANNELS_DOC: &str = "docs/surface-channels.md";

/// Boundary kinds a structure link admits at a surface cell, a channel cell, or a pool.
///
/// Bank links join a surface cell and a channel cell through the weir kind.
const STRUCTURE: &[BoundaryKind] = &[
    BoundaryKind::SuppressedRectangularWeir,
    BoundaryKind::SubmergedOrifice,
    BoundaryKind::GateFixedOpening,
    BoundaryKind::GatePrescribedOpening,
    BoundaryKind::GateActuatedOpening,
    BoundaryKind::PumpRunning,
    BoundaryKind::PumpStoppedIsolated,
    BoundaryKind::PumpSampledControl,
    BoundaryKind::Culvert,
];

/// Boundary kinds a pool admits: every structure kind and external inflows.
const POOL: &[BoundaryKind] = &[
    BoundaryKind::SuppressedRectangularWeir,
    BoundaryKind::SubmergedOrifice,
    BoundaryKind::GateFixedOpening,
    BoundaryKind::GatePrescribedOpening,
    BoundaryKind::GateActuatedOpening,
    BoundaryKind::PumpRunning,
    BoundaryKind::PumpStoppedIsolated,
    BoundaryKind::PumpSampledControl,
    BoundaryKind::Culvert,
    BoundaryKind::PrescribedInflowRate,
    BoundaryKind::PrescribedInflowVolume,
];

/// Boundary kinds a reach end admits: a port or a pool end.
const REACH_END: &[BoundaryKind] = &[BoundaryKind::FluxPort, BoundaryKind::QuiescentReservoir];

/// Boundary kinds each surface side of a reach-end port admits.
const PORT: &[BoundaryKind] = &[BoundaryKind::FluxPort];

const fn regime(kind: BoundaryKind, regime: Regime) -> RegimeSupport {
    RegimeSupport {
        scope: RegimeScope::Boundary(kind),
        regime,
    }
}

const fn port_regime(regime: Regime) -> RegimeSupport {
    self::regime(BoundaryKind::FluxPort, regime)
}

const fn weir_regime(regime: Regime) -> RegimeSupport {
    self::regime(BoundaryKind::SuppressedRectangularWeir, regime)
}

const fn gate_regime(regime: Regime) -> RegimeSupport {
    self::regime(BoundaryKind::GateFixedOpening, regime)
}

static SURFACE_CHANNELS: ComponentCapabilities = ComponentCapabilities {
    component: ComponentId::SurfaceChannels,
    contract: Some(SURFACE_CHANNEL_CONTRACT_VERSION),
    snapshot_schema: Some(SURFACE_CHANNEL_SNAPSHOT_SCHEMA),
    precision: &[Precision::F64],
    arithmetic: &[
        ArithmeticRequirement::F64Ieee754Basic,
        ArithmeticRequirement::NearestEvenGradualUnderflow,
        ArithmeticRequirement::RoundToNearestSqrt,
        ArithmeticRequirement::FusedMulAdd,
        ArithmeticRequirement::CorrectlyRoundedElementary,
    ],
    runtimes: &[
        RuntimeSupport {
            runtime: RuntimeKind::Cpu,
            status: RuntimeStatus::Reference,
            evidence: "selected_cpu_client_executes_surface_channel_overflow_reversal_and_restart",
        },
        RuntimeSupport {
            runtime: RuntimeKind::Cuda,
            status: RuntimeStatus::Executed,
            evidence: "selected_cuda_client_executes_surface_channel_overflow_reversal_and_restart",
        },
        RuntimeSupport {
            runtime: RuntimeKind::Wgpu,
            status: RuntimeStatus::RejectedLacksF64,
            evidence: "selected_wgpu_rejects_surface_channel_f64_without_fallback",
        },
    ],
    endpoints: &[
        Endpoint {
            id: EndpointId::SurfaceCell,
            admitted: STRUCTURE,
        },
        Endpoint {
            id: EndpointId::Lateral,
            admitted: STRUCTURE,
        },
        Endpoint {
            id: EndpointId::PoolConnection,
            admitted: POOL,
        },
        Endpoint {
            id: EndpointId::ReachLeft,
            admitted: REACH_END,
        },
        Endpoint {
            id: EndpointId::ReachRight,
            admitted: REACH_END,
        },
        Endpoint {
            id: EndpointId::ExteriorFace,
            admitted: PORT,
        },
        Endpoint {
            id: EndpointId::Outlet,
            admitted: &[BoundaryKind::RunoffOutletNoFeedback],
        },
    ],
    regimes: &[
        weir_regime(Regime::FreeWeirFlow),
        weir_regime(Regime::SubmergedWeirFlow),
        weir_regime(Regime::BelowCrest),
        port_regime(Regime::WetSubcritical),
        port_regime(Regime::Supercritical),
        port_regime(Regime::Critical),
        port_regime(Regime::Dry),
        port_regime(Regime::WetDryFront),
        port_regime(Regime::ReversalInBranch),
        regime(BoundaryKind::SubmergedOrifice, Regime::SubmergedOrificeFlow),
        gate_regime(Regime::FreeGateAperture),
        gate_regime(Regime::MixedGateAperture),
        gate_regime(Regime::SubmergedGateAperture),
        gate_regime(Regime::GateClosed),
        regime(BoundaryKind::PumpRunning, Regime::PumpCurve),
        regime(BoundaryKind::PumpRunning, Regime::PumpShutoff),
        regime(BoundaryKind::PumpStoppedIsolated, Regime::PumpIsolated),
        regime(BoundaryKind::Culvert, Regime::CulvertInletControl),
        regime(BoundaryKind::Culvert, Regime::CulvertOutletControl),
        regime(BoundaryKind::QuiescentReservoir, Regime::WetSubcritical),
    ],
    sources: &[
        SourceKind::Rain,
        SourceKind::ImperviousLand,
        SourceKind::ModifiedGreenAmpt,
        SourceKind::Evaporation,
        SourceKind::Withdrawal,
        SourceKind::LateralInflow,
        SourceKind::LateralWithdrawal,
        SourceKind::LocalPoolRain,
        SourceKind::LocalPoolLoss,
    ],
    options: &[],
    nested: &[
        ComponentId::SurfaceSolver,
        ComponentId::Soil,
        ComponentId::ChannelReachRectangular,
        ComponentId::ChannelReachTriangular,
        ComponentId::ChannelReachSurveyed,
        ComponentId::LevelPool,
        ComponentId::OperatingControl,
        ComponentId::GateActuator,
        ComponentId::ControlledGate,
        ComponentId::ControlledPump,
        ComponentId::Culvert,
        ComponentId::Subcatchment,
    ],
    attachable_to: &[],
    unsupported: unsupported::UNSUPPORTED,
    limits: &[
        Limit {
            id: "momentum-closure",
            summary: "Surface water enters a reach at its post-transport tangent velocity; channel water leaves at its accepted-start velocity.",
            doc: SURFACE_CHANNELS_DOC,
        },
        Limit {
            id: "caller-link-evidence",
            summary: "The caller attests that linked cells share a bank and that no water is stored twice.",
            doc: SURFACE_CHANNELS_DOC,
        },
        Limit {
            id: "original-donor-water",
            summary: "Each donor gives only water it holds at the accepted start, under one shared factor.",
            doc: SURFACE_CHANNELS_DOC,
        },
        Limit {
            id: "fixed-point-convergence",
            summary: "Each update solves every link against a plan-area model of its end stages, so steep weirs near level heads keep their step. Allowances add the change one rounding of the heads causes. A residual forecast ends a stalled attempt early, and its measured contraction sets the shorter retry.",
            doc: SURFACE_CHANNELS_DOC,
        },
        Limit {
            id: "child-endpoint-policies",
            summary: "Surface exterior faces and reach ends outside ports take the nested component's policies.",
            doc: SURFACE_CHANNELS_DOC,
        },
        Limit {
            id: "port-tangential-closure",
            summary: "Water entering the surface through a port carries no tangential velocity; tangential momentum leaving the surface ends at the junction.",
            doc: SURFACE_CHANNELS_DOC,
        },
        Limit {
            id: "port-bed-step",
            summary: "A port bed step uses hydrostatic reconstruction at the higher bed; no energy guarantee holds across the step.",
            doc: SURFACE_CHANNELS_DOC,
        },
        Limit {
            id: "port-explicit-flux",
            summary: "Port flux comes from the accepted-start states and stays fixed for one attempt; depletion shortens the attempt instead of scaling the flux.",
            doc: SURFACE_CHANNELS_DOC,
        },
        Limit {
            id: "levelling-cap",
            summary: "Exchange stops at level heads: a levelling cap, not the weir law, ends each link's flow.",
            doc: SURFACE_CHANNELS_DOC,
        },
        Limit {
            id: "first-order-time",
            summary: "Time integration remains first order, without an energy guarantee for the complete calculation.",
            doc: SURFACE_CHANNELS_DOC,
        },
        Limit {
            id: "suction-cap",
            summary: "A draining pump stops where its donor reaches the suction bound within the interval; a cap, not a rejection, ends its flow.",
            doc: SURFACE_CHANNELS_DOC,
        },
        Limit {
            id: "culvert-invert-cap",
            summary: "A draining culvert donor stops just above its invert within the interval; a cap, not a rejection, ends its flow.",
            doc: SURFACE_CHANNELS_DOC,
        },
        Limit {
            id: "culvert-sensitivity-step",
            summary: "A culvert link's response step comes from its certified head sensitivity, without bisection over its law. A lane at the sensitivity cap takes the plain update and adds no resolution allowance.",
            doc: SURFACE_CHANNELS_DOC,
        },
        Limit {
            id: "structure-momentum-closure",
            summary: "Structure water enters either side with zero velocity; jets dissipate.",
            doc: SURFACE_CHANNELS_DOC,
        },
        Limit {
            id: "interval-operation",
            summary: "A gate takes the midpoint opening of its linear path over each interval; a pump holds its accepted operation.",
            doc: SURFACE_CHANNELS_DOC,
        },
        Limit {
            id: "event-aligned-intervals",
            summary: "Every interval ends at or before the next prescribed knot, actuator event, or control decision; no interval contains one.",
            doc: SURFACE_CHANNELS_DOC,
        },
        Limit {
            id: "pool-plan-area-bound",
            summary: "The cap and response bounds of a pool use the smallest segment plan area over the covered stage range.",
            doc: SURFACE_CHANNELS_DOC,
        },
        Limit {
            id: "pool-end-reservoir-head",
            summary: "A pool end takes the pool's temporal midpoint stage as a quiescent reservoir head; the reach's own transport sets the end volume.",
            doc: SURFACE_CHANNELS_DOC,
        },
        Limit {
            id: "blockage-as-scenario",
            summary: "A blockage is a fixed caller scenario with evidence; the element's condition holds the blockage and the scenario names it.",
            doc: SURFACE_CHANNELS_DOC,
        },
        Limit {
            id: "runoff-free-outlet-attested",
            summary: "The caller attests each runoff outlet's invert and that receiving water below it cannot change the export.",
            doc: SURFACE_CHANNELS_DOC,
        },
        Limit {
            id: "runoff-stage-sampled",
            summary: "The receiving stage is checked against the outlet invert at the accepted start and the candidate end only; a crossing inside the interval and back stays undetected.",
            doc: SURFACE_CHANNELS_DOC,
        },
        Limit {
            id: "runoff-recomputed-on-shortening",
            summary: "A shorter attempt trials the runoff child again from its accepted state to the new end; no volume is scaled.",
            doc: SURFACE_CHANNELS_DOC,
        },
        Limit {
            id: "runoff-zero-momentum",
            summary: "Runoff enters a reach cell or a surface cell with zero velocity; on a surface cell with link inflow, the link momentum spreads over the link and runoff water together.",
            doc: SURFACE_CHANNELS_DOC,
        },
        Limit {
            id: "runoff-after-transport",
            summary: "Runoff enters a surface cell after the surface transport of the interval; the interval's surface fluxes and step limits never see it.",
            doc: SURFACE_CHANNELS_DOC,
        },
        Limit {
            id: "runoff-span-rounding",
            summary: "An outlet spread over several reach or surface cells splits each lane volume by normalized weights; the parts sum to the lane within a rounding bound. One target keeps every bit.",
            doc: SURFACE_CHANNELS_DOC,
        },
        Limit {
            id: "pool-rain-atoms-unmeasured",
            summary: "The support plan checks pool rain atom ownership, never pool plan area against atom area; pool plan area varies with stage.",
            doc: SURFACE_CHANNELS_DOC,
        },
    ],
    summary: Summary {
        equations: "2D shallow-water surface, 1D channel reaches, and level pools exchanging water through suppressed rectangular bank weirs with Villemonte submergence, structure links (weirs, submerged orifices, hydrostatic gates, pumps, HDS-5 culverts) between surface cells, channel cells, and pools, pool ends at quiescent reservoir heads, water and normal momentum through reach-end ports under one hydrostatic HLL face flux, and nonlinear-reservoir subcatchment runoff into surface cells, reach cells, and pools through free outlets",
        method: "fixed-point iteration of end stages, signed link volumes, and pool-end volumes over one common interval, with a plan-area response step per update, midpoint-head law volumes, levelling, crest, suction, and culvert invert caps per link, one shared factor per donor over surface lanes, channel slots, and post-source pool water, an explicit HLL port flux from the accepted-start states per attempt, event-aligned intervals with per-interval operations from prescribed, actuated, and controlled-pump children and decision completion from the converged candidate, and nested child admission",
        assessment: "selected spill, reversal, donor-limit, retry, restart, pool, structure, control, and runoff cases; field accuracy remains unestablished",
    },
};

/// Returns the surface/channel coordinator's declared scope, independent of application
/// eligibility.
///
/// Execution success establishes no field accuracy.
pub fn surface_channel_capabilities() -> &'static ComponentCapabilities {
    &SURFACE_CHANNELS
}
