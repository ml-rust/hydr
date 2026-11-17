//! Component capability declaration and its scope records.

use super::{
    ArithmeticRequirement, BoundaryKind, ComponentId, Endpoint, EndpointId, Precision,
    RegimeSupport, RuntimeKind, RuntimeSupport, SourceKind,
};

/// Named option with its accepted values and default.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OptionSet {
    /// Stable kebab-case option id.
    pub id: &'static str,
    /// Accepted values.
    pub values: &'static [&'static str],
    /// Value used when the caller selects none, or `None` when the caller must choose.
    /// A `Some` value is one of `values`.
    pub default: Option<&'static str>,
}

/// Reason a component does not support a capability.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum UnsupportedReason {
    /// Admission refuses the input with an error.
    RejectedByAdmission,
    /// The component has no attachment point for it.
    NotAttached,
    /// The physics lies outside the component's model.
    OutsideModel,
    /// The code runs it, but no test qualifies the result.
    Unqualified,
}

impl UnsupportedReason {
    /// Returns the stable kebab-case id. Persisted decisions store it, so it never changes.
    pub fn id(&self) -> &'static str {
        match self {
            Self::RejectedByAdmission => "rejected-by-admission",
            Self::NotAttached => "not-attached",
            Self::OutsideModel => "outside-model",
            Self::Unqualified => "unqualified",
        }
    }
}

/// Capability that a component does not support.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Unsupported {
    /// Stable kebab-case id.
    pub id: &'static str,
    /// Why the component does not support it.
    pub reason: UnsupportedReason,
    /// One-line description.
    pub summary: &'static str,
    /// Repository-relative path of the doc that explains it.
    pub doc: &'static str,
}

/// Boundary of a supported capability's valid range.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Limit {
    /// Stable kebab-case id.
    pub id: &'static str,
    /// One-line description.
    pub summary: &'static str,
    /// Repository-relative path of the doc that explains it.
    pub doc: &'static str,
}

/// Plain-text description of a component's model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Summary {
    /// Governing equations.
    pub equations: &'static str,
    /// Numerical method.
    pub method: &'static str,
    /// Scope of the evidence behind the declaration.
    pub assessment: &'static str,
}

/// Everything one component admits, refuses and limits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ComponentCapabilities {
    /// Declaring component.
    pub component: ComponentId,
    /// Public numerical contract version, when the component has one.
    pub contract: Option<&'static str>,
    /// Snapshot schema identity, when the component writes snapshots.
    pub snapshot_schema: Option<&'static str>,
    /// Supported state precisions.
    pub precision: &'static [Precision],
    /// Arithmetic properties the runtime must provide.
    pub arithmetic: &'static [ArithmeticRequirement],
    /// Standing on each runtime.
    pub runtimes: &'static [RuntimeSupport],
    /// Endpoints and their admitted boundary kinds.
    pub endpoints: &'static [Endpoint],
    /// Supported regimes per scope.
    pub regimes: &'static [RegimeSupport],
    /// Admitted local sources and sinks.
    pub sources: &'static [SourceKind],
    /// Options and their accepted values.
    pub options: &'static [OptionSet],
    /// Components that this component contains.
    pub nested: &'static [ComponentId],
    /// Components that this component attaches to.
    pub attachable_to: &'static [ComponentId],
    /// Capabilities the component does not support.
    pub unsupported: &'static [Unsupported],
    /// Limits of the supported range.
    pub limits: &'static [Limit],
    /// Model description.
    pub summary: Summary,
}

impl ComponentCapabilities {
    /// Returns the declared endpoint with identity `id`.
    pub fn endpoint(&self, id: EndpointId) -> Option<&'static Endpoint> {
        self.endpoints.iter().find(|e| e.id == id)
    }

    /// Returns `true` when endpoint `id` is declared and admits `kind`.
    pub fn admits(&self, id: EndpointId, kind: BoundaryKind) -> bool {
        self.endpoint(id).is_some_and(|e| e.admits(kind))
    }

    /// Returns the declared standing on `runtime`.
    pub fn runtime(&self, runtime: RuntimeKind) -> Option<&'static RuntimeSupport> {
        self.runtimes.iter().find(|r| r.runtime == runtime)
    }

    /// Returns the option with id `id`.
    pub fn option(&self, id: &str) -> Option<&'static OptionSet> {
        self.options.iter().find(|o| o.id == id)
    }

    /// Returns the unsupported entry with id `id`.
    pub fn unsupported_by_id(&self, id: &str) -> Option<&'static Unsupported> {
        self.unsupported.iter().find(|u| u.id == id)
    }

    /// Returns the limit with id `id`.
    pub fn limit_by_id(&self, id: &str) -> Option<&'static Limit> {
        self.limits.iter().find(|l| l.id == id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capability::RuntimeStatus;
    use crate::capability::id_check::assert_ids;

    const SAMPLE: ComponentCapabilities = ComponentCapabilities {
        component: ComponentId::LevelPool,
        contract: None,
        snapshot_schema: None,
        precision: &[Precision::F64],
        arithmetic: &[ArithmeticRequirement::F64Ieee754Basic],
        runtimes: &[RuntimeSupport {
            runtime: RuntimeKind::Cpu,
            status: RuntimeStatus::Reference,
            evidence: "cpu_reference",
        }],
        endpoints: &[Endpoint {
            id: EndpointId::Outlet,
            admitted: &[BoundaryKind::SuppressedRectangularWeir],
        }],
        regimes: &[],
        sources: &[SourceKind::LocalPoolRain],
        options: &[OptionSet {
            id: "step",
            values: &["fixed", "adaptive"],
            default: Some("fixed"),
        }],
        nested: &[],
        attachable_to: &[ComponentId::SurfaceSolver],
        unsupported: &[Unsupported {
            id: "seepage",
            reason: UnsupportedReason::OutsideModel,
            summary: "No seepage.",
            doc: "The pool holds no seepage term.",
        }],
        limits: &[Limit {
            id: "max-stage",
            summary: "Stage stays within the stage-volume table.",
            doc: "The stage-volume table ends at its top row.",
        }],
        summary: Summary {
            equations: "storage balance",
            method: "explicit",
            assessment: "sample",
        },
    };

    #[test]
    fn reason_ids_are_unique_kebab_case() {
        let all = [
            UnsupportedReason::RejectedByAdmission,
            UnsupportedReason::NotAttached,
            UnsupportedReason::OutsideModel,
            UnsupportedReason::Unqualified,
        ];
        assert_ids(&all.map(|r| r.id()));
    }

    #[test]
    fn admits_needs_declared_endpoint_and_kind() {
        assert!(SAMPLE.admits(EndpointId::Outlet, BoundaryKind::SuppressedRectangularWeir));
        assert!(!SAMPLE.admits(EndpointId::Outlet, BoundaryKind::Wall));
        assert!(!SAMPLE.admits(EndpointId::Lateral, BoundaryKind::SuppressedRectangularWeir));
    }

    #[test]
    fn lookups_find_declared_entries_only() {
        assert_eq!(
            SAMPLE.endpoint(EndpointId::Outlet).map(|e| e.id),
            Some(EndpointId::Outlet)
        );
        assert_eq!(SAMPLE.endpoint(EndpointId::ReachLeft), None);
        assert_eq!(
            SAMPLE.runtime(RuntimeKind::Cpu).map(|r| r.status),
            Some(RuntimeStatus::Reference)
        );
        assert_eq!(SAMPLE.runtime(RuntimeKind::Wgpu), None);
        assert_eq!(SAMPLE.option("step").and_then(|o| o.default), Some("fixed"));
        assert_eq!(SAMPLE.option("missing"), None);
        assert_eq!(
            SAMPLE.unsupported_by_id("seepage").map(|u| u.reason),
            Some(UnsupportedReason::OutsideModel)
        );
        assert_eq!(SAMPLE.unsupported_by_id("missing"), None);
        assert_eq!(
            SAMPLE.limit_by_id("max-stage").map(|l| l.id),
            Some("max-stage")
        );
        assert_eq!(SAMPLE.limit_by_id("missing"), None);
    }
}
