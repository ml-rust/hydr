//! Shared vocabulary for component capability declarations.
//!
//! A declaration states what one component admits: precision, arithmetic, runtimes,
//! endpoints, regimes, sources and options. It also states what the component refuses and
//! where its scope ends. The types hold `&'static` data only and carry no behavior beyond lookup.
//!
//! Each component crate declares its own capabilities next to its admission checks.
//! A private `static` holds the declaration. A public function returns it as
//! `&'static ComponentCapabilities`. The function is `capabilities()` when its module holds one
//! component, and `<component>_capabilities()` when it holds several. Each `doc` path is relative to the repository root.
//! A test in `tests/component_capabilities` backs every claim a declaration makes.
//!
//! Every enum has a stable kebab-case `id`. Persisted decisions store these ids, so an id
//! never changes once released.

mod boundary;
mod component;
mod declaration;
mod numeric;
mod regime;
mod source;

#[cfg(test)]
mod id_check;

pub use boundary::{BoundaryKind, Endpoint, EndpointId};
pub use component::ComponentId;
pub use declaration::{
    ComponentCapabilities, Limit, OptionSet, Summary, Unsupported, UnsupportedReason,
};
pub use numeric::{ArithmeticRequirement, Precision, RuntimeKind, RuntimeStatus, RuntimeSupport};
pub use regime::{Regime, RegimeScope, RegimeSupport};
pub use source::SourceKind;
