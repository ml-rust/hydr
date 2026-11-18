//! Signal bindings: each controller signal samples the stage of one coordinator node.
//!
//! The node table holds every pool and every surface or channel cell a link attaches to.

use super::{BoundSignal, SurfaceChannelSignal};
use crate::coupling::surface_channels::context::Context;
use crate::coupling::surface_channels::structures::SurfaceChannelNode;
use crate::coupling::surface_channels::{SurfaceChannelDefinition, SurfaceChannelError};
use crate::structures::OperatingControlDefinition;
use numr::dtype::DType;
use numr::runtime::Runtime;
use std::collections::{BTreeMap, BTreeSet};

/// Rejects with `operation-ownership`.
fn rejected(detail: String) -> SurfaceChannelError {
    SurfaceChannelError::input(format!("operation-ownership: {detail}"))
}

/// Binds every controller signal of `controls` (gate, then pump) to its node's donor group.
///
/// Returns the bound signals of each child in its controller order.
///
/// # Errors
/// Returns [`SurfaceChannelError::Input`] naming `operation-ownership` for a controller identity
/// in both children, a signal naming no controller signal, a signal bound twice, a signal on a
/// node outside the node table, or a controller signal without a binding. A signal outside the
/// surface datum rejects with `datum-or-gravity-mismatch`.
pub(super) fn bind_signals<R: Runtime<DType = DType>>(
    definition: &SurfaceChannelDefinition,
    context: &Context<R>,
    controls: (&[OperatingControlDefinition], &[OperatingControlDefinition]),
    signals: Vec<SurfaceChannelSignal>,
) -> Result<(Vec<BoundSignal>, Vec<BoundSignal>), SurfaceChannelError> {
    let datum = definition
        .setup()
        .surface
        .domain
        .reference()
        .vertical_datum();
    let mut controllers = BTreeSet::new();
    for control in controls.0.iter().chain(controls.1) {
        if !controllers.insert(control.controller_id.as_str()) {
            return Err(rejected(format!(
                "controller={:?} repeats across operating children: give each controller a unique identity",
                control.controller_id
            )));
        }
    }
    let mut bound: BTreeMap<String, (String, usize)> = BTreeMap::new();
    for signal in signals {
        let control = controls
            .0
            .iter()
            .chain(controls.1)
            .find(|c| c.controller_id == signal.controller_id)
            .filter(|c| c.signal.signal_id == signal.signal_id)
            .ok_or_else(|| {
                rejected(format!(
                    "signal controller={:?} signal={:?} names no controller signal: bind the signal of an operating child's controller",
                    signal.controller_id, signal.signal_id
                ))
            })?;
        if control.signal.datum != datum {
            return Err(SurfaceChannelError::input(format!(
                "datum-or-gravity-mismatch: signal controller={:?} datum={:?}, surface vertical_datum={datum:?}: express the signal in the surface datum",
                signal.controller_id, control.signal.datum
            )));
        }
        let node = signal.node.identity();
        let group = group_of(definition, context, &signal.node).ok_or_else(|| {
            rejected(format!(
                "signal controller={:?} node={node:?} is no coordinator node: bind a linked surface cell, a linked channel cell, or a pool",
                signal.controller_id
            ))
        })?;
        if bound
            .insert(signal.controller_id.clone(), (node, group))
            .is_some()
        {
            return Err(rejected(format!(
                "signal controller={:?} is bound twice: bind each controller signal once",
                signal.controller_id
            )));
        }
    }
    let resolve = |controls: &[OperatingControlDefinition]| {
        controls
            .iter()
            .map(|control| {
                let (node, group) = bound.get(&control.controller_id).ok_or_else(|| {
                    rejected(format!(
                        "controller={:?} signal={:?} has no binding: bind every controller signal to a node",
                        control.controller_id, control.signal.signal_id
                    ))
                })?;
                Ok(BoundSignal {
                    controller_id: control.controller_id.clone(),
                    signal_id: control.signal.signal_id.clone(),
                    support_id: control.signal.support_id.clone(),
                    datum: control.signal.datum.clone(),
                    node: node.clone(),
                    group: *group,
                })
            })
            .collect::<Result<Vec<_>, SurfaceChannelError>>()
    };
    Ok((resolve(controls.0)?, resolve(controls.1)?))
}

/// Donor group of `node` in `S + C + P`, or `None` outside the node table.
fn group_of<R: Runtime<DType = DType>>(
    definition: &SurfaceChannelDefinition,
    context: &Context<R>,
    node: &SurfaceChannelNode,
) -> Option<usize> {
    if let SurfaceChannelNode::Pool(id) = node {
        let position = definition
            .pools
            .as_ref()?
            .ids
            .iter()
            .position(|p| p == id)?;
        return Some(context.surface_count + context.slot_count + position);
    }
    let identity = node.identity();
    context
        .link_elements
        .iter()
        .zip(&context.link_groups)
        .find_map(|(element, groups)| {
            if element.endpoint_a == identity {
                Some(groups[0])
            } else if element.endpoint_b == identity {
                Some(groups[1])
            } else {
                None
            }
        })
}
