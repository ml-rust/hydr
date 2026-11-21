//! Frozen reservoir endpoint physics and continuation acceptance.

#[path = "channel_reservoir/admission.rs"]
mod admission;
#[allow(dead_code)]
#[path = "channel_boundaries/common.rs"]
mod boundary_common;
#[path = "channel_reservoir/fixtures.rs"]
mod fixtures;
#[allow(dead_code)]
#[path = "channel_reservoir/native.rs"]
mod native;
#[path = "channel_reservoir/physics.rs"]
mod physics;
#[allow(dead_code)]
#[path = "support/runtime.rs"]
mod runtime;
#[path = "channel_tensor_forcing/snapshot_bits.rs"]
mod snapshot_bits;
#[allow(dead_code)]
#[path = "channel_tensor_forcing/transport.rs"]
mod transport;

#[test]
fn native_reservoir_operands_preserve_bits_alias_isolation_and_complete_restart()
-> boundary_common::TestResult {
    native::execute::<runtime::Runtime>(runtime::client(), native::Selection::all())
}

#[test]
fn forged_reservoir_inputs_config_and_every_certificate_field_reject() -> boundary_common::TestResult
{
    native::forged_snapshots_reject()
}
