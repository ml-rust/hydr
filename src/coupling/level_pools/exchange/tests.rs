use super::assembly_bound;
use crate::coupling::level_pools::topology::TransferTree;
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
use numr::tensor::Tensor;

#[test]
fn incoming_and_outgoing_assembly_errors_both_charge_closed_mass()
-> Result<(), Box<dyn std::error::Error>> {
    let device = CpuDevice::new();
    let client = CpuClient::new(device.clone());
    let ids = ["in-large", "in-small", "out-large", "out-small"].map(String::from);
    let tree = TransferTree::<CpuRuntime>::new(&client, 2, &[(0, 1); 4], &ids)?;
    let small = 3.0 * f64::EPSILON / 2.0;
    let endpoints = Tensor::<CpuRuntime>::from_slice(
        &[1.0, small, 0.0, 0.0, 0.0, 0.0, 1.0, small],
        &[8],
        &device,
    )?;
    let reverse = Tensor::<CpuRuntime>::from_slice(
        &[0.0, 0.0, 1.0, small, 1.0, small, 0.0, 0.0],
        &[8],
        &device,
    )?;
    let outgoing = tree.aggregate(&client, &endpoints)?;
    let incoming = tree.aggregate(&client, &reverse)?;
    let incoming_bounds = incoming.bound.try_to_vec::<f64>()?;
    let outgoing_bounds = outgoing.bound.try_to_vec::<f64>()?;
    assert!(incoming_bounds.iter().all(|value| *value > 0.0));
    assert!(outgoing_bounds.iter().all(|value| *value > 0.0));
    let exact_charge: f64 = incoming_bounds.iter().chain(&outgoing_bounds).sum();
    let largest_single_side = incoming_bounds
        .iter()
        .chain(&outgoing_bounds)
        .copied()
        .fold(0.0, f64::max);
    assert!(2.0 * largest_single_side < exact_charge);
    let charged = assembly_bound::<CpuRuntime>(&client, &incoming.bound, &outgoing.bound, 2)?;
    assert!(charged >= exact_charge);
    Ok(())
}
