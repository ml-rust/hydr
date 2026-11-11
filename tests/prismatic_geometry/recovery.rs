use super::fixtures::INVERSE_CASES;
use super::support::{self, Client, Result};
use hydr::channel::PrismaticRecoveryConfig;
use numr::dtype::DType;
use numr::ops::BinaryOps;
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

pub(crate) fn run<R>(client: R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: Client<R>,
{
    let batch = support::batch::<R>(&client)?;
    let config = PrismaticRecoveryConfig::default();
    let cases: Vec<_> = INVERSE_CASES.iter().rev().collect();
    assert_eq!(cases.len(), 86);
    let owners: Vec<_> = cases
        .iter()
        .map(|case| i64::try_from(case.profile))
        .collect::<std::result::Result<_, _>>()?;
    let bits: Vec<_> = cases.iter().map(|case| case.area_bits).collect();
    let values: Vec<_> = bits.iter().copied().map(f64::from_bits).collect();
    let mut contiguous_properties = None;
    for strided in [false, true] {
        let (owner_input, area_input) = support::inputs::<R>(&client, &owners, &values, strided)?;
        let recovery = batch.recover_depths_tensor(&owner_input, &area_input, &config)?;
        support::snapshots(
            recovery.owners(),
            recovery.requested_areas_m2(),
            recovery.dry_mask(),
            &owners,
            &bits,
        )?;
        for bounds in [
            recovery.depth_m(),
            recovery.forward_area_m2(),
            recovery.top_width_m(),
            recovery.hydrostatic_integral_m3(),
            recovery.hydraulic_depth_m(),
        ] {
            support::native_bounds(bounds, cases.len());
        }
        support::dry_placeholders(recovery.top_width_m(), &bits)?;
        support::dry_placeholders(recovery.hydraulic_depth_m(), &bits)?;
        let properties = recovery.properties()?;
        assert_eq!(properties.len(), cases.len());
        for (index, (row, case)) in properties.iter().zip(&cases).enumerate() {
            let profile = support::profile(case.profile)?;
            let label = format!(
                "recovery profile={}, row={index}, strided={strided}",
                profile.id
            );
            let requested = f64::from_bits(case.area_bits);
            assert_eq!(row.section_id, profile.id);
            assert_eq!(row.requested_area_m2.to_bits(), case.area_bits);
            assert_eq!(row.dry, requested == 0.0, "{label}");
            support::enclosed(row.depth_m, case.depth, &format!("{label}, depth"));
            support::enclosed(
                row.forward_area_m2,
                case.forward_area,
                &format!("{label}, forward area"),
            );
            assert!(
                row.forward_area_m2[0] <= requested && requested <= row.forward_area_m2[1],
                "{label}"
            );
            support::optional(row.top_width_m, case.top_width, &format!("{label}, width"))?;
            support::enclosed(
                row.hydrostatic_integral_m3,
                case.integral,
                &format!("{label}, integral"),
            );
            support::optional(
                row.hydraulic_depth_m,
                case.hydraulic_depth,
                &format!("{label}, hydraulic depth"),
            )?;
            if let Some(original_bits) = case.original_depth_bits {
                let original = f64::from_bits(original_bits);
                assert!(
                    row.depth_m[0] <= original && original <= row.depth_m[1],
                    "{label}"
                );
            }
            if row.dry {
                assert_eq!(row.depth_m, [0.0, 0.0], "{label}");
                assert_eq!(row.forward_area_m2, [0.0, 0.0], "{label}");
                assert_eq!(row.hydrostatic_integral_m3, [0.0, 0.0], "{label}");
            } else {
                assert!(row.depth_m[0] > 0.0, "{label}");
            }
        }
        if let Some(expected) = &contiguous_properties {
            assert_eq!(&properties, expected);
        }
        support::overwrite_inputs::<R>(&client, &owner_input, &area_input)?;
        support::snapshots(
            recovery.owners(),
            recovery.requested_areas_m2(),
            recovery.dry_mask(),
            &owners,
            &bits,
        )?;
        assert_eq!(recovery.properties()?, properties);

        // Deliberately invalidate this result to check that compiled geometry owns separate storage.
        client.copy_into(
            recovery.forward_area_m2().lower(),
            &Tensor::zeros(&[cases.len()], DType::F64, client.device())?,
        )?;
        let (fresh_owners, fresh_areas) = support::inputs::<R>(&client, &owners, &values, strided)?;
        let repeated = batch.recover_depths_tensor(&fresh_owners, &fresh_areas, &config)?;
        assert_eq!(repeated.properties()?, properties);
        if contiguous_properties.is_none() {
            contiguous_properties = Some(properties);
        }
    }
    Ok(())
}

#[test]
fn cpu_area_recovery_preserves_conserved_operands_and_independent_geometry() -> Result {
    run::<super::runtime::Runtime>(super::runtime::client())
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_area_recovery_preserves_conserved_operands_and_independent_geometry() -> Result {
    use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
    run::<CudaRuntime>(CudaClient::new(CudaDevice::new(0))?)
}
