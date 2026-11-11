use super::fixtures::FORWARD_CASES;
use super::support::{self, Client, Result};
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
    assert_eq!(batch.definitions().len(), super::fixtures::PROFILES.len());
    assert!(batch.maximum_band_count() >= 3);
    let cases: Vec<_> = FORWARD_CASES.iter().rev().collect();
    assert_eq!(cases.len(), 79);
    let owners: Vec<_> = cases
        .iter()
        .map(|case| i64::try_from(case.profile))
        .collect::<std::result::Result<_, _>>()?;
    let bits: Vec<_> = cases.iter().map(|case| case.depth_bits).collect();
    let values: Vec<_> = bits.iter().copied().map(f64::from_bits).collect();
    let mut contiguous_properties = None;
    for strided in [false, true] {
        let (owner_input, depth_input) = support::inputs::<R>(&client, &owners, &values, strided)?;
        let query = batch.evaluate_depths_tensor(&owner_input, &depth_input)?;
        support::snapshots(
            query.owners(),
            query.requested_depths_m(),
            query.dry_mask(),
            &owners,
            &bits,
        )?;
        for bounds in [
            query.depth_m(),
            query.area_m2(),
            query.top_width_m(),
            query.hydrostatic_integral_m3(),
            query.hydraulic_depth_m(),
        ] {
            support::native_bounds(bounds, cases.len());
        }
        support::dry_placeholders(query.top_width_m(), &bits)?;
        support::dry_placeholders(query.hydraulic_depth_m(), &bits)?;
        let properties = query.properties()?;
        assert_eq!(properties.len(), cases.len());
        for (index, (row, case)) in properties.iter().zip(&cases).enumerate() {
            let profile = support::profile(case.profile)?;
            let label = format!(
                "forward profile={}, row={index}, strided={strided}",
                profile.id
            );
            assert_eq!(row.section_id, profile.id);
            assert_eq!(row.query_depth_m.to_bits(), case.depth_bits);
            let depth = f64::from_bits(case.depth_bits);
            assert_eq!(row.depth_m, [depth, depth], "{label}");
            assert_eq!(row.dry, depth == 0.0, "{label}");
            support::enclosed(row.area_m2, case.area, &format!("{label}, area"));
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
            if row.dry {
                assert_eq!(row.area_m2, [0.0, 0.0], "{label}");
                assert_eq!(row.hydrostatic_integral_m3, [0.0, 0.0], "{label}");
            }
        }
        if let Some(expected) = &contiguous_properties {
            assert_eq!(&properties, expected);
        }
        support::overwrite_inputs::<R>(&client, &owner_input, &depth_input)?;
        support::snapshots(
            query.owners(),
            query.requested_depths_m(),
            query.dry_mask(),
            &owners,
            &bits,
        )?;
        assert_eq!(query.properties()?, properties);

        // Deliberately invalidate this result to check that compiled geometry owns separate storage.
        client.copy_into(
            query.area_m2().lower(),
            &Tensor::zeros(&[cases.len()], DType::F64, client.device())?,
        )?;
        let (fresh_owners, fresh_depths) =
            support::inputs::<R>(&client, &owners, &values, strided)?;
        let repeated = batch.evaluate_depths_tensor(&fresh_owners, &fresh_depths)?;
        assert_eq!(repeated.properties()?, properties);
        if contiguous_properties.is_none() {
            contiguous_properties = Some(properties);
        }
    }
    Ok(())
}

#[test]
fn cpu_forward_geometry_encloses_original_coordinate_references() -> Result {
    run::<super::runtime::Runtime>(super::runtime::client())
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_forward_geometry_encloses_original_coordinate_references() -> Result {
    use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
    run::<CudaRuntime>(CudaClient::new(CudaDevice::new(0))?)
}
