use super::fixtures::{CASES, GEOMETRY_ORDER};
use super::point_support;
use super::support::{self, Client, Result};
use hydr::channel::PrismaticRecoveryConfig;
use numr::dtype::DType;
use numr::runtime::Runtime;

fn run<R>(client: R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: Client<R>,
{
    assert_eq!(CASES.len(), 75);
    let section = point_support::batch::<R>(&client)?;
    let batch = section.certify_perimeter()?;
    let definitions = batch.definitions().to_vec();
    let config = PrismaticRecoveryConfig::default();
    let cases: Vec<_> = CASES.iter().rev().filter(|case| case.resolved).collect();
    assert_eq!(cases.len(), 73);
    let owners: Vec<_> = cases
        .iter()
        .map(|case| i64::try_from(case.owner))
        .collect::<std::result::Result<_, _>>()?;
    let high: Vec<_> = cases
        .iter()
        .map(|case| f64::from_bits(case.input_high))
        .collect();
    let low: Vec<_> = cases
        .iter()
        .map(|case| f64::from_bits(case.input_low))
        .collect();
    let mut baseline = None;
    for strided in [false, true] {
        let (owner, area) = point_support::inputs::<R>(&client, &owners, &high, &low, strided)?;
        let high_bits = point_support::bits(area.high())?;
        let low_bits = point_support::bits(area.low())?;
        let direct = section.recover_area_points_tensor(&owner, &area, &config)?;
        let recovered = batch.recover_area_points_tensor(&owner, &area, &config)?;
        let geometry = recovered.geometry();
        assert_eq!(geometry.properties()?, direct.properties()?);
        assert_eq!(geometry.owners().try_to_vec::<i64>()?, owners);
        assert_eq!(
            point_support::bits(geometry.requested_areas_m2().high())?,
            high_bits
        );
        assert_eq!(
            point_support::bits(geometry.requested_areas_m2().low())?,
            low_bits
        );
        let native_perimeter = support::endpoints(recovered.wetted_perimeter_m(), cases.len())?;
        let native_radius = support::endpoints(recovered.hydraulic_radius_m(), cases.len())?;
        let rows = recovered.properties()?;
        let direct_rows = direct.properties()?;
        assert_eq!(rows.len(), cases.len());
        for (index, ((row, expected_geometry), case)) in
            rows.iter().zip(&direct_rows).zip(&cases).enumerate()
        {
            let label = format!("{}, strided={strided}", case.id);
            assert_eq!(&row.geometry, expected_geometry, "{label}");
            assert_eq!(
                row.geometry.requested_area_high_m2.to_bits(),
                high_bits[index],
                "{label}"
            );
            assert_eq!(
                row.geometry.requested_area_low_m2.to_bits(),
                low_bits[index],
                "{label}"
            );
            assert_eq!(row.geometry.dry, case.dry, "{label}");
            for ((actual, truth), quantity) in [
                Some(row.geometry.depth_m),
                Some(row.geometry.forward_area_m2),
                row.geometry.top_width_m,
                Some(row.geometry.hydrostatic_integral_m3),
                row.geometry.hydraulic_depth_m,
            ]
            .into_iter()
            .zip(case.geometry)
            .zip(GEOMETRY_ORDER)
            {
                match (actual, truth) {
                    (Some(actual), Some(truth)) => {
                        point_support::enclosed(actual, truth, &format!("{label}, {quantity}"))
                    }
                    (None, None) => {}
                    _ => {
                        return Err(format!(
                            "{label}, {quantity}: geometry wet/dry fields disagree"
                        )
                        .into());
                    }
                }
            }
            for &band_index in case.true_band_indices {
                assert!(
                    row.geometry
                        .bands
                        .iter()
                        .any(|band| band.band_index == band_index),
                    "{label}: missing physical band {band_index}"
                );
            }
            assert_eq!(native_perimeter[index], row.wetted_perimeter_m, "{label}");
            assert_eq!(
                native_radius[index],
                row.hydraulic_radius_m.unwrap_or([0.0, 0.0]),
                "{label}"
            );
            support::contact(
                row.wetted_perimeter_m,
                case.perimeter,
                case.perimeter_allowance,
                &format!("{label}, perimeter"),
            );
            match (row.hydraulic_radius_m, case.radius) {
                (Some(actual), Some(truth)) => {
                    support::contact(
                        actual,
                        truth,
                        case.radius_allowance,
                        &format!("{label}, radius"),
                    );
                    if case.radius_positive_lower {
                        assert!(actual[0] > 0.0, "{label}: positive radius lower bound");
                    }
                    if case.conditioned_radius {
                        let reference_lower = f64::from_bits(truth[0]);
                        assert!(
                            reference_lower > 0.0,
                            "{label}: conditioned radius reference"
                        );
                        assert!(
                            f64::from_bits(case.radius_allowance) <= reference_lower * 1e-10,
                            "{label}: conditioned radius allowance"
                        );
                    }
                }
                (None, None) => {
                    assert!(case.dry, "{label}");
                    assert_eq!(row.wetted_perimeter_m, [0.0, 0.0], "{label}");
                    assert_eq!(native_radius[index], [0.0, 0.0], "{label}");
                }
                _ => return Err(format!("{label}: radius wet/dry fields disagree").into()),
            }
            let scale = row.geometry.depth_m[0]
                .abs()
                .max(row.geometry.depth_m[1].abs());
            assert!(
                row.geometry.depth_m[1] - row.geometry.depth_m[0]
                    <= config.depth_atol_m + config.depth_rtol * scale,
                "{label}: geometric depth width"
            );
        }
        let find = |id: &str| {
            cases
                .iter()
                .position(|case| case.id == id)
                .ok_or_else(|| format!("Missing physical distinction reference {id}"))
        };
        let symmetric = &rows[find("symmetric_pointed_same_area_one")?];
        let asymmetric = &rows[find("asymmetric_pointed_same_area_same_area_one")?];
        assert!(symmetric.wetted_perimeter_m[1] < asymmetric.wetted_perimeter_m[0]);
        let before = rows[find("radius_decreases_across_widening_band_at_widening")?]
            .hydraulic_radius_m
            .ok_or("Missing radius before widening")?;
        let after = rows[find("radius_decreases_across_widening_band_after_widening")?]
            .hydraulic_radius_m
            .ok_or("Missing radius after widening")?;
        assert!(after[1] < before[0]);
        let cap = &rows[find("cap_benches_excluded_cap")?];
        assert!(cap.wetted_perimeter_m[0] <= 4.0 && 4.0 <= cap.wetted_perimeter_m[1]);
        assert!(cap.wetted_perimeter_m[1] < 6.0);
        if let Some(expected) = &baseline {
            assert_eq!(&rows, expected);
        }
        point_support::overwrite::<R>(&client, &owner)?;
        for component in [area.high(), area.low(), area.abs_error()] {
            point_support::overwrite::<R>(&client, component)?;
        }
        assert_eq!(geometry.owners().try_to_vec::<i64>()?, owners);
        assert_eq!(
            point_support::bits(geometry.requested_areas_m2().high())?,
            high_bits
        );
        assert_eq!(
            point_support::bits(geometry.requested_areas_m2().low())?,
            low_bits
        );
        assert_eq!(recovered.properties()?, rows);
        let (next_owner, next_area) =
            point_support::inputs::<R>(&client, &owners, &high, &low, strided)?;
        let retained = batch.recover_area_points_tensor(&next_owner, &next_area, &config)?;
        for endpoint in [
            recovered.wetted_perimeter_m().lower(),
            recovered.wetted_perimeter_m().upper(),
            recovered.hydraulic_radius_m().lower(),
            recovered.hydraulic_radius_m().upper(),
            geometry.band_coordinates().depth_above_start_m().lower(),
        ] {
            point_support::overwrite::<R>(&client, endpoint)?;
        }
        assert_eq!(retained.properties()?, rows);
        assert_eq!(
            batch
                .recover_area_points_tensor(&next_owner, &next_area, &config)?
                .properties()?,
            rows
        );
        assert_eq!(batch.definitions(), definitions.as_slice());
        baseline = Some(rows);
    }
    Ok(())
}

#[test]
fn configured_cpu_perimeter_area_points_preserve_geometry_contact_and_radius() -> Result {
    use numr::runtime::cpu::ParallelismConfig;
    let original = super::runtime::client();
    let before = original.parallelism();
    let config = ParallelismConfig::new(Some(2), Some(257));
    let client = original.with_parallelism(config);
    let retained = client.clone();
    run::<super::runtime::Runtime>(client)?;
    assert_eq!(retained.parallelism(), config);
    assert_eq!(original.parallelism(), before);
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_perimeter_area_points_preserve_geometry_contact_and_radius() -> Result {
    use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
    run::<CudaRuntime>(CudaClient::new(CudaDevice::new(0))?)
}
