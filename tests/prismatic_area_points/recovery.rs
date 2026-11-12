use super::fixtures::{CASES, FORMULAS, GEOMETRY_ORDER, LOCAL_ORDER, PROFILES};
use super::support::{self, Client, Result};
use hydr::channel::PrismaticRecoveryConfig;
use numr::dtype::DType;
use numr::runtime::Runtime;

fn run<R>(client: R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: Client<R>,
{
    let batch = support::batch::<R>(&client)?;
    let config = PrismaticRecoveryConfig::default();
    assert_eq!(FORMULAS.len(), 1);
    for formula in FORMULAS {
        assert!(!formula.id.is_empty());
        assert!(f64::from_bits(formula.area) > 0.0);
        assert!(f64::from_bits(formula.slope).is_subnormal());
        assert!(f64::from_bits(formula.depth[0]) > 0.0);
        assert_eq!(formula.depth[0], formula.depth[1]);
    }
    let cases: Vec<_> = CASES.iter().rev().filter(|case| case.resolved).collect();
    assert_eq!(cases.len(), 44);
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
        let (owner_input, area_input) =
            support::inputs::<R>(&client, &owners, &high, &low, strided)?;
        let expected_high = support::bits(area_input.high())?;
        let expected_low = support::bits(area_input.low())?;
        let recovery = batch.recover_area_points_tensor(&owner_input, &area_input, &config)?;
        assert_eq!(recovery.owners().try_to_vec::<i64>()?, owners);
        assert_eq!(
            support::bits(recovery.requested_areas_m2().high())?,
            expected_high
        );
        assert_eq!(
            support::bits(recovery.requested_areas_m2().low())?,
            expected_low
        );
        assert!(
            recovery
                .requested_areas_m2()
                .abs_error()
                .try_to_vec::<f64>()?
                .iter()
                .all(|&x| x == 0.0)
        );
        assert_eq!(
            recovery.dry_mask().try_to_vec::<f64>()?,
            cases
                .iter()
                .map(|case| if case.dry { 1.0 } else { 0.0 })
                .collect::<Vec<_>>()
        );
        let native_geometry = [
            recovery.depth_m(),
            recovery.forward_area_m2(),
            recovery.top_width_m(),
            recovery.hydrostatic_integral_m3(),
            recovery.hydraulic_depth_m(),
        ]
        .into_iter()
        .map(|bounds| {
            support::interval(bounds, &[cases.len()]);
            Ok((
                bounds.lower().try_to_vec::<f64>()?,
                bounds.upper().try_to_vec::<f64>()?,
            ))
        })
        .collect::<Result<Vec<_>>>()?;
        let coordinates = recovery.band_coordinates();
        let columns = batch.maximum_band_count();
        assert_eq!(
            coordinates.candidate_mask().shape(),
            &[cases.len(), columns]
        );
        assert_eq!(coordinates.candidate_mask().dtype(), DType::F64);
        let mask = coordinates.candidate_mask().try_to_vec::<f64>()?;
        let native_coordinates = [
            coordinates.area_above_start_m2(),
            coordinates.area_below_end_m2(),
            coordinates.depth_above_start_m(),
            coordinates.depth_below_end_m(),
        ];
        let native_bounds = native_coordinates
            .iter()
            .map(|bounds| {
                support::interval(bounds, &[cases.len(), columns]);
                Ok((
                    bounds.lower().try_to_vec::<f64>()?,
                    bounds.upper().try_to_vec::<f64>()?,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        let properties = recovery.properties()?;
        assert_eq!(properties.len(), cases.len());
        for (row_index, (row, case)) in properties.iter().zip(&cases).enumerate() {
            let label = format!("{}, strided={strided}", case.id);
            assert_eq!(
                row.section_id,
                PROFILES
                    .get(case.owner)
                    .ok_or("Missing area point owner")?
                    .id
            );
            assert_eq!(
                row.requested_area_high_m2.to_bits(),
                expected_high[row_index],
                "{label}"
            );
            assert_eq!(
                row.requested_area_low_m2.to_bits(),
                expected_low[row_index],
                "{label}"
            );
            assert_eq!(row.dry, case.dry, "{label}");
            for (field, (actual, truth)) in [
                Some(row.depth_m),
                Some(row.forward_area_m2),
                row.top_width_m,
                Some(row.hydrostatic_integral_m3),
                row.hydraulic_depth_m,
            ]
            .into_iter()
            .zip(case.geometry)
            .enumerate()
            {
                let quantity = GEOMETRY_ORDER[field];
                let field_label = format!("{label}, quantity={quantity}");
                let native = [
                    native_geometry[field].0[row_index],
                    native_geometry[field].1[row_index],
                ];
                assert_eq!(native, actual.unwrap_or([0.0, 0.0]), "{field_label}");
                match (actual, truth) {
                    (Some(actual), Some(truth)) => support::enclosed(actual, truth, &field_label),
                    (None, None) => {}
                    _ => {
                        return Err(
                            format!("{field_label}: geometry wet/dry fields disagree").into()
                        );
                    }
                }
            }
            let depth_scale = row.depth_m[0].abs().max(row.depth_m[1].abs());
            assert!(
                row.depth_m[1] - row.depth_m[0]
                    <= config.depth_atol_m + config.depth_rtol * depth_scale,
                "{label}"
            );
            if case.dry {
                assert_eq!(row.depth_m, [0.0, 0.0]);
                assert_eq!(row.forward_area_m2, [0.0, 0.0]);
                assert_eq!(row.hydrostatic_integral_m3, [0.0, 0.0]);
            } else {
                assert!(row.depth_m[0] > 0.0, "{label}");
            }
            assert_eq!(row.bands.len(), case.bands.len(), "{label}");
            for column in 0..columns {
                let index = row_index * columns + column;
                let reference = case.bands.iter().find(|band| band.band_index == column);
                assert_eq!(
                    mask[index],
                    if reference.is_some() { 1.0 } else { 0.0 },
                    "{label}, band={column}"
                );
                for (field, (lower, upper)) in native_bounds.iter().enumerate() {
                    let quantity = LOCAL_ORDER[field];
                    let field_label = format!("{label}, band={column}, quantity={quantity}");
                    let actual = [lower[index], upper[index]];
                    if let Some(reference) = reference {
                        support::enclosed(actual, reference.bounds[field], &field_label);
                        if reference.exact_zero[field] {
                            assert_eq!(actual, [0.0, 0.0], "{field_label}");
                        } else {
                            let truth = reference.bounds[field].map(f64::from_bits);
                            let error = (actual[0] - truth[1])
                                .abs()
                                .max((actual[1] - truth[0]).abs());
                            assert!(
                                error <= f64::from_bits(reference.allowances[field]),
                                "{field_label}: {actual:?}"
                            );
                        }
                    } else {
                        assert_eq!(actual, [0.0, 0.0], "{field_label}");
                    }
                }
            }
            for band in &row.bands {
                let index = row_index * columns + band.band_index;
                for (field, actual) in [
                    band.area_above_start_m2,
                    band.area_below_end_m2,
                    band.depth_above_start_m,
                    band.depth_below_end_m,
                ]
                .into_iter()
                .enumerate()
                {
                    let quantity = LOCAL_ORDER[field];
                    assert_eq!(
                        actual,
                        [native_bounds[field].0[index], native_bounds[field].1[index]],
                        "{label}, band={}, quantity={quantity}",
                        band.band_index
                    );
                }
            }
        }
        if let Some(expected) = &baseline {
            assert_eq!(&properties, expected);
        }
        for input in [area_input.high(), area_input.low(), area_input.abs_error()] {
            support::overwrite::<R>(&client, input)?;
        }
        support::overwrite::<R>(&client, &owner_input)?;
        assert_eq!(
            support::bits(recovery.requested_areas_m2().high())?,
            expected_high
        );
        assert_eq!(
            support::bits(recovery.requested_areas_m2().low())?,
            expected_low
        );
        assert_eq!(recovery.owners().try_to_vec::<i64>()?, owners);
        assert_eq!(recovery.properties()?, properties);
        support::overwrite::<R>(&client, recovery.forward_area_m2().lower())?;
        support::overwrite::<R>(&client, coordinates.depth_above_start_m().lower())?;
        let (fresh_owners, fresh_areas) =
            support::inputs::<R>(&client, &owners, &high, &low, strided)?;
        assert_eq!(
            batch
                .recover_area_points_tensor(&fresh_owners, &fresh_areas, &config)?
                .properties()?,
            properties
        );
        if baseline.is_none() {
            baseline = Some(properties);
        }
    }
    Ok(())
}

#[test]
fn cpu_area_points_retain_independent_geometry_and_endpoint_coordinates() -> Result {
    run::<super::runtime::Runtime>(super::runtime::client())
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_area_points_retain_independent_geometry_and_endpoint_coordinates() -> Result {
    use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
    run::<CudaRuntime>(CudaClient::new(CudaDevice::new(0))?)
}
