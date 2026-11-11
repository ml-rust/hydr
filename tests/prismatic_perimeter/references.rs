use super::fixtures::{FORWARD, INVERSE, PROFILES};
use super::model::Case;
use super::support::{self, Client, Result};
use hydr::channel::PrismaticRecoveryConfig;
use numr::dtype::DType;
use numr::runtime::Runtime;

type Operands = (Vec<i64>, Vec<u64>, Vec<f64>);
type ContactProjection = ([f64; 2], Option<[f64; 2]>);

fn cases(source: &'static [Case]) -> Vec<&'static Case> {
    // Interleave distant owners and retain repeated owners in each native call.
    let mut rows = Vec::with_capacity(source.len());
    let mut front = 0;
    let mut back = source.len();
    while front < back {
        rows.push(&source[front]);
        front += 1;
        if front < back {
            back -= 1;
            rows.push(&source[back]);
        }
    }
    rows
}

fn operands(rows: &[&Case]) -> Result<Operands> {
    let owners = rows
        .iter()
        .map(|row| i64::try_from(row.owner))
        .collect::<std::result::Result<_, _>>()?;
    let bits: Vec<_> = rows.iter().map(|row| row.operand).collect();
    let values = bits.iter().copied().map(f64::from_bits).collect();
    Ok((owners, bits, values))
}

fn contact(perimeter: [f64; 2], radius: Option<[f64; 2]>, case: &Case, label: &str) -> Result {
    support::enclosed(
        perimeter,
        case.perimeter,
        &format!("{label}: perimeter"),
        1e-9,
        1e-10,
    );
    match (radius, case.radius) {
        (Some(actual), Some(truth)) => {
            support::enclosed(actual, truth, &format!("{label}: radius"), 1e-9, 1e-10)
        }
        (None, None) => assert_eq!(perimeter, [0.0, 0.0], "{label}"),
        _ => return Err(format!("{label}: hydraulic radius disagrees with dry reference").into()),
    }
    Ok(())
}

fn native_projection<R: Runtime<DType = DType>>(
    perimeter: &solvr::common::arithmetic::Interval<R>,
    radius: &solvr::common::arithmetic::Interval<R>,
    expected: &[ContactProjection],
) -> Result {
    support::native(perimeter, expected.len());
    support::native(radius, expected.len());
    let pl = perimeter.lower().try_to_vec::<f64>()?;
    let pu = perimeter.upper().try_to_vec::<f64>()?;
    let rl = radius.lower().try_to_vec::<f64>()?;
    let ru = radius.upper().try_to_vec::<f64>()?;
    for (index, &(p, r)) in expected.iter().enumerate() {
        assert_eq!([pl[index], pu[index]], p);
        assert_eq!([rl[index], ru[index]], r.unwrap_or([0.0, 0.0]));
    }
    Ok(())
}

fn forward<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: Client<R>,
{
    let batch = support::batch::<R>(client)?;
    assert_eq!(PROFILES.len(), 9);
    assert_eq!(batch.definitions().len(), 9);
    assert!(batch.maximum_band_count() >= 2);
    let definitions = batch.definitions().to_vec();
    assert_eq!(FORWARD.len(), 58);
    let rows = cases(FORWARD);
    let (owners, bits, values) = operands(&rows)?;
    let mut baseline = None;
    for strided in [false, true] {
        let (owner, depth) = support::inputs::<R>(client, &owners, &values, strided)?;
        let query = batch.evaluate_depths_tensor(&owner, &depth)?;
        let geometry = query.geometry();
        support::snapshots(
            geometry.owners(),
            geometry.requested_depths_m(),
            geometry.dry_mask(),
            &owners,
            &bits,
        )?;
        let properties = query.properties()?;
        let native_geometry = geometry.properties()?;
        assert_eq!(properties.len(), rows.len());
        for (index, ((row, reference), native)) in properties
            .iter()
            .zip(&rows)
            .zip(&native_geometry)
            .enumerate()
        {
            let label = format!(
                "forward owner={}, row={index}, strided={strided}",
                reference.owner
            );
            assert_eq!(&row.geometry, native);
            assert_eq!(row.geometry.section_id, PROFILES[reference.owner].id);
            assert_eq!(row.geometry.query_depth_m.to_bits(), reference.operand);
            assert_eq!(row.geometry.dry, f64::from_bits(reference.operand) == 0.0);
            contact(
                row.wetted_perimeter_m,
                row.hydraulic_radius_m,
                reference,
                &label,
            )?;
        }
        let expected: Vec<_> = properties
            .iter()
            .map(|row| (row.wetted_perimeter_m, row.hydraulic_radius_m))
            .collect();
        native_projection(
            query.wetted_perimeter_m(),
            query.hydraulic_radius_m(),
            &expected,
        )?;
        if let Some(expected) = &baseline {
            assert_eq!(&properties, expected);
        }
        support::overwrite::<R>(client, &[&owner, &depth])?;
        support::snapshots(
            geometry.owners(),
            geometry.requested_depths_m(),
            geometry.dry_mask(),
            &owners,
            &bits,
        )?;
        assert_eq!(query.properties()?, properties);
        // Query aliases cannot mutate the compiled contact table or another result.
        let (next_owner, next_depth) = support::inputs::<R>(client, &owners, &values, strided)?;
        let retained = batch.evaluate_depths_tensor(&next_owner, &next_depth)?;
        support::overwrite::<R>(
            client,
            &[
                query.wetted_perimeter_m().lower(),
                query.wetted_perimeter_m().upper(),
                query.hydraulic_radius_m().lower(),
                query.hydraulic_radius_m().upper(),
            ],
        )?;
        assert_eq!(retained.properties()?, properties);
        assert_eq!(
            batch
                .evaluate_depths_tensor(&next_owner, &next_depth)?
                .properties()?,
            properties
        );
        assert_eq!(batch.definitions(), definitions.as_slice());
        baseline = Some(properties);
    }
    Ok(())
}

fn recovery<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: Client<R>,
{
    let batch = support::batch::<R>(client)?;
    let config = PrismaticRecoveryConfig::default();
    assert_eq!(config.depth_atol_m, 1e-12);
    assert_eq!(config.depth_rtol, 1e-12);
    assert_eq!(INVERSE.len(), 27);
    let rows = cases(INVERSE);
    let (owners, bits, values) = operands(&rows)?;
    let mut baseline = None;
    for strided in [false, true] {
        let (owner, area) = support::inputs::<R>(client, &owners, &values, strided)?;
        let recovered = batch.recover_depths_tensor(&owner, &area, &config)?;
        let geometry = recovered.geometry();
        support::snapshots(
            geometry.owners(),
            geometry.requested_areas_m2(),
            geometry.dry_mask(),
            &owners,
            &bits,
        )?;
        let properties = recovered.properties()?;
        let native_geometry = geometry.properties()?;
        assert_eq!(properties.len(), rows.len());
        for (index, ((row, reference), native)) in properties
            .iter()
            .zip(&rows)
            .zip(&native_geometry)
            .enumerate()
        {
            let label = format!(
                "recovery owner={}, row={index}, strided={strided}",
                reference.owner
            );
            assert_eq!(&row.geometry, native);
            assert_eq!(row.geometry.section_id, PROFILES[reference.owner].id);
            assert_eq!(row.geometry.requested_area_m2.to_bits(), reference.operand);
            let truth = reference
                .depth
                .ok_or("Missing frozen inverse depth endpoints")?;
            support::enclosed(row.geometry.depth_m, truth, &label, 1e-12, 0.0);
            let requested = f64::from_bits(reference.operand);
            assert!(
                row.geometry.forward_area_m2[0] <= requested
                    && requested <= row.geometry.forward_area_m2[1]
            );
            assert_eq!(row.geometry.dry, requested == 0.0);
            contact(
                row.wetted_perimeter_m,
                row.hydraulic_radius_m,
                reference,
                &label,
            )?;
        }
        let expected: Vec<_> = properties
            .iter()
            .map(|row| (row.wetted_perimeter_m, row.hydraulic_radius_m))
            .collect();
        native_projection(
            recovered.wetted_perimeter_m(),
            recovered.hydraulic_radius_m(),
            &expected,
        )?;
        if let Some(expected) = &baseline {
            assert_eq!(&properties, expected);
        }
        support::overwrite::<R>(client, &[&owner, &area])?;
        support::snapshots(
            geometry.owners(),
            geometry.requested_areas_m2(),
            geometry.dry_mask(),
            &owners,
            &bits,
        )?;
        assert_eq!(recovered.properties()?, properties);
        let (next_owner, next_area) = support::inputs::<R>(client, &owners, &values, strided)?;
        let retained = batch.recover_depths_tensor(&next_owner, &next_area, &config)?;
        support::overwrite::<R>(
            client,
            &[
                recovered.wetted_perimeter_m().lower(),
                recovered.wetted_perimeter_m().upper(),
                recovered.hydraulic_radius_m().lower(),
                recovered.hydraulic_radius_m().upper(),
            ],
        )?;
        assert_eq!(retained.properties()?, properties);
        assert_eq!(
            batch
                .recover_depths_tensor(&next_owner, &next_area, &config)?
                .properties()?,
            properties
        );
        baseline = Some(properties);
    }
    // Signed zero remains a conserved operand, including its original sign bit.
    let (owners, areas) = support::inputs::<R>(client, &[7], &[-0.0], true)?;
    let recovered = batch.recover_depths_tensor(&owners, &areas, &config)?;
    support::snapshots(
        recovered.geometry().owners(),
        recovered.geometry().requested_areas_m2(),
        recovered.geometry().dry_mask(),
        &[7],
        &[(-0.0f64).to_bits()],
    )?;
    let rows = recovered.properties()?;
    let dry = rows.first().ok_or("Missing signed-zero recovery")?;
    assert_eq!(dry.wetted_perimeter_m, [0.0, 0.0]);
    assert_eq!(dry.hydraulic_radius_m, None);
    Ok(())
}

fn physical_distinctions<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: Client<R>,
{
    let batch = support::batch::<R>(client)?;
    for strided in [false, true] {
        let (owners, depths) = support::inputs::<R>(
            client,
            &[1, 2, 8, 8, 7],
            &[1.0, 1.0, 1.0, 1.0 + 1.0 / 1024.0, 1.0],
            strided,
        )?;
        let result = batch
            .evaluate_depths_tensor(&owners, &depths)?
            .properties()?;
        let [symmetric, asymmetric, before, after, cap] = result.as_slice() else {
            return Err("Missing physical distinction rows".into());
        };
        for row in [symmetric, asymmetric] {
            assert!(row.geometry.area_m2[0] <= 1.0 && row.geometry.area_m2[1] >= 1.0);
        }
        assert!(symmetric.wetted_perimeter_m[1] < asymmetric.wetted_perimeter_m[0]);
        let before = before
            .hydraulic_radius_m
            .ok_or("Missing wet radius before widening")?;
        let after = after
            .hydraulic_radius_m
            .ok_or("Missing wet radius after widening")?;
        assert!(after[1] < before[0]);
        assert!(cap.wetted_perimeter_m[0] <= 4.0 && cap.wetted_perimeter_m[1] >= 4.0);
        assert!(cap.wetted_perimeter_m[1] < 6.0);
    }
    Ok(())
}

pub(crate) fn run<R>(client: R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: Client<R>,
{
    forward::<R>(&client)?;
    recovery::<R>(&client)?;
    physical_distinctions::<R>(&client)
}

#[test]
fn configured_cpu_encloses_frozen_contact_and_radius_references() -> Result {
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
fn actual_cuda_encloses_frozen_contact_and_radius_references() -> Result {
    use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
    run::<CudaRuntime>(CudaClient::new(CudaDevice::new(0))?)
}
