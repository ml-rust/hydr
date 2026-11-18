use crate::channel::{
    flow::surveyed::{
        analytic_bottom::kinds,
        band_lanes::IntegrationLanes,
        equation_ops::{choose, scalar},
        error::{Arithmetic, HydraulicResult},
    },
    section::PreparedAreaPrefix,
};
use numr::{
    dtype::DType,
    ops::{
        BinaryOps, CompareOps, NextRepresentableOps, ScalarOps, TensorOps, TwoSumOps,
        TypeConversionOps, UtilityOps,
    },
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};

pub(super) struct PointPotentialLanes<R: Runtime<DType = DType>> {
    pub(super) lanes: IntegrationLanes<R>,
    pub(super) pointed: Tensor<R>,
    #[cfg_attr(not(test), expect(dead_code, reason = "tests read it"))]
    pub(super) original_rows: Tensor<R>,
    #[cfg_attr(not(test), expect(dead_code, reason = "tests read it"))]
    pub(super) section_owner: Tensor<R>,
}

pub(super) fn build<R>(
    a: &mut Arithmetic<'_, R>,
    prefix: &PreparedAreaPrefix<R>,
    gravity: f64,
) -> HydraulicResult<PointPotentialLanes<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    let c = a.client();
    let physical = prefix.integration_geometry(a)?;
    let m = physical.extent.lower().numel();
    let shape = [physical.face_count, physical.band_count];
    let face = c
        .arange(0.0, physical.face_count as f64, 1.0, DType::I64)?
        .reshape(&[physical.face_count, 1])?;
    let face_owner = face.broadcast_to(&shape)?.contiguous()?.reshape(&[m])?;
    let bottom = c.cast(
        &c.eq(
            &physical.band_index,
            &Tensor::zeros(&[], DType::I64, c.device())?,
        )?,
        DType::F64,
    )?;
    let zero = scalar(a, 0.0)?;
    let one = scalar(a, 1.0)?;
    let (rectangular, pointed) = kinds(
        a,
        &bottom,
        &physical.pointed,
        &physical.start_width,
        &physical.slope,
    )?;
    let zero_extent = c.eq(physical.extent.upper(), zero.upper())?;
    let analytic = c.maximum(&zero_extent, &c.maximum(&pointed, &rectangular)?)?;
    let active = c.mul(&physical.valid, &c.eq(&analytic, zero.lower())?)?;
    let safe_width = choose(a, &active, &physical.start_width, &one)?;
    let safe_area = choose(a, &active, &physical.start_area, &one)?;
    let safe_slope = choose(a, &active, &physical.slope, &zero)?;
    Ok(PointPotentialLanes {
        lanes: IntegrationLanes {
            face_owner,
            band_index: physical.band_index,
            bottom,
            active,
            analytic,
            extent: physical.extent,
            width: safe_width,
            slope: safe_slope,
            area: safe_area,
            gravity: scalar(a, gravity)?,
            face_count: physical.face_count,
            band_count: physical.band_count,
        },
        pointed: physical.pointed,
        original_rows: physical.original_rows,
        section_owner: physical.section_owner,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::channel::flow::surveyed_points::{
        source_fixture_support as support, source_potential_fixtures as references,
    };
    use crate::channel::{
        flow::surveyed::{
            PrismaticRiemannConfig, error, integration::integrate_lanes, work::WorkLedger,
        },
        section::{
            AreaAnchor, AreaDirection, AreaQuery, ConditionalBandSegments, HydraulicGeometry,
            PrismaticRecoveryConfig,
        },
    };
    use solvr::integrate::CertifiedSimpsonAlgorithms;
    use support::{Client, Result, hydraulic};

    fn prefixes<R>(client: &R::Client) -> Result
    where
        R: Runtime<DType = DType>,
        R::Client: Client<R> + CertifiedSimpsonAlgorithms<R>,
    {
        let section = support::batch::<R>(client)?;
        let geometry = HydraulicGeometry::from_batch(&section);
        let selected = [
            (0_usize, 0_usize),
            (28, 0),
            (28, 0),
            (28, 1),
            (29, 0),
            (29, 1),
            (30, 0),
            (28, 0),
        ];
        let cases = support::cases();
        let owner_values: Vec<_> = selected
            .iter()
            .map(|&(case, _)| cases[case].owner as i64)
            .collect();
        let areas = support::expansion::<R>(
            client,
            &selected
                .iter()
                .map(|&(case, side)| cases[case].inputs[2 * side])
                .collect::<Vec<_>>(),
            true,
        )?;
        let owners = Tensor::from_slice(&owner_values, &[selected.len()], client.device())?;
        let source = section.recover_area_points_tensor(
            &owners,
            &areas,
            &PrismaticRecoveryConfig::default(),
        )?;
        let reorder = [7_i64, 0, 7, 1, 2, 3, 4, 5, 6];
        let physical = hydraulic(error::checked::<R, _>(
            client,
            "retained prefix potential lanes",
            |a| {
                let bands = geometry.gather(a, &owners)?;
                let zero = Tensor::zeros(&[selected.len()], DType::F64, client.device())?;
                let scale = a.expansion_point(&zero)?;
                let parameter = a.point(&zero)?;
                let endpoint = AreaQuery::source_affine(
                    a,
                    &bands,
                    &source,
                    AreaDirection::Add,
                    &scale,
                    &parameter,
                )?
                .prove_support(a, &bands)?;
                let rows = Tensor::from_slice(&reorder, &[reorder.len()], client.device())?;
                let endpoint = endpoint.select(a, &rows)?;
                let bands = bands.select(a, &rows)?;
                let boundaries = geometry
                    .boundary_areas(a)?
                    .gather(a, &owners)?
                    .select(a, &rows)?;
                let prefix = ConditionalBandSegments::prefix_from_exact_dry(
                    a,
                    &bands,
                    &endpoint,
                    &boundaries,
                )?;
                let physical = build(a, &prefix, 1.0)?;
                let b = section.maximum_band_count();
                assert_eq!(physical.lanes.face_count, reorder.len());
                assert_eq!(physical.lanes.band_count, b);
                let dense: Vec<_> = (0..reorder.len())
                    .flat_map(|row| std::iter::repeat_n(row as i64, b))
                    .collect();
                let originals: Vec<_> = reorder
                    .into_iter()
                    .flat_map(|row| std::iter::repeat_n(row, b))
                    .collect();
                let expected_owners: Vec<_> = reorder
                    .into_iter()
                    .flat_map(|row| std::iter::repeat_n(owner_values[row as usize], b))
                    .collect();
                assert_eq!(physical.lanes.face_owner.try_to_vec::<i64>()?, dense);
                assert_eq!(physical.original_rows.try_to_vec::<i64>()?, originals);
                assert_eq!(physical.section_owner.try_to_vec::<i64>()?, expected_owners);
                let inactive = client
                    .eq(
                        &physical.lanes.active,
                        &Tensor::zeros(physical.lanes.active.shape(), DType::F64, client.device())?,
                    )?
                    .try_to_vec::<f64>()?;
                let widths = physical.lanes.width.lower().try_to_vec::<f64>()?;
                let areas = physical.lanes.area.lower().try_to_vec::<f64>()?;
                let slopes = physical.lanes.slope.upper().try_to_vec::<f64>()?;
                for (lane, &inactive) in inactive.iter().enumerate() {
                    if inactive != 0.0 {
                        assert_eq!([widths[lane], areas[lane], slopes[lane]], [1.0, 1.0, 0.0]);
                    }
                }
                Ok(physical)
            },
        ))?;
        let config = PrismaticRiemannConfig {
            gravity_m_per_s2: 1.0,
            ..Default::default()
        };
        let work = WorkLedger::new(&config);
        let potential = hydraulic(integrate_lanes(
            client,
            &physical.lanes,
            &physical.pointed,
            &config,
            512,
            &work,
        ))?;
        let observed = support::endpoints(&potential)?;
        for (row, &original) in reorder.iter().enumerate() {
            let (case, side) = selected[original as usize];
            let reference = references::CASES
                .iter()
                .find(|reference| reference.source_index == case)
                .ok_or("Missing independent source potential")?;
            assert_eq!(reference.gravity_bits, 1.0_f64.to_bits());
            let reference = reference.sides[side].phi;
            let expected = reference.bounds.map(f64::from_bits);
            let actual = observed[row];
            assert!(
                actual[0] <= expected[0] && expected[1] <= actual[1],
                "case={case}, side={side}: {actual:?} excludes {expected:?}"
            );
            let excess = (actual[0] - expected[1])
                .abs()
                .max((actual[1] - expected[0]).abs());
            assert!(
                excess <= f64::from_bits(reference.native_tolerance),
                "case={case}, side={side}: excess={excess}"
            );
        }
        assert!(work.snapshot().simpson_value_points > 0);
        endpoint_controls::<R>(client)
    }

    fn endpoint_controls<R>(client: &R::Client) -> Result
    where
        R: Runtime<DType = DType>,
        R::Client: Client<R>,
    {
        let section = support::batch::<R>(client)?;
        let geometry = HydraulicGeometry::from_batch(&section);
        hydraulic(error::checked::<R, _>(
            client,
            "potential prefix empty and dry geometry",
            |a| {
                let owners = Tensor::from_slice(&[2_i64, 1, 4], &[3], client.device())?;
                let bands = geometry.gather(a, &owners)?;
                let table = geometry.boundary_areas(a)?;
                let boundaries = table.gather(a, &owners)?;
                let distance = f64::from_bits(0x35f0000000000000);
                let scale = a.expansion_point(&Tensor::from_slice(
                    &[distance, 0.0, distance],
                    &[3],
                    client.device(),
                )?)?;
                let parameter = a.point(&Tensor::ones(&[3], DType::F64, client.device())?)?;
                let anchor = Tensor::from_slice(&[1_i64, 0, 0], &[3], client.device())?;
                let endpoint = AreaQuery::band_affine(
                    a,
                    &bands,
                    AreaAnchor::BandStart(anchor),
                    AreaDirection::Add,
                    &scale,
                    &parameter,
                )?
                .prove_support(a, &bands)?;
                let prefix = ConditionalBandSegments::prefix_from_exact_dry(
                    a,
                    &bands,
                    &endpoint,
                    &boundaries,
                )?;
                let physical = build(a, &prefix, 1.0)?;
                let extents = physical.lanes.extent.upper().try_to_vec::<f64>()?;
                assert!(extents[1] > 0.0 && extents[1] <= distance);
                assert_eq!(&extents[3..6], &[0.0; 3]);
                assert!(extents[6] > 0.0);
                assert_eq!(physical.pointed.try_to_vec::<f64>()?[6], 1.0);
                assert_eq!(physical.lanes.active.try_to_vec::<f64>()?[6], 0.0);
                let subnormal_scale = a.expansion_point(&Tensor::from_slice(
                    &[f64::from_bits(1); 3],
                    &[3],
                    client.device(),
                )?)?;
                for (direction, signed_half) in
                    [(AreaDirection::Add, 0.5), (AreaDirection::Subtract, -0.5)]
                {
                    let parameter = a.point(&Tensor::from_slice(
                        &[signed_half; 3],
                        &[3],
                        client.device(),
                    )?)?;
                    let dry_containing = AreaQuery::band_affine(
                        a,
                        &bands,
                        AreaAnchor::Zero,
                        direction,
                        &subnormal_scale,
                        &parameter,
                    )?
                    .prove_support(a, &bands)
                    .inspect_err(|error| {
                        eprintln!("positive underflow map, parameter={signed_half}: {error:?}")
                    })?;
                    assert_eq!(dry_containing.area.lower().try_to_vec::<f64>()?, [0.0; 3]);
                    let prefix = ConditionalBandSegments::prefix_from_exact_dry(
                        a,
                        &bands,
                        &dry_containing,
                        &boundaries,
                    )?;
                    assert_eq!(prefix.endpoint_area.high().try_to_vec::<f64>()?, [0.0; 3]);
                    assert_eq!(prefix.endpoint_area.low().try_to_vec::<f64>()?, [0.0; 3]);
                    assert!(
                        prefix
                            .endpoint_area
                            .abs_error()
                            .try_to_vec::<f64>()?
                            .iter()
                            .all(|&radius| radius > 0.0)
                    );
                    assert_eq!(prefix.segments.zero_gap_mask().try_to_vec::<f64>()?[0], 0.0);
                    assert_eq!(prefix.segments.active_mask().try_to_vec::<f64>()?[0], 1.0);
                    let physical = build(a, &prefix, 1.0)?;
                    assert_eq!(physical.lanes.extent.lower().try_to_vec::<f64>()?[0], 0.0);
                    assert!(physical.lanes.extent.upper().try_to_vec::<f64>()?[0] > 0.0);
                }
                for (direction, signed_half) in
                    [(AreaDirection::Add, -0.5), (AreaDirection::Subtract, 0.5)]
                {
                    let parameter = a.point(&Tensor::from_slice(
                        &[signed_half; 3],
                        &[3],
                        client.device(),
                    )?)?;
                    let negative = AreaQuery::band_affine(
                        a,
                        &bands,
                        AreaAnchor::Zero,
                        direction,
                        &subnormal_scale,
                        &parameter,
                    )?
                    .prove_support(a, &bands);
                    assert!(
                        matches!(negative, Err(error::HydraulicError::Geometry(crate::channel::section::CrossSectionError::InvalidInput { ref context, .. })) if context.contains("not proved nonnegative"))
                    );
                }
                let empty = Tensor::from_slice::<i64>(&[], &[0], client.device())?;
                let empty_bands = geometry.gather(a, &empty)?;
                let empty_boundaries = table.gather(a, &empty)?;
                let empty_values = Tensor::zeros(&[0], DType::F64, client.device())?;
                let empty_scale = a.expansion_point(&empty_values)?;
                let empty_parameter = a.point(&empty_values)?;
                let empty_query = AreaQuery::band_affine(
                    a,
                    &empty_bands,
                    AreaAnchor::Zero,
                    AreaDirection::Add,
                    &empty_scale,
                    &empty_parameter,
                )?
                .prove_support(a, &empty_bands)?;
                let prefix = ConditionalBandSegments::prefix_from_exact_dry(
                    a,
                    &empty_bands,
                    &empty_query,
                    &empty_boundaries,
                )?;
                let physical = build(a, &prefix, 1.0)?;
                assert_eq!(physical.lanes.face_count, 0);
                assert_eq!(physical.lanes.band_count, section.maximum_band_count());
                assert_eq!(physical.lanes.extent.lower().shape(), &[0]);
                Ok(())
            },
        ))
    }

    #[test]
    fn cpu_point_potential_lanes_retain_prefix_coordinates_and_dense_rows() -> Result {
        use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};
        prefixes::<CpuRuntime>(
            &CpuClient::new(CpuDevice::new())
                .with_parallelism(ParallelismConfig::new(Some(1), None)),
        )
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_point_potential_lanes_retain_prefix_coordinates_and_dense_rows() -> Result {
        use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
        prefixes::<CudaRuntime>(&CudaClient::new(CudaDevice::new(0))?)
    }
}
