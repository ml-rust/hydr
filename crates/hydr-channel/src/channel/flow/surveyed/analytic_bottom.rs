mod front;

use super::equation_ops::{choose, scalar, scale};
use super::error::{Arithmetic, HydraulicResult};
use super::types::SideState;
use crate::channel::section::GatheredBands;
use numr::dtype::DType;
use numr::ops::{BinaryOps, CompareOps, NextRepresentableOps, ScalarOps, TensorOps, TwoSumOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;
use solvr::common::arithmetic::Interval;

pub(crate) use front::{dry_front, vacuum};

pub(crate) struct AnalyticBottom<R: Runtime<DType = DType>> {
    pub(crate) eligible: Tensor<R>,
    pub(crate) potential_squared: Interval<R>,
}

pub(crate) fn kinds<R>(
    a: &mut Arithmetic<'_, R>,
    bottom: &Tensor<R>,
    pointed: &Tensor<R>,
    width: &Interval<R>,
    slope: &Interval<R>,
) -> HydraulicResult<(Tensor<R>, Tensor<R>)>
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
    let zero = scalar(a, 0.0)?;
    let rectangular = c.mul(
        bottom,
        &c.mul(
            &c.gt(width.lower(), zero.upper())?,
            &c.mul(
                &c.eq(slope.lower(), zero.lower())?,
                &c.eq(slope.upper(), zero.upper())?,
            )?,
        )?,
    )?;
    let pointed = c.mul(
        bottom,
        &c.mul(
            pointed,
            &c.mul(
                &c.gt(slope.lower(), zero.upper())?,
                &c.mul(
                    &c.eq(width.lower(), zero.lower())?,
                    &c.eq(width.upper(), zero.upper())?,
                )?,
            )?,
        )?,
    )?;
    Ok((rectangular, pointed))
}

pub(crate) fn potential_squared<R>(
    a: &mut Arithmetic<'_, R>,
    bands: &GatheredBands<R>,
    side: &SideState<R>,
    gravity: f64,
) -> HydraulicResult<AnalyticBottom<R>>
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
    let column = |value: &Tensor<R>| -> HydraulicResult<Tensor<R>> {
        Ok(value.narrow(1, 0, 1)?.squeeze(Some(1)))
    };
    let width = a.bounds(
        &column(bands.start_width.lower())?,
        &column(bands.start_width.upper())?,
    )?;
    let slope = a.bounds(&column(bands.slope.lower())?, &column(bands.slope.upper())?)?;
    let (rectangular, pointed) = kinds(
        a,
        &column(&bands.valid)?,
        &column(&bands.pointed)?,
        &width,
        &slope,
    )?;
    let supported = c.le(&side.original_area, &column(bands.end_area.lower())?)?;
    let rectangular = c.mul(&rectangular, &supported)?;
    let pointed = c.mul(&pointed, &supported)?;
    let eligible = c.maximum(&rectangular, &pointed)?;
    let zero = scalar(a, 0.0)?;
    let one = scalar(a, 1.0)?;
    // The original conserved area determines rectangular potential without recovering a square root.
    let area = a.point(&side.original_area)?;
    let area = choose(a, &rectangular, &area, &zero)?;
    let denominator = choose(a, &rectangular, &width, &one)?;
    let numerator = scale(a, &area, 4.0)?;
    let numerator = scale(a, &numerator, gravity)?;
    let rectangular_squared = a.div_positive_interval(&numerator, &denominator)?;
    let depth = choose(a, &pointed, &side.geometry.depth, &zero)?;
    let pointed_squared = scale(a, &depth, 8.0)?;
    let pointed_squared = scale(a, &pointed_squared, gravity)?;
    let potential_squared = a.add(&rectangular_squared, &pointed_squared)?;
    Ok(AnalyticBottom {
        eligible,
        potential_squared,
    })
}

#[cfg(test)]
mod tests {
    use super::super::PrismaticRiemannConfig;
    use super::super::curves::prepare_side;
    use super::super::error::{PrismaticRiemannError, checked, into_public};
    use super::super::types::WaveSide;
    use super::super::work::WorkLedger;
    use super::*;
    use crate::channel::section::{
        Constitutive, CrossSectionBatch, CrossSectionDefinition, CrossSectionPoint,
        HydraulicGeometry,
    };
    use numr::ops::{HypotOps, MidpointOps};
    use solvr::integrate::CertifiedSimpsonAlgorithms;

    fn constant<R>(
        a: &mut Arithmetic<'_, R>,
        count: usize,
        value: f64,
        columns: bool,
    ) -> HydraulicResult<Interval<R>>
    where
        R: Runtime<DType = DType>,
        R::Client: TensorOps<R>
            + HypotOps<R>
            + MidpointOps<R>
            + CompareOps<R>
            + NextRepresentableOps<R>
            + ScalarOps<R>
            + TwoSumOps<R>
            + RuntimeClient<R>,
    {
        let shape = if columns { vec![count, 1] } else { vec![count] };
        Ok(a.point(&Tensor::from_slice(
            &vec![value; count],
            &shape,
            a.client().device(),
        )?)?)
    }

    fn rectangle<R>(a: &mut Arithmetic<'_, R>, count: usize) -> HydraulicResult<GatheredBands<R>>
    where
        R: Runtime<DType = DType>,
        R::Client: TensorOps<R>
            + HypotOps<R>
            + MidpointOps<R>
            + CompareOps<R>
            + NextRepresentableOps<R>
            + ScalarOps<R>
            + TwoSumOps<R>
            + RuntimeClient<R>,
    {
        let c = a.client();
        let (geometry, _) =
            native_geometry::<R>(c, &[(0.0, 4.0), (0.0, 0.0), (2.0, 0.0), (2.0, 4.0)])?;
        let owners = Tensor::from_slice(&vec![0_i64; count], &[count], c.device())?;
        let mut bands = geometry.gather(a, &owners)?;
        // Keep exact synthetic intervals while retaining the compiled model's provenance.
        bands.section_owner = Tensor::from_slice(&vec![0_i64; count], &[count], c.device())?;
        bands.band_count = 1;
        bands.valid = Tensor::from_slice(&vec![1.0; count], &[count, 1], c.device())?;
        bands.pointed = Tensor::zeros(&[count, 1], DType::F64, c.device())?;
        bands.start_depth = constant(a, count, 0.0, true)?;
        bands.end_depth = constant(a, count, 4.0, true)?;
        bands.thickness = constant(a, count, 4.0, true)?;
        bands.start_area = constant(a, count, 0.0, true)?;
        bands.end_area = constant(a, count, 8.0, true)?;
        bands.area_increment = constant(a, count, 8.0, true)?;
        bands.start_width = constant(a, count, 2.0, true)?;
        bands.slope = constant(a, count, 0.0, true)?;
        bands.start_integral = constant(a, count, 0.0, true)?;
        bands.cap_depth = constant(a, count, 4.0, false)?;
        bands.cap_area = constant(a, count, 8.0, false)?;
        bands.cap_width = constant(a, count, 2.0, false)?;
        bands.cap_integral = constant(a, count, 16.0, false)?;
        Ok(bands)
    }

    fn side<R>(a: &mut Arithmetic<'_, R>, velocities: &[f64]) -> HydraulicResult<SideState<R>>
    where
        R: Runtime<DType = DType>,
        R::Client: TensorOps<R>
            + HypotOps<R>
            + MidpointOps<R>
            + CompareOps<R>
            + NextRepresentableOps<R>
            + ScalarOps<R>
            + TwoSumOps<R>
            + RuntimeClient<R>,
    {
        let c = a.client();
        let count = velocities.len();
        let area = Tensor::from_slice(&vec![2.0; count], &[count], c.device())?;
        let velocity = Tensor::from_slice(velocities, &[count], c.device())?;
        Ok(SideState {
            original_discharge: c.mul(&area, &velocity)?,
            original_area: area,
            geometry: Constitutive {
                depth: constant(a, count, 1.0, false)?,
                forward_area: constant(a, count, 2.0, false)?,
                area: constant(a, count, 2.0, false)?,
                top_width: constant(a, count, 2.0, false)?,
                integral: constant(a, count, 1.0, false)?,
                hydraulic_depth: constant(a, count, 1.0, false)?,
                wet_possible: Tensor::from_slice(&vec![1.0; count], &[count], c.device())?,
                dry_possible: Tensor::zeros(&[count], DType::F64, c.device())?,
            },
            velocity: a.point(&velocity)?,
            sound: constant(a, count, 1.0, false)?,
            pressure: constant(a, count, 1.0, false)?,
            potential: constant(a, count, 2.0, false)?,
            wet: Tensor::from_slice(&vec![1.0; count], &[count], c.device())?,
            dry: Tensor::zeros(&[count], DType::F64, c.device())?,
        })
    }

    fn thresholds<R>(client: R::Client) -> HydraulicResult<()>
    where
        R: Runtime<DType = DType>,
        R::Client: TensorOps<R>
            + HypotOps<R>
            + MidpointOps<R>
            + CompareOps<R>
            + NextRepresentableOps<R>
            + ScalarOps<R>
            + TwoSumOps<R>
            + RuntimeClient<R>
            + CertifiedSimpsonAlgorithms<R>,
    {
        let (vacuum_mask, left_front, right_front) =
            checked::<R, _>(&client, "analytic bottom threshold certificates", |a| {
                let bands = rectangle(a, 3)?;
                let left = side(a, &[-2.0, (-2.0_f64).next_down(), (-2.0_f64).next_up()])?;
                let right = side(a, &[2.0, 2.0_f64.next_up(), 2.0_f64.next_down()])?;
                let left_proof = potential_squared(a, &bands, &left, 1.0)?;
                let right_proof = potential_squared(a, &bands, &right, 1.0)?;
                let vacuum_mask = vacuum(a, &left, &right, &left_proof, &right_proof)?;
                let left = side(a, &[-2.0, (-2.0_f64).next_down(), (-2.0_f64).next_up()])?;
                let left_proof = potential_squared(a, &bands, &left, 1.0)?;
                let left_front = dry_front(a, &left, WaveSide::Left, &left_proof)?;
                let right_front = dry_front(a, &right, WaveSide::Right, &right_proof)?;
                Ok((vacuum_mask, left_front, right_front))
            })?;
        assert_eq!(vacuum_mask.try_to_vec::<f64>()?, [1.0, 1.0, 0.0]);
        assert_eq!(left_front.try_to_vec::<f64>()?, [1.0, 1.0, 0.0]);
        assert_eq!(right_front.try_to_vec::<f64>()?, [1.0, 1.0, 0.0]);
        eligibility::<R>(&client)?;
        Ok(())
    }

    fn native_geometry<R>(
        client: &R::Client,
        points: &[(f64, f64)],
    ) -> HydraulicResult<(HydraulicGeometry<R>, GatheredBands<R>)>
    where
        R: Runtime<DType = DType>,
        R::Client: TensorOps<R>
            + HypotOps<R>
            + MidpointOps<R>
            + CompareOps<R>
            + NextRepresentableOps<R>
            + ScalarOps<R>
            + TwoSumOps<R>
            + RuntimeClient<R>,
    {
        let definition = CrossSectionDefinition {
            id: "analytic-bottom-eligibility".into(),
            vertical_datum: "Private analytical datum".into(),
            points: points
                .iter()
                .map(|&(station_m, elevation_m)| CrossSectionPoint {
                    station_m,
                    elevation_m,
                })
                .collect(),
            maximum_stage_m: points[0].1,
            profile_provenance: "Original-coordinate analytic certificate regression".into(),
        };
        let batch =
            CrossSectionBatch::<R>::new(client.clone(), vec![definition])?.certify_prismatic()?;
        let geometry = HydraulicGeometry::from_batch(&batch);
        let owners = Tensor::from_slice(&[0_i64], &[1], client.device())?;
        let bands = checked::<R, _>(client, "analytic eligibility original bands", |a| {
            geometry.gather(a, &owners)
        })?;
        Ok((geometry, bands))
    }

    fn eligibility<R>(client: &R::Client) -> HydraulicResult<()>
    where
        R: Runtime<DType = DType>,
        R::Client: TensorOps<R>
            + HypotOps<R>
            + MidpointOps<R>
            + CompareOps<R>
            + NextRepresentableOps<R>
            + ScalarOps<R>
            + TwoSumOps<R>
            + RuntimeClient<R>
            + CertifiedSimpsonAlgorithms<R>,
    {
        let config = PrismaticRiemannConfig {
            gravity_m_per_s2: 1.0,
            ..Default::default()
        };
        let work = WorkLedger::new(&config);
        let area = Tensor::from_slice(&[1.0], &[1], client.device())?;

        let (geometry, bands) =
            native_geometry::<R>(client, &[(-4.0, 4.0), (0.0, 0.0), (4.0, 4.0)])?;
        let left = prepare_side(
            &geometry,
            &bands,
            &area,
            &Tensor::from_slice(&[-4.0], &[1], client.device())?,
            &config,
            config.initial_simpson_panels,
            &work,
        )?;
        let right = prepare_side(
            &geometry,
            &bands,
            &area,
            &Tensor::from_slice(&[4.0], &[1], client.device())?,
            &config,
            config.initial_simpson_panels,
            &work,
        )?;
        let (proof, vacuum_mask, left_front, right_front) =
            checked::<R, _>(client, "pointed bottom eligibility", |a| {
                let proof = potential_squared(a, &bands, &left, 1.0)?;
                let right_proof = potential_squared(a, &bands, &right, 1.0)?;
                let vacuum_mask = vacuum(a, &left, &right, &proof, &right_proof)?;
                let left_front = dry_front(a, &left, WaveSide::Left, &proof)?;
                let right_front = dry_front(a, &right, WaveSide::Right, &right_proof)?;
                Ok((proof, vacuum_mask, left_front, right_front))
            })?;
        assert_eq!(proof.eligible.try_to_vec::<f64>()?, [1.0]);
        assert!(
            proof.potential_squared.lower().item::<f64>()? <= 8.0
                && proof.potential_squared.upper().item::<f64>()? >= 8.0
        );
        assert_eq!(vacuum_mask.try_to_vec::<f64>()?, [1.0]);
        assert_eq!(left_front.try_to_vec::<f64>()?, [1.0]);
        assert_eq!(right_front.try_to_vec::<f64>()?, [1.0]);

        let (geometry, bands) =
            native_geometry::<R>(client, &[(0.1, 4.0), (0.1, 0.0), (1.0, 0.0), (1.0, 4.0)])?;
        assert!(
            bands.start_width.lower().item::<f64>()? < bands.start_width.upper().item::<f64>()?
        );
        let left = prepare_side(
            &geometry,
            &bands,
            &area,
            &Tensor::from_slice(&[-2.0], &[1], client.device())?,
            &config,
            config.initial_simpson_panels,
            &work,
        )?;
        let right = prepare_side(
            &geometry,
            &bands,
            &area,
            &Tensor::from_slice(&[2.0], &[1], client.device())?,
            &config,
            config.initial_simpson_panels,
            &work,
        )?;
        let (proof, vacuum_mask, left_front, right_front) =
            checked::<R, _>(client, "uncertain rectangular width eligibility", |a| {
                let proof = potential_squared(a, &bands, &left, 1.0)?;
                let right_proof = potential_squared(a, &bands, &right, 1.0)?;
                let vacuum_mask = vacuum(a, &left, &right, &proof, &right_proof)?;
                let left_front = dry_front(a, &left, WaveSide::Left, &proof)?;
                let right_front = dry_front(a, &right, WaveSide::Right, &right_proof)?;
                Ok((proof, vacuum_mask, left_front, right_front))
            })?;
        assert_eq!(proof.eligible.try_to_vec::<f64>()?, [1.0]);
        // Original width is less than one, so Phi²=4*A/b exceeds four at A=1.
        assert!(proof.potential_squared.lower().item::<f64>()? > 4.0);
        assert_eq!(vacuum_mask.try_to_vec::<f64>()?, [0.0]);
        assert_eq!(left_front.try_to_vec::<f64>()?, [0.0]);
        assert_eq!(right_front.try_to_vec::<f64>()?, [0.0]);

        let (geometry, bands) = native_geometry::<R>(
            client,
            &[
                (-1.5, 2.0),
                (-1.0, 1.0),
                (-1.0, 0.0),
                (1.0, 0.0),
                (1.0, 1.0),
                (1.5, 2.0),
            ],
        )?;
        assert!(bands.band_count > 1);
        let above_area = Tensor::from_slice(&[3.0], &[1], client.device())?;
        let state = prepare_side(
            &geometry,
            &bands,
            &above_area,
            &Tensor::from_slice(&[0.0], &[1], client.device())?,
            &config,
            config.initial_simpson_panels,
            &work,
        )?;
        let proof = checked::<R, _>(client, "state above analytic bottom support", |a| {
            potential_squared(a, &bands, &state, 1.0)
        })?;
        assert!(
            state.original_area.item::<f64>()?
                > bands.end_area.lower().narrow(1, 0, 1)?.item::<f64>()?
        );
        assert_eq!(proof.eligible.try_to_vec::<f64>()?, [0.0]);
        Ok(())
    }

    #[test]
    fn analytic_bottom_thresholds_include_equality_and_distinguish_neighbors_cpu()
    -> Result<(), PrismaticRiemannError> {
        use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};
        let client = CpuClient::new(CpuDevice::new())
            .with_parallelism(ParallelismConfig::new(Some(1), None));
        thresholds::<CpuRuntime>(client).map_err(|error| into_public(error, Default::default()))
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn analytic_bottom_thresholds_include_equality_and_distinguish_neighbors_cuda()
    -> Result<(), Box<dyn std::error::Error>> {
        use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
        thresholds::<CudaRuntime>(CudaClient::new(CudaDevice::new(0))?)
            .map_err(|error| into_public(error, Default::default()))?;
        Ok(())
    }
}
