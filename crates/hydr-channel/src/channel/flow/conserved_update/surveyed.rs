//! Surveyed face transfers and donor bounds on two-part conserved state.

use super::super::{ChannelError, conserved_point::ConservedPoint};
use numr::dtype::DType;
use numr::ops::{CompareOps, ConditionalOps, UnaryOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;
use solvr::common::arithmetic::ArithmeticResult;

use super::expansion::{
    ExpansionClient, Parts, bounded, center, exact, exact_over, expanded, parts, transfer,
};

/// Shared face points and the cell transfers formed from them.
///
/// Face `bound` holds the product radius. Transfer `bound` holds the subtraction radius.
pub(in crate::channel::flow) struct SurveyedFaces<R: Runtime<DType = DType>> {
    pub volume: Parts<R>,
    pub momentum: Parts<R>,
    pub water_delta: Parts<R>,
    pub momentum_delta: Parts<R>,
}

/// Integrate two-part face rates over `dt` and form cell transfers from the selected face points.
///
/// Each face `bound` holds the rate radius carried through the product.
///
/// `direct` marks the left and right end faces whose rates come from the endpoint policy.
/// Such a face point keeps one F64 part, `RN(rate · dt)`, with a `+0.0` low part. Its exact
/// product tail joins the face `bound`. The end cell then receives exactly the F64 volume
/// that a coordinator or a pool shares.
pub(in crate::channel::flow) fn surveyed_faces<R>(
    client: &R::Client,
    water_rate: &Parts<R>,
    momentum_rate: &Parts<R>,
    dt: f64,
    count: usize,
    direct: [bool; 2],
) -> Result<SurveyedFaces<R>, ChannelError>
where
    R: Runtime<DType = DType>,
    R::Client: ExpansionClient<R>,
{
    let faces = water_rate.high.numel();
    let single = if direct.contains(&true) {
        let mut mask = vec![0.0; faces];
        if direct[0] {
            mask[0] = 1.0;
        }
        if direct[1] {
            mask[faces - 1] = 1.0;
        }
        let mask = Tensor::from_slice(&mask, &[faces], client.device())?;
        let zero = Tensor::zeros(&[faces], DType::F64, client.device())?;
        Some((client.ne(&mask, &zero)?, zero))
    } else {
        None
    };
    expanded(client, "surveyed shared face impulses", |a| {
        let duration =
            Tensor::full_scalar(water_rate.high.shape(), DType::F64, dt, client.device())?;
        let duration = a.expansion_point(&duration)?;
        let mut product = |rate: &Parts<R>| -> ArithmeticResult<_> {
            let rate = bounded(a, rate)?;
            let raw = a.expansion_mul(&rate, &duration)?;
            let face = match &single {
                None => parts(&raw),
                Some((mask, zero)) => {
                    let c = a.client();
                    // Radius plus product tail, rounded upward.
                    let radius = a.point(raw.abs_error())?;
                    let tail = a.point(&c.abs(raw.low())?)?;
                    let charged = a.add(&radius, &tail)?;
                    Parts {
                        high: raw.high().clone(),
                        low: c.where_cond(mask, zero, raw.low())?,
                        bound: c.where_cond(mask, charged.upper(), raw.abs_error())?,
                    }
                }
            };
            // The center becomes the shared face point. Its radius is charged separately.
            let point = exact(a, &face.high, &face.low)?;
            let transfer = transfer(a, &point, count)?;
            Ok((face, parts(&transfer)))
        };
        let (volume, water_delta) = product(water_rate)?;
        let (momentum, momentum_delta) = product(momentum_rate)?;
        Ok(SurveyedFaces {
            volume,
            momentum,
            water_delta,
            momentum_delta,
        })
    })
}

/// Add two expansions and select the center once.
///
/// The returned bound is the radius of this sum. The selected point has radius zero.
pub(in crate::channel::flow) fn select_sum<R>(
    client: &R::Client,
    first: &Parts<R>,
    second: &Parts<R>,
    stage: &'static str,
) -> Result<(ConservedPoint<R>, Tensor<R>), ChannelError>
where
    R: Runtime<DType = DType>,
    R::Client: ExpansionClient<R>,
{
    expanded(client, stage, |a| {
        let first = bounded(a, first)?;
        let second = bounded(a, second)?;
        let sum = a.expansion_add(&first, &second)?;
        Ok((center(&sum), sum.abs_error().clone()))
    })
}

/// Lower bound of each normalized cell value.
///
/// A negative low part lowers the bound below `high`, so it never adds donor water.
pub(in crate::channel::flow) fn lower_bound<R>(
    client: &R::Client,
    point: &ConservedPoint<R>,
) -> Result<Tensor<R>, ChannelError>
where
    R: Runtime<DType = DType>,
    R::Client: ExpansionClient<R>,
{
    expanded(client, "surveyed donor water lower bound", |a| {
        let value = exact(a, &point.high, &point.low)?;
        Ok(a.expansion_interval(&value)?.lower().clone())
    })
}

/// Upper bound of the water each cell sends through its two faces.
pub(in crate::channel::flow) fn outgoing_upper<R>(
    client: &R::Client,
    face_high: &Tensor<R>,
    face_low: &Tensor<R>,
    count: usize,
) -> Result<Tensor<R>, ChannelError>
where
    R: Runtime<DType = DType>,
    R::Client: ExpansionClient<R>,
{
    let c = client;
    expanded(client, "surveyed outgoing shared face volumes", |a| {
        let zero = Tensor::zeros(&[count], DType::F64, c.device())?;
        let right_high = face_high.narrow(0, 1, count)?;
        let right_low = face_low.narrow(0, 1, count)?;
        let left_high = face_high.narrow(0, 0, count)?;
        let left_low = face_low.narrow(0, 0, count)?;
        // A normalized point has the sign of its high part.
        let right_out = c.gt(&right_high, &zero)?;
        let left_out = c.lt(&left_high, &zero)?;
        let right = exact_over(
            a,
            &c.where_cond(&right_out, &right_high, &zero)?,
            &c.where_cond(&right_out, &right_low, &zero)?,
            &zero,
        )?;
        let left = exact_over(
            a,
            &c.where_cond(&left_out, &c.neg(&left_high)?, &zero)?,
            &c.where_cond(&left_out, &c.neg(&left_low)?, &zero)?,
            &zero,
        )?;
        let sum = a.expansion_add(&right, &left)?;
        Ok(a.expansion_interval(&sum)?.upper().clone())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};

    type TestResult<T = ()> = Result<T, Box<dyn std::error::Error>>;

    fn tensor<R>(client: &R::Client, values: &[f64]) -> TestResult<Tensor<R>>
    where
        R: Runtime<DType = DType>,
        R::Client: ExpansionClient<R>,
    {
        Ok(Tensor::from_slice(
            values,
            &[values.len()],
            client.device(),
        )?)
    }

    fn values<R: Runtime<DType = DType>>(value: &Tensor<R>) -> TestResult<Vec<f64>> {
        Ok(value.try_to_vec::<f64>()?)
    }

    /// Rate parts with the given components on every face.
    fn rate<R>(client: &R::Client, high: &[f64], low: &[f64], bound: &[f64]) -> TestResult<Parts<R>>
    where
        R: Runtime<DType = DType>,
        R::Client: ExpansionClient<R>,
    {
        Ok(Parts {
            high: tensor::<R>(client, high)?,
            low: tensor::<R>(client, low)?,
            bound: tensor::<R>(client, bound)?,
        })
    }

    fn two_sum(a: f64, b: f64) -> (f64, f64) {
        let sum = a + b;
        let virtual_b = sum - a;
        (sum, (a - (sum - virtual_b)) + (b - virtual_b))
    }

    fn sub_ulp_impulse<R>(client: R::Client) -> TestResult
    where
        R: Runtime<DType = DType>,
        R::Client: ExpansionClient<R>,
    {
        let tensor = |values: &[f64]| tensor::<R>(&client, values);
        let impulse = -4791380824461455.0 * 2f64.powi(-104);
        // The outgoing right face carries the impulse away from the cell.
        let zero = [0.0, 0.0];
        let faces = surveyed_faces::<R>(
            &client,
            &rate::<R>(&client, &[0.0, -impulse], &zero, &zero)?,
            &rate::<R>(&client, &zero, &zero, &zero)?,
            1.0,
            1,
            [false, false],
        )?;
        assert_eq!(values(&faces.volume.high)?, vec![0.0, -impulse]);
        assert_eq!(values(&faces.volume.bound)?, vec![0.0, 0.0]);
        assert_eq!(values(&faces.water_delta.high)?, vec![impulse]);
        assert_eq!(35.0 + impulse, 35.0);
        let state = Parts {
            high: tensor(&[35.0])?,
            low: tensor(&[0.0])?,
            bound: tensor(&[0.0])?,
        };
        let (point, bound) = select_sum::<R>(&client, &state, &faces.water_delta, "test")?;
        assert_eq!(values(&point.high)?[0].to_bits(), 35f64.to_bits());
        assert_eq!(values(&point.low)?[0].to_bits(), impulse.to_bits());
        assert_eq!(values(&bound)?, vec![0.0]);
        let lower = lower_bound::<R>(&client, &point)?;
        assert_eq!(values(&lower)?, vec![f64::from_bits(35f64.to_bits() - 1)]);
        let outgoing = outgoing_upper::<R>(&client, &faces.volume.high, &faces.volume.low, 1)?;
        assert_eq!(values(&outgoing)?, vec![-impulse]);
        Ok(())
    }

    fn fused_residual<R>(client: R::Client) -> TestResult
    where
        R: Runtime<DType = DType>,
        R::Client: ExpansionClient<R>,
    {
        let rate = 1.0 + f64::EPSILON;
        let zero = [0.0, 0.0];
        let faces = surveyed_faces::<R>(
            &client,
            &self::rate::<R>(&client, &[rate, 0.0], &zero, &zero)?,
            &self::rate::<R>(&client, &zero, &zero, &zero)?,
            rate,
            1,
            [false, false],
        )?;
        let (high, low) = (
            values(&faces.volume.high)?[0],
            values(&faces.volume.low)?[0],
        );
        assert_eq!(high, rate * rate);
        assert_eq!(low, rate.mul_add(rate, -(rate * rate)));
        assert!(values(&faces.volume.bound)?[0] >= 0.0);
        Ok(())
    }

    fn two_part_rate<R>(client: R::Client) -> TestResult
    where
        R: Runtime<DType = DType>,
        R::Client: ExpansionClient<R>,
    {
        let (high, low, radius, dt) = (3.0, 2f64.powi(-60), 2f64.powi(-70), 0.1);
        let water = rate::<R>(&client, &[high, -high], &[low, -low], &[radius, radius])?;
        let momentum = rate::<R>(&client, &[-high, 0.0], &[-low, 0.0], &[radius, 0.0])?;
        let faces = surveyed_faces::<R>(&client, &water, &momentum, dt, 1, [false, false])?;
        let direct = expanded(&client, "test direct face products", |a| {
            let duration =
                Tensor::full_scalar(water.high.shape(), DType::F64, dt, client.device())?;
            let duration = a.expansion_point(&duration)?;
            let water = bounded(a, &water)?;
            let momentum = bounded(a, &momentum)?;
            Ok((
                parts(&a.expansion_mul(&water, &duration)?),
                parts(&a.expansion_mul(&momentum, &duration)?),
            ))
        })?;
        for (face, direct) in [(&faces.volume, &direct.0), (&faces.momentum, &direct.1)] {
            for (actual, expected) in [
                (&face.high, &direct.high),
                (&face.low, &direct.low),
                (&face.bound, &direct.bound),
            ] {
                let bits = |v: &Tensor<R>| -> TestResult<Vec<u64>> {
                    Ok(values(v)?.iter().map(|x| x.to_bits()).collect())
                };
                assert_eq!(bits(actual)?, bits(expected)?);
            }
        }
        assert_ne!(values(&faces.volume.low)?[0], 0.0);
        assert!(values(&faces.volume.bound)?[0] >= radius * dt);
        assert_eq!(values(&faces.momentum.bound)?[1], 0.0);
        Ok(())
    }

    fn selection_bound<R>(client: R::Client) -> TestResult
    where
        R: Runtime<DType = DType>,
        R::Client: ExpansionClient<R>,
    {
        let tensor = |values: &[f64]| tensor::<R>(&client, values);
        // Four exactly separated terms cannot fit in two parts.
        let (a, b, c, d) = (1.0, 2f64.powi(-60), 2f64.powi(-120), 2f64.powi(-180));
        let first = Parts {
            high: tensor(&[a])?,
            low: tensor(&[c])?,
            bound: tensor(&[0.0])?,
        };
        let second = Parts {
            high: tensor(&[b])?,
            low: tensor(&[d])?,
            bound: tensor(&[0.0])?,
        };
        let (point, bound) = select_sum::<R>(&client, &first, &second, "test")?;
        let (high, low) = (values(&point.high)?[0], values(&point.low)?[0]);
        let (oracle_high, oracle_low) = two_sum(a, b);
        assert_eq!(two_sum(high, low), (high, low));
        assert_eq!((high, low), (oracle_high, oracle_low));
        let bound = values(&bound)?[0];
        assert!(bound >= c + d && bound <= 2.0 * (c + d), "bound={bound:e}");
        // The selected point carries no radius into the next update.
        let next = Parts {
            high: point.high.clone(),
            low: point.low.clone(),
            bound: tensor(&[0.0])?,
        };
        let zero = Parts {
            high: tensor(&[0.0])?,
            low: tensor(&[0.0])?,
            bound: tensor(&[0.0])?,
        };
        let (again, again_bound) = select_sum::<R>(&client, &next, &zero, "test")?;
        assert_eq!(values(&again.high)?, vec![high]);
        assert_eq!(values(&again.low)?, vec![low]);
        assert_eq!(values(&again_bound)?, vec![0.0]);
        Ok(())
    }

    /// Direct end faces keep `RN(rate · dt)` and charge the product tail to their bound.
    fn single_part_ends<R>(client: R::Client) -> TestResult
    where
        R: Runtime<DType = DType>,
        R::Client: ExpansionClient<R>,
    {
        let bits = |v: &Tensor<R>| -> TestResult<Vec<u64>> {
            Ok(values(v)?.iter().map(|x| x.to_bits()).collect())
        };
        let (rate, dt) = (1.0 + f64::EPSILON, 1.0 + 2.0 * f64::EPSILON);
        let (high, low, radius) = (3.0, 2f64.powi(-60), 2f64.powi(-70));
        // Faces 0 and 2 are direct ends with one-part rates. Face 1 is a two-part interior face.
        let water = rate_parts::<R>(&client, [rate, high, -rate], [0.0, low, 0.0], radius)?;
        let momentum = rate_parts::<R>(&client, [-rate, high, rate], [0.0, low, 0.0], radius)?;
        let direct = surveyed_faces::<R>(&client, &water, &momentum, dt, 2, [true, true])?;
        let riemann = surveyed_faces::<R>(&client, &water, &momentum, dt, 2, [false, false])?;
        let product = rate * dt;
        let tail = rate.mul_add(dt, -product);
        assert_ne!(tail, 0.0);
        for (face, riemann, sign) in [
            (&direct.volume, &riemann.volume, 1.0),
            (&direct.momentum, &riemann.momentum, -1.0),
        ] {
            let high = values(&face.high)?;
            let low = values(&face.low)?;
            let bound = values(&face.bound)?;
            for (index, signed) in [(0, sign * product), (2, -sign * product)] {
                assert_eq!(high[index].to_bits(), signed.to_bits());
                assert_eq!(low[index].to_bits(), 0.0_f64.to_bits());
                assert!(bound[index] >= tail.abs() && bound[index] <= 2.0 * tail.abs());
            }
            // The interior face keeps the exact two-part product of the Riemann rule.
            for (actual, expected) in [
                (&face.high, &riemann.high),
                (&face.low, &riemann.low),
                (&face.bound, &riemann.bound),
            ] {
                assert_eq!(bits(actual)?[1], bits(expected)?[1]);
            }
            assert_eq!(values(&riemann.low)?[0], tail * sign);
        }
        // One cell between two direct ends receives exactly both one-part end volumes.
        let ends = self::rate::<R>(&client, &[rate, -rate], &[0.0, 0.0], &[0.0, 0.0])?;
        let cell = surveyed_faces::<R>(&client, &ends, &ends, dt, 1, [true, true])?;
        for delta in [&cell.water_delta, &cell.momentum_delta] {
            assert_eq!(values(&delta.high)?[0].to_bits(), (2.0 * product).to_bits());
            assert_eq!(values(&delta.low)?[0].to_bits(), 0.0_f64.to_bits());
            assert_eq!(values(&delta.bound)?[0], 0.0);
        }
        let riemann = surveyed_faces::<R>(&client, &ends, &ends, dt, 1, [false, false])?;
        assert_eq!(values(&riemann.water_delta.low)?[0], 2.0 * tail);
        Ok(())
    }

    /// Three face rates with one radius on the interior face only.
    fn rate_parts<R>(
        client: &R::Client,
        high: [f64; 3],
        low: [f64; 3],
        radius: f64,
    ) -> TestResult<Parts<R>>
    where
        R: Runtime<DType = DType>,
        R::Client: ExpansionClient<R>,
    {
        rate::<R>(client, &high, &low, &[0.0, radius, 0.0])
    }

    fn cpu() -> CpuClient {
        CpuClient::new(CpuDevice::new())
    }

    #[test]
    fn cpu_direct_end_faces_keep_one_part_and_charge_the_product_tail() -> TestResult {
        single_part_ends::<CpuRuntime>(cpu())
    }

    #[test]
    fn cpu_cap_cell_retains_a_sub_ulp_shared_impulse() -> TestResult {
        sub_ulp_impulse::<CpuRuntime>(cpu())
    }

    #[test]
    fn cpu_face_product_keeps_its_fused_residual() -> TestResult {
        fused_residual::<CpuRuntime>(cpu())
    }

    #[test]
    fn cpu_compress_discards_a_term_into_one_selection_bound() -> TestResult {
        selection_bound::<CpuRuntime>(cpu())
    }

    #[test]
    fn cpu_two_part_rate_lifts_every_component_into_the_face_product() -> TestResult {
        two_part_rate::<CpuRuntime>(cpu())
    }

    #[cfg(feature = "cuda")]
    mod cuda {
        use super::*;
        use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};

        #[test]
        fn cuda_direct_end_faces_keep_one_part_and_charge_the_product_tail() -> TestResult {
            single_part_ends::<CudaRuntime>(CudaClient::new(CudaDevice::new(0))?)
        }

        #[test]
        fn cuda_cap_cell_retains_a_sub_ulp_shared_impulse() -> TestResult {
            sub_ulp_impulse::<CudaRuntime>(CudaClient::new(CudaDevice::new(0))?)
        }

        #[test]
        fn cuda_face_product_keeps_its_fused_residual() -> TestResult {
            fused_residual::<CudaRuntime>(CudaClient::new(CudaDevice::new(0))?)
        }

        #[test]
        fn cuda_compress_discards_a_term_into_one_selection_bound() -> TestResult {
            selection_bound::<CudaRuntime>(CudaClient::new(CudaDevice::new(0))?)
        }

        #[test]
        fn cuda_two_part_rate_lifts_every_component_into_the_face_product() -> TestResult {
            two_part_rate::<CudaRuntime>(CudaClient::new(CudaDevice::new(0))?)
        }
    }
}
