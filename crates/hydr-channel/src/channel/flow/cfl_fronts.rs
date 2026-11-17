//! Dry-front speed bounds at wet/dry faces for the channel CFL rule.
//!
//! A face Riemann problem at a wet/dry face sees only the cell-averaged tip state. Its
//! rarefaction-to-dry speed `u + φ(A)` can fall below the true front speed. The true front
//! carries the Riemann invariant `w = u + φ(A)` of the water behind it, with
//! `φ(A) = ∫_0^A c(a)/a da`. Godunov-type updates with enclosing wave bounds preserve the
//! invariant region `w ≤ W`. Within one step the dry front therefore comes from the contiguous
//! wet run beside the dry cell. The run maximum of `w` bounds the front speed into that cell.
//! The mirror invariant `-(u - φ(A))` bounds a front that moves left.
//!
//! Lateral and friction sources act after transport within a step. They move no water between
//! cells in that step, so the accepted-start bound covers the transport. They can change `w`.
//! A candidate check re-reads these fronts from the post-source state. Without a candidate
//! check, the next step's start bound reads the changed state.

use super::{
    ChannelError,
    cfl_speeds::FaceSpeeds,
    segmented_max::{ScanDirection, segmented_max},
};
use numr::dtype::DType;
use numr::ops::{BinaryOps, CompareOps, ConditionalOps, ReduceOps, ShapeOps, UnaryOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Dry-front speeds of the states on one side of every face, in m/s.
pub(super) struct SideFronts<R: Runtime<DType = DType>> {
    /// Upper bound of `u + φ(A)`, clamped at zero. Zero where the state is dry.
    pub rightward: Tensor<R>,
    /// Upper bound of `-(u - φ(A))`, clamped at zero. Zero where the state is dry.
    pub leftward: Tensor<R>,
    /// `1` where the state holds water, `0` where `A = 0`.
    pub wet: Tensor<R>,
}

/// Dry-front speeds of the left and right states at the `count + 1` faces.
pub(super) struct DryFronts<R: Runtime<DType = DType>> {
    pub left: SideFronts<R>,
    pub right: SideFronts<R>,
}

/// Nonnegative directional dry-front speeds from signed bounds `[u - φ, u + φ]`.
pub(super) fn side_fronts<R, C>(
    client: &C,
    lower: &Tensor<R>,
    upper: &Tensor<R>,
    wet: &Tensor<R>,
) -> Result<SideFronts<R>, ChannelError>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + BinaryOps<R> + ConditionalOps<R> + UnaryOps<R>,
{
    let zero = Tensor::<R>::zeros(lower.shape(), DType::F64, client.device())?;
    Ok(SideFronts {
        rightward: client.where_cond(wet, &client.maximum(upper, &zero)?, &zero)?,
        leftward: client.where_cond(wet, &client.maximum(&client.neg(lower)?, &zero)?, &zero)?,
        wet: wet.clone(),
    })
}

/// Whether any face has water on exactly one side.
pub(super) fn has_dry_contact<R, C>(
    client: &C,
    left_wet: &Tensor<R>,
    right_wet: &Tensor<R>,
) -> Result<bool, ChannelError>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + CompareOps<R> + ReduceOps<R>,
{
    let contact = client.ne(left_wet, right_wet)?;
    Ok(client.max(&contact, &[0], false)?.item::<f64>()? > 0.0)
}

/// Raise the inward speed into each dry cell to the dry-front bound of its wet run.
///
/// A wet run on the left of face `f` holds the faces `g ≤ f` whose left state is wet, back to
/// the nearest face with a dry left state. Each such face contributes both states of the cell
/// left of it. The mirror run on the right of a face serves a dry cell on its left.
pub(super) fn bound_dry_faces<R, C>(
    client: &C,
    speed: &FaceSpeeds<R>,
    fronts: &DryFronts<R>,
) -> Result<FaceSpeeds<R>, ChannelError>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + BinaryOps<R> + ConditionalOps<R> + ShapeOps<R>,
{
    let DryFronts { left, right } = fronts;
    let n = left.wet.numel();
    let tensors = [
        &left.rightward,
        &left.leftward,
        &right.rightward,
        &right.leftward,
        &right.wet,
        &speed.rightward,
        &speed.leftward,
    ];
    if left.wet.shape().len() != 1 || tensors.iter().any(|t| t.shape() != left.wet.shape()) {
        return Err(ChannelError::numerical(format!(
            "dry-front face shape={:?} differs from another face array: pass one entry per face",
            left.wet.shape()
        )));
    }
    let one = Tensor::<R>::full_scalar(&[n], DType::F64, 1.0, client.device())?;
    let zero = Tensor::<R>::zeros(&[1], DType::F64, client.device())?;
    let zeros = Tensor::<R>::zeros(&[n], DType::F64, client.device())?;
    // The cell left of face g also owns the right state of face g - 1.
    let cell_left = if n > 1 {
        client.maximum(
            &left.rightward,
            &client.cat(&[&zero, &right.rightward.narrow(0, 0, n - 1)?], 0)?,
        )?
    } else {
        left.rightward.clone()
    };
    let cell_left = client.where_cond(&left.wet, &cell_left, &zeros)?;
    let left_dry = client.sub(&one, &left.wet)?;
    let right_dry = client.sub(&one, &right.wet)?;
    let run_left = segmented_max(client, &cell_left, &left_dry, ScanDirection::Forward)?;
    let into_dry_right = client.mul(&left.wet, &right_dry)?;
    let rightward = client.where_cond(
        &into_dry_right,
        &client.maximum(&speed.rightward, &run_left)?,
        &speed.rightward,
    )?;
    // The cell right of face g also owns the left state of face g + 1.
    let cell_right = if n > 1 {
        client.maximum(
            &right.leftward,
            &client.cat(&[&left.leftward.narrow(0, 1, n - 1)?, &zero], 0)?,
        )?
    } else {
        right.leftward.clone()
    };
    let cell_right = client.where_cond(&right.wet, &cell_right, &zeros)?;
    let run_right = segmented_max(client, &cell_right, &right_dry, ScanDirection::Backward)?;
    let into_dry_left = client.mul(&right.wet, &left_dry)?;
    let leftward = client.where_cond(
        &into_dry_left,
        &client.maximum(&speed.leftward, &run_right)?,
        &speed.leftward,
    )?;
    Ok(FaceSpeeds {
        rightward,
        leftward,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};

    type TestResult<T = ()> = Result<T, Box<dyn std::error::Error>>;

    fn cpu() -> CpuClient {
        CpuClient::new(CpuDevice::new()).with_parallelism(ParallelismConfig::new(Some(1), None))
    }

    fn tensor(client: &CpuClient, values: &[f64]) -> TestResult<Tensor<CpuRuntime>> {
        Ok(Tensor::from_slice(
            values,
            &[values.len()],
            client.device(),
        )?)
    }

    /// Fronts from first-order states `[left exterior, cells..., right exterior]`.
    ///
    /// `w` holds `u + φ` and `z` holds `-(u - φ)`. A state is wet where `wet` holds `1`.
    fn fronts(
        client: &CpuClient,
        w: &[f64],
        z: &[f64],
        wet: &[f64],
    ) -> TestResult<DryFronts<CpuRuntime>> {
        let faces = wet.len() - 1;
        let side = |start: usize| -> TestResult<SideFronts<CpuRuntime>> {
            Ok(SideFronts {
                rightward: tensor(client, &w[start..start + faces])?,
                leftward: tensor(client, &z[start..start + faces])?,
                wet: tensor(client, &wet[start..start + faces])?,
            })
        };
        Ok(DryFronts {
            left: side(0)?,
            right: side(1)?,
        })
    }

    fn bound(
        client: &CpuClient,
        rightward: &[f64],
        leftward: &[f64],
        fronts: &DryFronts<CpuRuntime>,
    ) -> TestResult<(Vec<f64>, Vec<f64>)> {
        let speed = FaceSpeeds {
            rightward: tensor(client, rightward)?,
            leftward: tensor(client, leftward)?,
        };
        let bounded = bound_dry_faces(client, &speed, fronts)?;
        Ok((
            bounded.rightward.try_to_vec()?,
            bounded.leftward.try_to_vec()?,
        ))
    }

    #[test]
    fn front_into_a_dry_cell_on_the_right_takes_the_run_maximum() -> TestResult {
        let client = cpu();
        // A wet left exterior feeds cells 0..=2. Cells 3 and 4 and the right exterior are dry.
        let wet = [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0];
        let w = [4.0, 5.0, 9.0, 7.0, 0.0, 0.0, 0.0];
        let fronts = fronts(&client, &w, &[0.0; 7], &wet)?;
        let local = [1.0, 1.0, 1.0, 2.0, 0.0, 0.0];
        let (rightward, leftward) = bound(&client, &local, &local, &fronts)?;
        assert_eq!(rightward, [1.0, 1.0, 1.0, 9.0, 0.0, 0.0]);
        assert_eq!(leftward, local);
        Ok(())
    }

    #[test]
    fn front_into_a_dry_cell_on_the_left_takes_the_run_maximum() -> TestResult {
        let client = cpu();
        // Cells 2..=4 and the right exterior are wet. Cells 0 and 1 and the left exterior are dry.
        let wet = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
        let z = [0.0, 0.0, 0.0, 6.0, 8.0, 3.0, 2.0];
        let fronts = fronts(&client, &[0.0; 7], &z, &wet)?;
        let local = [0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
        let (rightward, leftward) = bound(&client, &local, &local, &fronts)?;
        assert_eq!(rightward, local);
        assert_eq!(leftward, [0.0, 0.0, 8.0, 1.0, 1.0, 1.0]);
        Ok(())
    }

    #[test]
    fn separate_wet_runs_bound_only_their_own_fronts() -> TestResult {
        let client = cpu();
        // Runs: cells 0..=1 and cells 3..=4. Cells 2 and 5 and both exteriors are dry.
        let wet = [0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0];
        let w = [0.0, 10.0, 3.0, 0.0, 2.0, 4.0, 0.0, 0.0];
        let z = [0.0, 3.0, 10.0, 0.0, 4.0, 2.0, 0.0, 0.0];
        let fronts = fronts(&client, &w, &z, &wet)?;
        let local = [0.5; 7];
        let (rightward, leftward) = bound(&client, &local, &local, &fronts)?;
        assert_eq!(rightward, [0.5, 0.5, 10.0, 0.5, 0.5, 4.0, 0.5]);
        assert_eq!(leftward, [10.0, 0.5, 0.5, 4.0, 0.5, 0.5, 0.5]);
        Ok(())
    }

    #[test]
    fn a_faster_local_speed_is_kept() -> TestResult {
        let client = cpu();
        let wet = [1.0, 1.0, 0.0];
        let fronts = fronts(&client, &[3.0, 2.0, 0.0], &[0.0; 3], &wet)?;
        let (rightward, _) = bound(&client, &[0.0, 7.0], &[0.0, 0.0], &fronts)?;
        assert_eq!(rightward, [0.0, 7.0]);
        Ok(())
    }

    #[test]
    fn contact_is_found_only_at_wet_dry_faces() -> TestResult {
        let client = cpu();
        let wet = tensor(&client, &[1.0, 1.0, 0.0])?;
        let dry = tensor(&client, &[0.0, 0.0, 0.0])?;
        assert!(has_dry_contact(&client, &wet, &dry)?);
        assert!(!has_dry_contact(&client, &wet, &wet)?);
        assert!(!has_dry_contact(&client, &dry, &dry)?);
        Ok(())
    }

    #[test]
    fn side_fronts_clamp_and_zero_dry_states() -> TestResult {
        let client = cpu();
        let lower = tensor(&client, &[-4.0, 1.0, -9.0])?;
        let upper = tensor(&client, &[6.0, 5.0, 9.0])?;
        let wet = tensor(&client, &[1.0, 1.0, 0.0])?;
        let side = side_fronts(&client, &lower, &upper, &wet)?;
        assert_eq!(side.rightward.try_to_vec::<f64>()?, [6.0, 5.0, 0.0]);
        assert_eq!(side.leftward.try_to_vec::<f64>()?, [4.0, 0.0, 0.0]);
        Ok(())
    }
}
