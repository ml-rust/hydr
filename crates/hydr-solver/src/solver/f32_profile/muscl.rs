//! 2nd-order MUSCL minmod slope-limited reconstruction in `DType::F32`.
//!
//! Preserves exact lake-at-rest well-balancing by limiting water-surface elevation `eta = h + z`
//! and suppressing slopes at dry/near-dry interfaces.

use crate::contract::EngineError;
use numr::dtype::DType;
use numr::ops::{BinaryOps, CompareOps, ConditionalOps, ScalarOps, ShapeOps, UnaryOps};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Spatial reconstruction order for `SurfaceSolverF32`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum SurfaceF32SpatialOrder {
    /// First-order piecewise-constant hydrostatic reconstruction (default).
    #[default]
    FirstOrder,
    /// Second-order MUSCL minmod-limited hydrostatic reconstruction (sharper wave fronts).
    SecondOrderMuscl,
}

/// Computes `minmod(a, b) = 0.5 * (sign(a) + sign(b)) * min(|a|, |b|)` in `DType::F32`.
fn minmod<R: Runtime<DType = DType>>(
    client: &R::Client,
    a: &Tensor<R>,
    b: &Tensor<R>,
) -> Result<Tensor<R>, EngineError>
where
    R::Client: BinaryOps<R> + CompareOps<R> + ConditionalOps<R> + ScalarOps<R> + UnaryOps<R>,
{
    let zero = Tensor::<R>::zeros(a.shape(), DType::F32, a.device())?;
    let prod = client.mul(a, b)?;
    let same_sign = client.gt(&prod, &zero)?;
    let abs_a = client.abs(a)?;
    let abs_b = client.abs(b)?;
    let min_abs = client.minimum(&abs_a, &abs_b)?;
    let a_pos = client.gt(a, &zero)?;
    let signed = client.where_cond(&a_pos, &min_abs, &client.neg(&min_abs)?)?;
    Ok(client.where_cond(&same_sign, &signed, &zero)?)
}

/// Computes minmod-limited cell slopes along `axis` (`1` for x, `0` for y) from ghost-padded
/// tensor `ext` of extent `n + 2` along `axis`.
fn limited_slope<R: Runtime<DType = DType>>(
    client: &R::Client,
    ext: &Tensor<R>,
    axis: isize,
    n: usize,
) -> Result<Tensor<R>, EngineError>
where
    R::Client: BinaryOps<R> + CompareOps<R> + ConditionalOps<R> + ScalarOps<R> + UnaryOps<R>,
{
    let left = ext.narrow(axis, 0, n)?.contiguous()?;
    let mid = ext.narrow(axis, 1, n)?.contiguous()?;
    let right = ext.narrow(axis, 2, n)?.contiguous()?;
    let dl = client.sub(&mid, &left)?;
    let dr = client.sub(&right, &mid)?;
    minmod(client, &dl, &dr)
}

/// Applies well-balanced MUSCL minmod reconstruction to ghost-padded `(h, qn, qt, z)` along `axis`
/// (`1` for x, `0` for y), returning `(h_l, h_r, qn_l, qn_r, qt_l, qt_r, z_l, z_r, s_c)` where
/// `s_c = -2 * g * h * s_z` is the Audusse-Bristeau centered bed-slope source term on interior cells.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub(super) fn muscl_reconstruct_faces<R: Runtime<DType = DType>>(
    client: &R::Client,
    h_ext: &Tensor<R>,
    qn_ext: &Tensor<R>,
    qt_ext: &Tensor<R>,
    z_ext: &Tensor<R>,
    axis: isize,
    n: usize,
    gravity: f32,
    dry_depth: f32,
) -> Result<
    (
        Tensor<R>,
        Tensor<R>,
        Tensor<R>,
        Tensor<R>,
        Tensor<R>,
        Tensor<R>,
        Tensor<R>,
        Tensor<R>,
        Tensor<R>,
    ),
    EngineError,
>
where
    R::Client:
        BinaryOps<R> + CompareOps<R> + ConditionalOps<R> + ScalarOps<R> + ShapeOps<R> + UnaryOps<R>,
{
    let nf = n + 1;
    let h_l0 = h_ext.narrow(axis, 0, nf)?.contiguous()?;
    let h_r0 = h_ext.narrow(axis, 1, nf)?.contiguous()?;
    let qn_l0 = qn_ext.narrow(axis, 0, nf)?.contiguous()?;
    let qn_r0 = qn_ext.narrow(axis, 1, nf)?.contiguous()?;
    let qt_l0 = qt_ext.narrow(axis, 0, nf)?.contiguous()?;
    let qt_r0 = qt_ext.narrow(axis, 1, nf)?.contiguous()?;
    let z_l0 = z_ext.narrow(axis, 0, nf)?.contiguous()?;
    let z_r0 = z_ext.narrow(axis, 1, nf)?.contiguous()?;
    let h_mid = h_ext.narrow(axis, 1, n)?.contiguous()?;
    let zero_c = Tensor::<R>::zeros(h_mid.shape(), DType::F32, h_mid.device())?;

    if n < 2 {
        return Ok((h_l0, h_r0, qn_l0, qn_r0, qt_l0, qt_r0, z_l0, z_r0, zero_c));
    }

    // Slopes on interior cells (length n along axis); zero slope on exterior ghost cells.
    let eta_ext = client.add(h_ext, z_ext)?;
    let d_eta = limited_slope(client, &eta_ext, axis, n)?;
    let d_h = limited_slope(client, h_ext, axis, n)?;
    let d_qn = limited_slope(client, qn_ext, axis, n)?;
    let d_qt = limited_slope(client, qt_ext, axis, n)?;

    // Suppress MUSCL slope on dry cells or where depth reconstruction would go negative.
    let eps = Tensor::<R>::full_scalar(
        h_mid.shape(),
        DType::F32,
        f64::from(dry_depth),
        h_mid.device(),
    )?;
    let wet = client.gt(&h_mid, &eps)?;
    let half_dh = client.mul_scalar(&d_h, 0.5)?;
    let half_abs_dh = client.abs(&half_dh)?;
    let pos_ok = client.gt(&client.sub(&h_mid, &half_abs_dh)?, &eps)?;
    let mask = client.where_cond(
        &wet,
        &client.where_cond(
            &pos_ok,
            &Tensor::<R>::ones(h_mid.shape(), DType::F32, h_mid.device())?,
            &zero_c,
        )?,
        &zero_c,
    )?;

    let s_eta = client.mul(&client.mul_scalar(&d_eta, 0.5)?, &mask)?;
    let s_h = client.mul(&half_dh, &mask)?;
    let s_z = client.sub(&s_eta, &s_h)?;
    let s_qn = client.mul(&client.mul_scalar(&d_qn, 0.5)?, &mask)?;
    let s_qt = client.mul(&client.mul_scalar(&d_qt, 0.5)?, &mask)?;

    let mut edge_shape = h_mid.shape().to_vec();
    edge_shape[axis as usize] = 1;
    let zero_e = Tensor::<R>::zeros(&edge_shape, DType::F32, h_mid.device())?;

    let s_h_l = client.cat(&[&zero_e, &s_h], axis)?;
    let s_h_r = client.cat(&[&s_h, &zero_e], axis)?;
    let s_z_l = client.cat(&[&zero_e, &s_z], axis)?;
    let s_z_r = client.cat(&[&s_z, &zero_e], axis)?;
    let s_qn_l = client.cat(&[&zero_e, &s_qn], axis)?;
    let s_qn_r = client.cat(&[&s_qn, &zero_e], axis)?;
    let s_qt_l = client.cat(&[&zero_e, &s_qt], axis)?;
    let s_qt_r = client.cat(&[&s_qt, &zero_e], axis)?;

    let h_l = client.add(&h_l0, &s_h_l)?;
    let h_r = client.sub(&h_r0, &s_h_r)?;
    let z_l = client.add(&z_l0, &s_z_l)?;
    let z_r = client.sub(&z_r0, &s_z_r)?;
    let qn_l = client.add(&qn_l0, &s_qn_l)?;
    let qn_r = client.sub(&qn_r0, &s_qn_r)?;
    let qt_l = client.add(&qt_l0, &s_qt_l)?;
    let qt_r = client.sub(&qt_r0, &s_qt_r)?;

    // Audusse-Bristeau centered bed source: -2 * g * h_i * s_z_i cancels 0.5*g*((h-s_h)^2 - (h+s_h)^2) at rest.
    let s_c = client.mul_scalar(&client.mul(&h_mid, &s_z)?, -2.0 * f64::from(gravity))?;
    Ok((h_l, h_r, qn_l, qn_r, qt_l, qt_r, z_l, z_r, s_c))
}
