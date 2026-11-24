//! Vectorized, scatter-free `DType::F32` 2D finite-volume shallow-water fluxes and friction.
//!
//! Uses only `numr` `F32` tensor ops supported on `CpuRuntime`, `CudaRuntime`, and `WgpuRuntime`.

use crate::contract::EngineError;
use numr::dtype::DType;
use numr::ops::{BinaryOps, CompareOps, ConditionalOps, ReduceOps, ScalarOps, ShapeOps, UnaryOps};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Per-face HLL water/momentum fluxes and hydrostatic bed-pressure corrections.
pub(super) struct SweepDiv<R: Runtime<DType = DType>> {
    pub(super) dh: Tensor<R>,
    pub(super) dqx: Tensor<R>,
    pub(super) dqy: Tensor<R>,
    pub(super) max_speed: f32,
}

#[allow(clippy::too_many_arguments)]
pub(super) fn sweep_x<R: Runtime<DType = DType>>(
    client: &R::Client,
    h: &Tensor<R>,
    qx: &Tensor<R>,
    qy: &Tensor<R>,
    bed: &Tensor<R>,
    active: &Tensor<R>,
    _zero_col: &Tensor<R>,
    dx: f32,
    gravity: f32,
    dry_depth: f32,
) -> Result<SweepDiv<R>, EngineError>
where
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + ShapeOps<R>
        + UnaryOps<R>,
{
    let [ny, nx] = [h.shape()[0], h.shape()[1]];
    if nx == 0 || ny == 0 {
        let z = Tensor::<R>::zeros(&[ny, nx], DType::F32, h.device())?;
        return Ok(SweepDiv {
            dh: z.clone(),
            dqx: z.clone(),
            dqy: z,
            max_speed: 0.0,
        });
    }
    let nf = nx + 1;
    let pad = |t: &Tensor<R>, reflect_sign: f64| -> Result<(Tensor<R>, Tensor<R>), EngineError> {
        let left_edge = t.narrow(1, 0, 1)?.contiguous()?;
        let right_edge = t.narrow(1, nx - 1, 1)?.contiguous()?;
        let (gl, gr) = if reflect_sign < 0.0 {
            (client.neg(&left_edge)?, client.neg(&right_edge)?)
        } else {
            (left_edge, right_edge)
        };
        let ext = client.cat(&[&gl, t, &gr], 1)?;
        Ok((
            ext.narrow(1, 0, nf)?.contiguous()?,
            ext.narrow(1, 1, nf)?.contiguous()?,
        ))
    };

    let (h_l, h_r) = pad(h, 1.0)?;
    let (qn_l, qn_r) = pad(qx, -1.0)?;
    let (qt_l, qt_r) = pad(qy, 1.0)?;
    let (z_l, z_r) = pad(bed, 1.0)?;
    let (act_l, act_r) = pad(active, 1.0)?;

    let (fh, fqn, fqt, corr_l, corr_r, max_speed) = hll_faces(
        client, &h_l, &h_r, &qn_l, &qn_r, &qt_l, &qt_r, &z_l, &z_r, &act_l, &act_r, gravity,
        dry_depth,
    )?;

    let inv_dx = 1.0_f64 / f64::from(dx);
    let fh_west = fh.narrow(1, 0, nx)?.contiguous()?;
    let fh_east = fh.narrow(1, 1, nx)?.contiguous()?;
    let dh = client.mul(
        &client.mul_scalar(&client.sub(&fh_west, &fh_east)?, inv_dx)?,
        active,
    )?;

    let fqn_west = fqn.narrow(1, 0, nx)?.contiguous()?;
    let fqn_east = fqn.narrow(1, 1, nx)?.contiguous()?;
    let p_east = corr_l.narrow(1, 1, nx)?.contiguous()?;
    let p_west = corr_r.narrow(1, 0, nx)?.contiguous()?;
    let dqx = client.mul(
        &client.mul_scalar(
            &client.add(
                &client.sub(&fqn_west, &fqn_east)?,
                &client.sub(&p_east, &p_west)?,
            )?,
            inv_dx,
        )?,
        active,
    )?;

    let fqt_west = fqt.narrow(1, 0, nx)?.contiguous()?;
    let fqt_east = fqt.narrow(1, 1, nx)?.contiguous()?;
    let dqy = client.mul(
        &client.mul_scalar(&client.sub(&fqt_west, &fqt_east)?, inv_dx)?,
        active,
    )?;

    Ok(SweepDiv {
        dh,
        dqx,
        dqy,
        max_speed,
    })
}

#[allow(clippy::too_many_arguments)]
pub(super) fn sweep_y<R: Runtime<DType = DType>>(
    client: &R::Client,
    h: &Tensor<R>,
    qx: &Tensor<R>,
    qy: &Tensor<R>,
    bed: &Tensor<R>,
    active: &Tensor<R>,
    _zero_row: &Tensor<R>,
    dy: f32,
    gravity: f32,
    dry_depth: f32,
) -> Result<SweepDiv<R>, EngineError>
where
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + ShapeOps<R>
        + UnaryOps<R>,
{
    let [ny, nx] = [h.shape()[0], h.shape()[1]];
    if nx == 0 || ny == 0 {
        let z = Tensor::<R>::zeros(&[ny, nx], DType::F32, h.device())?;
        return Ok(SweepDiv {
            dh: z.clone(),
            dqx: z.clone(),
            dqy: z,
            max_speed: 0.0,
        });
    }
    let nf = ny + 1;
    let pad = |t: &Tensor<R>, reflect_sign: f64| -> Result<(Tensor<R>, Tensor<R>), EngineError> {
        let bot_edge = t.narrow(0, 0, 1)?.contiguous()?;
        let top_edge = t.narrow(0, ny - 1, 1)?.contiguous()?;
        let (gb, gt) = if reflect_sign < 0.0 {
            (client.neg(&bot_edge)?, client.neg(&top_edge)?)
        } else {
            (bot_edge, top_edge)
        };
        let ext = client.cat(&[&gb, t, &gt], 0)?;
        Ok((
            ext.narrow(0, 0, nf)?.contiguous()?,
            ext.narrow(0, 1, nf)?.contiguous()?,
        ))
    };

    let (h_l, h_r) = pad(h, 1.0)?;
    let (qn_l, qn_r) = pad(qy, -1.0)?;
    let (qt_l, qt_r) = pad(qx, 1.0)?;
    let (z_l, z_r) = pad(bed, 1.0)?;
    let (act_l, act_r) = pad(active, 1.0)?;

    let (fh, fqn, fqt, corr_l, corr_r, max_speed) = hll_faces(
        client, &h_l, &h_r, &qn_l, &qn_r, &qt_l, &qt_r, &z_l, &z_r, &act_l, &act_r, gravity,
        dry_depth,
    )?;

    let inv_dy = 1.0_f64 / f64::from(dy);
    let fh_south = fh.narrow(0, 0, ny)?.contiguous()?;
    let fh_north = fh.narrow(0, 1, ny)?.contiguous()?;
    let dh = client.mul(
        &client.mul_scalar(&client.sub(&fh_south, &fh_north)?, inv_dy)?,
        active,
    )?;

    let fqn_south = fqn.narrow(0, 0, ny)?.contiguous()?;
    let fqn_north = fqn.narrow(0, 1, ny)?.contiguous()?;
    let p_north = corr_l.narrow(0, 1, ny)?.contiguous()?;
    let p_south = corr_r.narrow(0, 0, ny)?.contiguous()?;
    let dqy = client.mul(
        &client.mul_scalar(
            &client.add(
                &client.sub(&fqn_south, &fqn_north)?,
                &client.sub(&p_north, &p_south)?,
            )?,
            inv_dy,
        )?,
        active,
    )?;

    let fqt_south = fqt.narrow(0, 0, ny)?.contiguous()?;
    let fqt_north = fqt.narrow(0, 1, ny)?.contiguous()?;
    let dqx = client.mul(
        &client.mul_scalar(&client.sub(&fqt_south, &fqt_north)?, inv_dy)?,
        active,
    )?;

    Ok(SweepDiv {
        dh,
        dqx,
        dqy,
        max_speed,
    })
}

/// Hydrostatic reconstruction + HLL Riemann flux on faces in `DType::F32`.
///
/// Inactive neighbors adjacent to an active cell act as reflecting impermeable walls.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn hll_faces<R: Runtime<DType = DType>>(
    client: &R::Client,
    h_l: &Tensor<R>,
    h_r: &Tensor<R>,
    qn_l: &Tensor<R>,
    qn_r: &Tensor<R>,
    qt_l: &Tensor<R>,
    qt_r: &Tensor<R>,
    z_l: &Tensor<R>,
    z_r: &Tensor<R>,
    act_l: &Tensor<R>,
    act_r: &Tensor<R>,
    gravity: f32,
    dry_depth: f32,
) -> Result<(Tensor<R>, Tensor<R>, Tensor<R>, Tensor<R>, Tensor<R>, f32), EngineError>
where
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + UnaryOps<R>,
{
    let shape = h_l.shape();
    let device = h_l.device();
    let zero = Tensor::<R>::zeros(shape, DType::F32, device)?;
    let eps = Tensor::<R>::full_scalar(shape, DType::F32, f64::from(dry_depth), device)?;
    let g = f64::from(gravity);
    let half_g = 0.5 * g;

    // Reflect any inactive neighbor across the face so active-inactive interfaces are impermeable walls.
    let l_active = client.gt(act_l, &zero)?;
    let r_active = client.gt(act_r, &zero)?;
    let face_active = client.maximum(act_l, act_r)?;
    let h_l = &client.where_cond(&l_active, h_l, h_r)?;
    let h_r = &client.where_cond(&r_active, h_r, h_l)?;
    let neg_qn_l = client.neg(qn_l)?;
    let neg_qn_r = client.neg(qn_r)?;
    let qn_l = &client.where_cond(&l_active, qn_l, &neg_qn_r)?;
    let qn_r = &client.where_cond(&r_active, qn_r, &neg_qn_l)?;
    let qt_l = &client.where_cond(&l_active, qt_l, qt_r)?;
    let qt_r = &client.where_cond(&r_active, qt_r, qt_l)?;
    let z_l = &client.where_cond(&l_active, z_l, z_r)?;
    let z_r = &client.where_cond(&r_active, z_r, z_l)?;

    // Hydrostatic bed reconstruction: z* = max(z_L, z_R), h* = max(0, h + z - z*).
    let z_star = client.maximum(z_l, z_r)?;
    let hs_l = client.maximum(&client.sub(&client.add(h_l, z_l)?, &z_star)?, &zero)?;
    let hs_r = client.maximum(&client.sub(&client.add(h_r, z_r)?, &z_star)?, &zero)?;

    // Velocities from original cells (zero when dry).
    let wet_l = client.gt(h_l, &eps)?;
    let wet_r = client.gt(h_r, &eps)?;
    let safe_hl = client.where_cond(&wet_l, h_l, &eps)?;
    let safe_hr = client.where_cond(&wet_r, h_r, &eps)?;
    let un_l = client.where_cond(&wet_l, &client.div(qn_l, &safe_hl)?, &zero)?;
    let un_r = client.where_cond(&wet_r, &client.div(qn_r, &safe_hr)?, &zero)?;
    let ut_l = client.where_cond(&wet_l, &client.div(qt_l, &safe_hl)?, &zero)?;
    let ut_r = client.where_cond(&wet_r, &client.div(qt_r, &safe_hr)?, &zero)?;

    // Reconstructed discharges.
    let qs_nl = client.mul(&hs_l, &un_l)?;
    let qs_nr = client.mul(&hs_r, &un_r)?;
    let qs_tl = client.mul(&hs_l, &ut_l)?;
    let qs_tr = client.mul(&hs_r, &ut_r)?;

    // Gravity wave speeds.
    let c_l = client.sqrt(&client.mul_scalar(&hs_l, g)?)?;
    let c_r = client.sqrt(&client.mul_scalar(&hs_r, g)?)?;
    let s_l = client.minimum(&client.sub(&un_l, &c_l)?, &client.sub(&un_r, &c_r)?)?;
    let s_r = client.maximum(&client.add(&un_l, &c_l)?, &client.add(&un_r, &c_r)?)?;
    let s_l_neg = client.minimum(&s_l, &zero)?;
    let s_r_pos = client.maximum(&s_r, &zero)?;

    let denom_raw = client.sub(&s_r_pos, &s_l_neg)?;
    let has_wave = client.gt(&denom_raw, &zero)?;
    let one = Tensor::<R>::ones(shape, DType::F32, device)?;
    let denom = client.where_cond(&has_wave, &denom_raw, &one)?;

    // Physical fluxes F_L, F_R.
    let p_sl = client.mul_scalar(&client.mul(&hs_l, &hs_l)?, half_g)?;
    let p_sr = client.mul_scalar(&client.mul(&hs_r, &hs_r)?, half_g)?;
    let fn_l = client.add(&client.mul(&qs_nl, &un_l)?, &p_sl)?;
    let fn_r = client.add(&client.mul(&qs_nr, &un_r)?, &p_sr)?;
    let ft_l = client.mul(&qs_tl, &un_l)?;
    let ft_r = client.mul(&qs_tr, &un_r)?;

    let s_prod = client.mul(&s_l_neg, &s_r_pos)?;
    let hll = |fl: &Tensor<R>,
               fr: &Tensor<R>,
               ul: &Tensor<R>,
               ur: &Tensor<R>|
     -> Result<Tensor<R>, EngineError> {
        let num = client.add(
            &client.sub(&client.mul(&s_r_pos, fl)?, &client.mul(&s_l_neg, fr)?)?,
            &client.mul(&s_prod, &client.sub(ur, ul)?)?,
        )?;
        let raw = client.div(&num, &denom)?;
        Ok(client.mul(&client.where_cond(&has_wave, &raw, &zero)?, &face_active)?)
    };

    let fh = hll(&qs_nl, &qs_nr, &hs_l, &hs_r)?;
    let fqn = hll(&fn_l, &fn_r, &qs_nl, &qs_nr)?;
    let fqt = hll(&ft_l, &ft_r, &qs_tl, &qs_tr)?;

    // Hydrostatic bed-pressure corrections: 0.5*g*(h*^2 - h^2) masked by face_active.
    let p_orig_l = client.mul_scalar(&client.mul(h_l, h_l)?, half_g)?;
    let p_orig_r = client.mul_scalar(&client.mul(h_r, h_r)?, half_g)?;
    let corr_l = client.mul(&client.sub(&p_sl, &p_orig_l)?, &face_active)?;
    let corr_r = client.mul(&client.sub(&p_sr, &p_orig_r)?, &face_active)?;

    let abs_sl = client.abs(&s_l_neg)?;
    let abs_sr = client.abs(&s_r_pos)?;
    let max_face_s = client.mul(&client.maximum(&abs_sl, &abs_sr)?, &face_active)?;
    let max_speed = client.max(&max_face_s, &[0, 1], false)?.item::<f32>()?;

    Ok((fh, fqn, fqt, corr_l, corr_r, max_speed))
}

/// Applies semi-implicit Manning bed friction in `DType::F32`.
#[allow(clippy::too_many_arguments)]
pub(super) fn apply_manning_f32<R: Runtime<DType = DType>>(
    client: &R::Client,
    h: &Tensor<R>,
    qx: &Tensor<R>,
    qy: &Tensor<R>,
    manning_sq: &Tensor<R>,
    dt: f32,
    gravity: f32,
    dry_depth: f32,
) -> Result<(Tensor<R>, Tensor<R>), EngineError>
where
    R::Client: BinaryOps<R> + CompareOps<R> + ConditionalOps<R> + ScalarOps<R> + UnaryOps<R>,
{
    let shape = h.shape();
    let device = h.device();
    let zero = Tensor::<R>::zeros(shape, DType::F32, device)?;
    let eps = Tensor::<R>::full_scalar(shape, DType::F32, f64::from(dry_depth), device)?;
    let wet = client.gt(h, &eps)?;
    let safe_h = client.where_cond(&wet, h, &eps)?;

    let ux = client.where_cond(&wet, &client.div(qx, &safe_h)?, &zero)?;
    let uy = client.where_cond(&wet, &client.div(qy, &safe_h)?, &zero)?;
    let speed = client.sqrt(&client.add(&client.mul(&ux, &ux)?, &client.mul(&uy, &uy)?)?)?;

    let h_4_3 = client.pow_scalar(&safe_h, 4.0 / 3.0)?;
    let coeff = f64::from(dt) * f64::from(gravity);
    let drag = client.div(
        &client.mul_scalar(&client.mul(manning_sq, &speed)?, coeff)?,
        &h_4_3,
    )?;
    let denom = client.add_scalar(&drag, 1.0)?;

    let new_qx = client.where_cond(&wet, &client.div(qx, &denom)?, &zero)?;
    let new_qy = client.where_cond(&wet, &client.div(qy, &denom)?, &zero)?;
    Ok((new_qx, new_qy))
}
