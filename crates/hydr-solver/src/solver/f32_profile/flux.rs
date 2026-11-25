//! Vectorized, scatter-free `DType::F32` 2D finite-volume shallow-water fluxes and friction.
//!
//! Uses only `numr` `F32` tensor ops supported on `CpuRuntime`, `CudaRuntime`, and `WgpuRuntime`.

use super::boundaries::{SurfaceF32SideBoundary, build_ghost_edge};
use super::muscl::{SurfaceF32SpatialOrder, muscl_reconstruct_faces};
use crate::contract::EngineError;
use numr::dtype::DType;
use numr::ops::{BinaryOps, CompareOps, ConditionalOps, ReduceOps, ScalarOps, ShapeOps, UnaryOps};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Per-face HLL water/momentum fluxes, hydrostatic bed corrections, and exterior boundary rates
/// kept entirely in device tensors (no synchronous host readbacks).
pub(super) struct SweepTensors<R: Runtime<DType = DType>> {
    pub(super) dh: Tensor<R>,
    pub(super) dqx: Tensor<R>,
    pub(super) dqy: Tensor<R>,
    /// Maximum wave speed across all faces (shape `[1]`, `F32`).
    pub(super) max_speed: Tensor<R>,
    /// Total inward exterior boundary flux sum before multiplying by transverse cell width (shape `[1]`, `F32`).
    pub(super) boundary_in_flux: Tensor<R>,
    /// Total outward exterior boundary flux sum before multiplying by transverse cell width (shape `[1]`, `F32`).
    pub(super) boundary_out_flux: Tensor<R>,
}

/// Per-face HLL water/momentum fluxes, hydrostatic bed corrections, and exterior boundary rates.
pub(super) struct SweepDiv<R: Runtime<DType = DType>> {
    pub(super) dh: Tensor<R>,
    pub(super) dqx: Tensor<R>,
    pub(super) dqy: Tensor<R>,
    pub(super) max_speed: f32,
    /// Inward exterior boundary volume rate (`m³/s`, `>= 0`).
    pub(super) boundary_in_m3_per_s: f64,
    /// Outward exterior boundary volume rate (`m³/s`, `>= 0`).
    pub(super) boundary_out_m3_per_s: f64,
}

#[allow(clippy::too_many_arguments)]
pub(super) fn sweep_x<R: Runtime<DType = DType>>(
    client: &R::Client,
    h: &Tensor<R>,
    qx: &Tensor<R>,
    qy: &Tensor<R>,
    bed: &Tensor<R>,
    active: &Tensor<R>,
    west: SurfaceF32SideBoundary,
    east: SurfaceF32SideBoundary,
    west_mask: Option<&Tensor<R>>,
    east_mask: Option<&Tensor<R>>,
    dx: f32,
    dy: f32,
    gravity: f32,
    dry_depth: f32,
    order: SurfaceF32SpatialOrder,
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
    let t = sweep_x_tensors(
        client, h, qx, qy, bed, active, west, east, west_mask, east_mask, dx, gravity, dry_depth,
        order,
    )?;
    let max_speed = t.max_speed.item::<f32>()?;
    let b_in = f64::from(t.boundary_in_flux.item::<f32>()?) * f64::from(dy);
    let b_out = f64::from(t.boundary_out_flux.item::<f32>()?) * f64::from(dy);
    Ok(SweepDiv {
        dh: t.dh,
        dqx: t.dqx,
        dqy: t.dqy,
        max_speed,
        boundary_in_m3_per_s: b_in,
        boundary_out_m3_per_s: b_out,
    })
}

#[allow(clippy::too_many_arguments)]
pub(super) fn sweep_x_tensors<R: Runtime<DType = DType>>(
    client: &R::Client,
    h: &Tensor<R>,
    qx: &Tensor<R>,
    qy: &Tensor<R>,
    bed: &Tensor<R>,
    active: &Tensor<R>,
    west: SurfaceF32SideBoundary,
    east: SurfaceF32SideBoundary,
    west_mask: Option<&Tensor<R>>,
    east_mask: Option<&Tensor<R>>,
    dx: f32,
    gravity: f32,
    dry_depth: f32,
    order: SurfaceF32SpatialOrder,
) -> Result<SweepTensors<R>, EngineError>
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
        let z1 = Tensor::<R>::zeros(&[1], DType::F32, h.device())?;
        return Ok(SweepTensors {
            dh: z.clone(),
            dqx: z.clone(),
            dqy: z,
            max_speed: z1.clone(),
            boundary_in_flux: z1.clone(),
            boundary_out_flux: z1,
        });
    }
    let nf = nx + 1;
    let (hw, qxw, qyw, zw) = build_ghost_edge(
        client,
        west,
        west_mask,
        &h.narrow(1, 0, 1)?.contiguous()?,
        &qx.narrow(1, 0, 1)?.contiguous()?,
        &qy.narrow(1, 0, 1)?.contiguous()?,
        &bed.narrow(1, 0, 1)?.contiguous()?,
        1.0,
    )?;
    let (he, qxe, qye, ze) = build_ghost_edge(
        client,
        east,
        east_mask,
        &h.narrow(1, nx - 1, 1)?.contiguous()?,
        &qx.narrow(1, nx - 1, 1)?.contiguous()?,
        &qy.narrow(1, nx - 1, 1)?.contiguous()?,
        &bed.narrow(1, nx - 1, 1)?.contiguous()?,
        -1.0,
    )?;
    let aw = active.narrow(1, 0, 1)?.contiguous()?;
    let ae = active.narrow(1, nx - 1, 1)?.contiguous()?;

    let h_ext = client.cat(&[&hw, h, &he], 1)?;
    let qn_ext = client.cat(&[&qxw, qx, &qxe], 1)?;
    let qt_ext = client.cat(&[&qyw, qy, &qye], 1)?;
    let z_ext = client.cat(&[&zw, bed, &ze], 1)?;
    let act_ext = client.cat(&[&aw, active, &ae], 1)?;
    let act_l = act_ext.narrow(1, 0, nf)?.contiguous()?;
    let act_r = act_ext.narrow(1, 1, nf)?.contiguous()?;

    let (h_l, h_r, qn_l, qn_r, qt_l, qt_r, z_l, z_r, s_c) = match order {
        SurfaceF32SpatialOrder::FirstOrder => (
            h_ext.narrow(1, 0, nf)?.contiguous()?,
            h_ext.narrow(1, 1, nf)?.contiguous()?,
            qn_ext.narrow(1, 0, nf)?.contiguous()?,
            qn_ext.narrow(1, 1, nf)?.contiguous()?,
            qt_ext.narrow(1, 0, nf)?.contiguous()?,
            qt_ext.narrow(1, 1, nf)?.contiguous()?,
            z_ext.narrow(1, 0, nf)?.contiguous()?,
            z_ext.narrow(1, 1, nf)?.contiguous()?,
            Tensor::<R>::zeros(&[ny, nx], DType::F32, h.device())?,
        ),
        SurfaceF32SpatialOrder::SecondOrderMuscl => muscl_reconstruct_faces(
            client, &h_ext, &qn_ext, &qt_ext, &z_ext, 1, nx, gravity, dry_depth,
        )?,
    };

    let (fh, fqn, fqt, corr_l, corr_r, max_speed) = hll_faces_tensors(
        client, &h_l, &h_r, &qn_l, &qn_r, &qt_l, &qt_r, &z_l, &z_r, &act_l, &act_r, gravity,
        dry_depth,
    )?;

    // Track exterior boundary mass exchange on west (face 0, + is inward) and east (face nx, - is inward).
    let zero_edge = Tensor::<R>::zeros(&[ny, 1], DType::F32, h.device())?;
    let fh_w_edge = fh.narrow(1, 0, 1)?.contiguous()?;
    let fh_e_edge = fh.narrow(1, nx, 1)?.contiguous()?;
    let w_in = client.sum(&client.maximum(&fh_w_edge, &zero_edge)?, &[0, 1], false)?;
    let w_out = client.sum(
        &client.maximum(&client.neg(&fh_w_edge)?, &zero_edge)?,
        &[0, 1],
        false,
    )?;
    let e_in = client.sum(
        &client.maximum(&client.neg(&fh_e_edge)?, &zero_edge)?,
        &[0, 1],
        false,
    )?;
    let e_out = client.sum(&client.maximum(&fh_e_edge, &zero_edge)?, &[0, 1], false)?;
    let boundary_in_flux = client.add(&w_in, &e_in)?.reshape(&[1])?;
    let boundary_out_flux = client.add(&w_out, &e_out)?.reshape(&[1])?;

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
                &client.add(
                    &client.sub(&fqn_west, &fqn_east)?,
                    &client.sub(&p_east, &p_west)?,
                )?,
                &s_c,
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

    Ok(SweepTensors {
        dh,
        dqx,
        dqy,
        max_speed,
        boundary_in_flux,
        boundary_out_flux,
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
    south: SurfaceF32SideBoundary,
    north: SurfaceF32SideBoundary,
    south_mask: Option<&Tensor<R>>,
    north_mask: Option<&Tensor<R>>,
    dx: f32,
    dy: f32,
    gravity: f32,
    dry_depth: f32,
    order: SurfaceF32SpatialOrder,
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
    let t = sweep_y_tensors(
        client, h, qx, qy, bed, active, south, north, south_mask, north_mask, dy, gravity,
        dry_depth, order,
    )?;
    let max_speed = t.max_speed.item::<f32>()?;
    let b_in = f64::from(t.boundary_in_flux.item::<f32>()?) * f64::from(dx);
    let b_out = f64::from(t.boundary_out_flux.item::<f32>()?) * f64::from(dx);
    Ok(SweepDiv {
        dh: t.dh,
        dqx: t.dqx,
        dqy: t.dqy,
        max_speed,
        boundary_in_m3_per_s: b_in,
        boundary_out_m3_per_s: b_out,
    })
}

#[allow(clippy::too_many_arguments)]
pub(super) fn sweep_y_tensors<R: Runtime<DType = DType>>(
    client: &R::Client,
    h: &Tensor<R>,
    qx: &Tensor<R>,
    qy: &Tensor<R>,
    bed: &Tensor<R>,
    active: &Tensor<R>,
    south: SurfaceF32SideBoundary,
    north: SurfaceF32SideBoundary,
    south_mask: Option<&Tensor<R>>,
    north_mask: Option<&Tensor<R>>,
    dy: f32,
    gravity: f32,
    dry_depth: f32,
    order: SurfaceF32SpatialOrder,
) -> Result<SweepTensors<R>, EngineError>
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
        let z1 = Tensor::<R>::zeros(&[1], DType::F32, h.device())?;
        return Ok(SweepTensors {
            dh: z.clone(),
            dqx: z.clone(),
            dqy: z,
            max_speed: z1.clone(),
            boundary_in_flux: z1.clone(),
            boundary_out_flux: z1,
        });
    }
    let nf = ny + 1;
    let (hs, qys, qxs, zs) = build_ghost_edge(
        client,
        south,
        south_mask,
        &h.narrow(0, 0, 1)?.contiguous()?,
        &qy.narrow(0, 0, 1)?.contiguous()?,
        &qx.narrow(0, 0, 1)?.contiguous()?,
        &bed.narrow(0, 0, 1)?.contiguous()?,
        1.0,
    )?;
    let (hn, qyn, qxn, zn) = build_ghost_edge(
        client,
        north,
        north_mask,
        &h.narrow(0, ny - 1, 1)?.contiguous()?,
        &qy.narrow(0, ny - 1, 1)?.contiguous()?,
        &qx.narrow(0, ny - 1, 1)?.contiguous()?,
        &bed.narrow(0, ny - 1, 1)?.contiguous()?,
        -1.0,
    )?;
    let as_ = active.narrow(0, 0, 1)?.contiguous()?;
    let an = active.narrow(0, ny - 1, 1)?.contiguous()?;

    let h_ext = client.cat(&[&hs, h, &hn], 0)?;
    let qn_ext = client.cat(&[&qys, qy, &qyn], 0)?;
    let qt_ext = client.cat(&[&qxs, qx, &qxn], 0)?;
    let z_ext = client.cat(&[&zs, bed, &zn], 0)?;
    let act_ext = client.cat(&[&as_, active, &an], 0)?;
    let act_l = act_ext.narrow(0, 0, nf)?.contiguous()?;
    let act_r = act_ext.narrow(0, 1, nf)?.contiguous()?;

    let (h_l, h_r, qn_l, qn_r, qt_l, qt_r, z_l, z_r, s_c) = match order {
        SurfaceF32SpatialOrder::FirstOrder => (
            h_ext.narrow(0, 0, nf)?.contiguous()?,
            h_ext.narrow(0, 1, nf)?.contiguous()?,
            qn_ext.narrow(0, 0, nf)?.contiguous()?,
            qn_ext.narrow(0, 1, nf)?.contiguous()?,
            qt_ext.narrow(0, 0, nf)?.contiguous()?,
            qt_ext.narrow(0, 1, nf)?.contiguous()?,
            z_ext.narrow(0, 0, nf)?.contiguous()?,
            z_ext.narrow(0, 1, nf)?.contiguous()?,
            Tensor::<R>::zeros(&[ny, nx], DType::F32, h.device())?,
        ),
        SurfaceF32SpatialOrder::SecondOrderMuscl => muscl_reconstruct_faces(
            client, &h_ext, &qn_ext, &qt_ext, &z_ext, 0, ny, gravity, dry_depth,
        )?,
    };

    let (fh, fqn, fqt, corr_l, corr_r, max_speed) = hll_faces_tensors(
        client, &h_l, &h_r, &qn_l, &qn_r, &qt_l, &qt_r, &z_l, &z_r, &act_l, &act_r, gravity,
        dry_depth,
    )?;

    let zero_edge = Tensor::<R>::zeros(&[1, nx], DType::F32, h.device())?;
    let fh_s_edge = fh.narrow(0, 0, 1)?.contiguous()?;
    let fh_n_edge = fh.narrow(0, ny, 1)?.contiguous()?;
    let s_in = client.sum(&client.maximum(&fh_s_edge, &zero_edge)?, &[0, 1], false)?;
    let s_out = client.sum(
        &client.maximum(&client.neg(&fh_s_edge)?, &zero_edge)?,
        &[0, 1],
        false,
    )?;
    let n_in = client.sum(
        &client.maximum(&client.neg(&fh_n_edge)?, &zero_edge)?,
        &[0, 1],
        false,
    )?;
    let n_out = client.sum(&client.maximum(&fh_n_edge, &zero_edge)?, &[0, 1], false)?;
    let boundary_in_flux = client.add(&s_in, &n_in)?.reshape(&[1])?;
    let boundary_out_flux = client.add(&s_out, &n_out)?.reshape(&[1])?;

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
                &client.add(
                    &client.sub(&fqn_south, &fqn_north)?,
                    &client.sub(&p_north, &p_south)?,
                )?,
                &s_c,
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

    Ok(SweepTensors {
        dh,
        dqx,
        dqy,
        max_speed,
        boundary_in_flux,
        boundary_out_flux,
    })
}

/// Hydrostatic reconstruction + HLL Riemann flux on faces in `DType::F32`.
///
/// Inactive neighbors adjacent to an active cell act as reflecting impermeable walls.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn hll_faces_tensors<R: Runtime<DType = DType>>(
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
) -> Result<
    (
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
    let max_speed = client.max(&max_face_s, &[0, 1], false)?.reshape(&[1])?;

    Ok((fh, fqn, fqt, corr_l, corr_r, max_speed))
}
