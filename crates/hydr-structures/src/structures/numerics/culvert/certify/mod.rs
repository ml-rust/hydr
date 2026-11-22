//! Construction-time depth certificates of arc barrels, in unit geometry `s = y/D`.
//!
//! An arc barrel has free shape ratios, so no frozen constant bounds its slopes. Each batch
//! certifies its own arc rows once, when its context is built.
//!
//! # Below the widest depth
//!
//! Every arc section is convex. Below its widest depth `T` is concave and increasing with
//! `T(0) >= 0`, and `P` is concave with `P(0) >= 0`, because `dT/dy = 2 cot psi` and
//! `dP/dy = 2 / sin psi` both fall as the normal angle `psi` rises to `pi/2`.
//!
//! - Concavity gives `y T' <= T - T(0) <= T`, and `T` increasing gives `A <= y T`. So
//!   `A T' <= T^2`, and `d r_c / ds = D (3 T/A - T'/T) >= D (3 T/A - T/A) = 2 D T/A >= 2/s >= 2`.
//! - `P` concave gives `y P' <= P`, and `A <= y T` gives `5 T/A >= 5/y`. So
//!   `d r_n / ds = D (5 T/A - 2 P'/P) >= D (5/y - 2/y) = 3/s >= 3`.
//!
//! # From the widest depth to the domain top
//!
//! A cover of pieces in `s` runs from the widest depth, or `2^-30` when the widest depth lies
//! at the floor, to the critical domain top `1 - 2^-53`. Its pieces shrink quadratically
//! toward the crown, where the slopes grow without bound.
//!
//! - Critical: `kappa_c` is the least piece lower bound, capped at 2.
//! - Normal: `s_N` ends the run of positive pieces from the cover start, and `kappa_n` is the
//!   least lower bound on that run, capped at 3. `s_F` starts the run of negative pieces that
//!   reaches the top. The conveyance peak lies in `[s_N, s_F]`.
//! - Refinement splits `[s_N, s_F]` into equal pieces for a fixed number of rounds, moving
//!   `s_N` up and `s_F` down.
//! - `log_peak` encloses `(5 ln a - 2 ln p) / 3`, the log of the unit conveyance, over
//!   `[s_N, s_F]`.
//!
//! A row fails when a critical bound is not positive, when a negative normal piece lies below
//! a positive one or inside `[s_N, s_F]` after the last round, when no negative run reaches
//! the top, or when `s_F - s_N` stays above `2^-20`. A failing row has no certified single
//! conveyance peak below the crown.

#[cfg(test)]
mod certificate_rows;
mod cover;
mod slopes;

pub(crate) use cover::{REFINEMENT_ROUNDS, certify};

use crate::structures::tensor::culvert::Bounds;
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Certificates of `[k]` unit-rise arc rows.
pub(crate) struct ArcCertificates<R: Runtime<DType = DType>> {
    /// `kappa_c`, a lower bound of `d r_c / ds` on `[2^-30, 1 - 2^-53]`.
    pub(crate) critical_slope: Tensor<R>,
    /// `s_N`, the top of the rising limb.
    pub(crate) normal_top: Tensor<R>,
    /// `kappa_n`, a lower bound of `d r_n / ds` on `[2^-30, s_N]`.
    pub(crate) normal_slope: Tensor<R>,
    /// `[s_N, s_F]`, which holds the conveyance peak.
    pub(crate) peak: Bounds<R>,
    /// Enclosure of the log unit conveyance over the peak enclosure.
    pub(crate) log_peak: Bounds<R>,
    /// 1 on rows without a certificate.
    pub(crate) failed: Tensor<R>,
}
