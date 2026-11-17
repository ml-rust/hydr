//! Directional face signal speeds for the channel CFL rule.
//!
//! First-order Godunov waves must not interact inside a cell. The inward waves from both faces
//! must together travel less than one cell length. Each cell frequency is
//! `(inward_left + inward_right) / (2 L)`. A step `dt = cfl / frequency` moves the inward waves
//! `2 cfl` cell lengths. That stays below one for `cfl < 0.5`.
//!
//! A face with a dry cell on one side also takes the dry-front bound of the adjacent wet run.
//! See `cfl_fronts` for why that bound holds.

use super::{
    ChannelError,
    boundary::ScalarBoundaryFrame,
    cfl_fronts::{DryFronts, bound_dry_faces},
    context::Context,
    geometry::TensorGeometry,
};
use numr::dtype::DType;
use numr::ops::{BinaryOps, CompareOps, ConditionalOps, ReduceOps, ScalarOps, ShapeOps, UnaryOps};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Nonnegative signal speeds that enter the two cells beside each face, in m/s.
pub(super) struct FaceSpeeds<R: Runtime<DType = DType>> {
    /// Speed into the cell right of each face: `max(upper, 0)`.
    pub rightward: Tensor<R>,
    /// Speed into the cell left of each face: `max(-lower, 0)`.
    pub leftward: Tensor<R>,
}

/// Face speeds for one CFL limit.
pub(super) struct CflSpeeds<R: Runtime<DType = DType>> {
    /// Speeds at the `count + 1` cell faces.
    pub outer: FaceSpeeds<R>,
    /// Speeds at the face inside each cell that splits it into two half-cells.
    pub internal: Option<FaceSpeeds<R>>,
    /// Dry-front states at the cell faces. `None` when no face is wet on exactly one side.
    pub fronts: Option<DryFronts<R>>,
}

impl<R: Runtime<DType = DType>> Clone for FaceSpeeds<R> {
    fn clone(&self) -> Self {
        Self {
            rightward: self.rightward.clone(),
            leftward: self.leftward.clone(),
        }
    }
}

impl<R: Runtime<DType = DType>> Clone for CflSpeeds<R> {
    fn clone(&self) -> Self {
        Self {
            outer: self.outer.clone(),
            internal: self.internal.clone(),
            fronts: self.fronts.clone(),
        }
    }
}

impl<R: Runtime<DType = DType>> Context<R>
where
    R::Client:
        BinaryOps<R> + CompareOps<R> + ConditionalOps<R> + ReduceOps<R> + ShapeOps<R> + UnaryOps<R>,
{
    /// Split a signed wave-speed interval `[lower, upper]` into inward speeds.
    ///
    /// The interval need not contain zero. A supercritical face gives one direction zero.
    pub fn directional_speeds(
        &self,
        lower: &Tensor<R>,
        upper: &Tensor<R>,
    ) -> Result<FaceSpeeds<R>, ChannelError> {
        let c = &self.client;
        self.finite(lower, "lower face signal speed")?;
        self.finite(upper, "upper face signal speed")?;
        let zero = self.constant(lower, 0.0)?;
        Ok(FaceSpeeds {
            rightward: c.maximum(upper, &zero)?,
            leftward: c.maximum(&c.neg(lower)?, &zero)?,
        })
    }

    /// Place scalar boundary speeds around the numerical face speeds.
    ///
    /// A scalar boundary trace holds one unsigned bound. It serves both directions.
    pub fn assemble_face_speeds(
        &self,
        frame: Option<&ScalarBoundaryFrame<R>>,
        numerical: Option<&FaceSpeeds<R>>,
    ) -> Result<FaceSpeeds<R>, ChannelError> {
        let scalar = frame.map(|f| &f.traces.signal_speed_bound_m_per_s);
        Ok(FaceSpeeds {
            rightward: self.assemble_boundary_faces(
                frame,
                numerical.map(|speeds| &speeds.rightward),
                scalar,
            )?,
            leftward: self.assemble_boundary_faces(
                frame,
                numerical.map(|speeds| &speeds.leftward),
                scalar,
            )?,
        })
    }
}

impl<R: Runtime<DType = DType>> Context<R>
where
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + ShapeOps<R>
        + UnaryOps<R>,
{
    /// Largest step whose inward waves cross at most `2 cfl` of each cell or half-cell.
    ///
    /// A half-cell split by `internal` has length `L / 2`. Its frequency is `sum / L`.
    /// `fronts` raises the inward speed into each dry cell to its wet run's front bound.
    pub fn cfl_limit(
        &self,
        speed: &FaceSpeeds<R>,
        internal: Option<&FaceSpeeds<R>>,
        fronts: Option<&DryFronts<R>>,
    ) -> Result<f64, ChannelError> {
        let c = &self.client;
        let count = self.count();
        let bounded = fronts
            .map(|fronts| bound_dry_faces(c, speed, fronts))
            .transpose()?;
        let speed = bounded.as_ref().unwrap_or(speed);
        let frequency = match &self.geometry {
            TensorGeometry::Rectangular(section) => {
                let face_width = &section.face_width;
                let weighted = |face: &Tensor<R>| -> Result<Tensor<R>, ChannelError> {
                    let weighted = c.mul(face_width, face)?;
                    let moving = c.gt(face, &self.constant(face, 0.0)?)?;
                    self.represented(&moving, &weighted, "width-weighted wave speed")?;
                    Ok(weighted)
                };
                let rightward = weighted(&speed.rightward)?;
                let leftward = weighted(&speed.leftward)?;
                let frequency = c.div(
                    &c.add(
                        &rightward.narrow(0, 0, count)?,
                        &leftward.narrow(0, 1, count)?,
                    )?,
                    &c.mul_scalar(&c.mul(&section.width, &self.length)?, 2.0)?,
                )?;
                self.finite(&frequency, "width-aware CFL frequency")?;
                frequency
            }
            TensorGeometry::Triangular(_) | TensorGeometry::Surveyed(_) => {
                let from_left = speed.rightward.narrow(0, 0, count)?;
                let from_right = speed.leftward.narrow(0, 1, count)?;
                let frequency = if let Some(internal) = internal {
                    let left_half = c.add(&from_left, &internal.leftward)?;
                    let right_half = c.add(&internal.rightward, &from_right)?;
                    c.div(&c.maximum(&left_half, &right_half)?, &self.length)?
                } else {
                    c.div(
                        &c.add(&from_left, &from_right)?,
                        &c.mul_scalar(&self.length, 2.0)?,
                    )?
                };
                self.finite(&frequency, "prismatic CFL frequency")?;
                frequency
            }
        };
        let frequency = c.max(&frequency, &[0], false)?.item::<f64>()?;
        Ok(if frequency > 0.0 {
            self.definition.config.cfl / frequency
        } else {
            f64::INFINITY
        })
    }
}
