use super::{ChannelError, context::Context, geometry::TensorGeometry};
use crate::numerics::subcatchment::bounds::rounding_with_client;
use numr::dtype::DType;
use numr::ops::{BinaryOps, CompareOps, ConditionalOps, ReduceOps, ScalarOps, TwoSumOps, UnaryOps};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

pub(super) struct IntervalControl {
    pub start: f64,
    pub requested_end: f64,
    pub minimum: f64,
    pub exact: bool,
}

pub(super) struct DonorReservation<R: Runtime<DType = DType>> {
    pub limit_s: f64,
    pub admitted: bool,
    pub remaining_lower: Tensor<R>,
}

impl IntervalControl {
    pub fn shorten(
        &self,
        current_dt: f64,
        desired_dt: f64,
        retries: usize,
        reason: String,
    ) -> Result<f64, ChannelError> {
        let suggested_end = self.start + desired_dt;
        let dt = suggested_end - self.start;
        if !dt.is_finite() || dt <= 0.0 || dt >= current_dt || dt < self.minimum {
            return Err(ChannelError::Unresolved {
                dt_s: dt,
                retries,
                reason,
            });
        }
        if self.exact {
            return Err(ChannelError::RecomputeInterval {
                start_s: self.start,
                requested_end_s: self.requested_end,
                suggested_end_s: suggested_end,
                reason,
            });
        }
        Ok(dt)
    }
}

impl<R: Runtime<DType = DType>> Context<R>
where
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + UnaryOps<R>,
{
    pub fn cfl_limit(
        &self,
        speed: &Tensor<R>,
        internal_speed: Option<&Tensor<R>>,
    ) -> Result<f64, ChannelError> {
        let c = &self.client;
        let count = self.count();
        let frequency = match &self.geometry {
            TensorGeometry::Rectangular(section) => {
                let width = &section.width;
                let face_width = &section.face_width;
                let weighted_speed = c.mul(face_width, speed)?;
                let moving_signal = c.gt(speed, &self.constant(speed, 0.0)?)?;
                self.represented(&moving_signal, &weighted_speed, "width-weighted wave speed")?;
                let frequency = c.div(
                    &c.add(
                        &weighted_speed.narrow(0, 0, count)?,
                        &weighted_speed.narrow(0, 1, count)?,
                    )?,
                    &c.mul(width, &self.length)?,
                )?;
                self.finite(&frequency, "width-aware CFL frequency")?;
                frequency
            }
            TensorGeometry::Triangular(_) => {
                let frequency = if let Some(internal) = internal_speed {
                    let left = c.add(&speed.narrow(0, 0, count)?, internal)?;
                    let right = c.add(internal, &speed.narrow(0, 1, count)?)?;
                    c.div(
                        &c.mul_scalar(&c.maximum(&left, &right)?, 2.0)?,
                        &self.length,
                    )?
                } else {
                    c.div(
                        &c.add(&speed.narrow(0, 0, count)?, &speed.narrow(0, 1, count)?)?,
                        &self.length,
                    )?
                };
                self.finite(&frequency, "triangular CFL frequency")?;
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

    pub fn outgoing_rate(&self, face_rate: &Tensor<R>) -> Result<Tensor<R>, ChannelError> {
        let c = &self.client;
        let zero = self.constant(&self.length, 0.0)?;
        let outgoing = c.add(
            &c.maximum(&face_rate.narrow(0, 1, self.count())?, &zero)?,
            &c.maximum(&c.neg(&face_rate.narrow(0, 0, self.count())?)?, &zero)?,
        )?;
        self.finite(&outgoing, "all-outgoing water rate")?;
        Ok(outgoing)
    }

    pub fn face_depletion_limit(
        &self,
        volume: &Tensor<R>,
        outgoing: &Tensor<R>,
    ) -> Result<f64, ChannelError> {
        let c = &self.client;
        let zero = self.constant(volume, 0.0)?;
        let one = self.constant(volume, 1.0)?;
        let exporting = c.gt(outgoing, &zero)?;
        let safe_outgoing = c.where_cond(&exporting, outgoing, &one)?;
        let bounds = c.div(volume, &safe_outgoing)?;
        let bounds = c.where_cond(&exporting, &bounds, &self.constant(&bounds, f64::INFINITY)?)?;
        Ok(c.min(&bounds, &[0], false)?.item::<f64>()?)
    }
}

impl<R: Runtime<DType = DType>> Context<R>
where
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + UnaryOps<R>,
{
    pub fn combined_depletion_limit(
        &self,
        volume: &Tensor<R>,
        outgoing: &Tensor<R>,
        withdrawal: &Tensor<R>,
        withdrawal_bound: &Tensor<R>,
        dt: f64,
    ) -> Result<DonorReservation<R>, ChannelError> {
        let c = &self.client;
        let zero = self.constant(volume, 0.0)?;
        let withdrawal_upper = self.bound_add(withdrawal, withdrawal_bound)?;
        if self.all(&c.eq(&withdrawal_upper, &zero)?)? {
            let limit = self.face_depletion_limit(volume, outgoing)?;
            return Ok(DonorReservation {
                limit_s: limit,
                admitted: dt <= limit,
                remaining_lower: volume.clone(),
            });
        }
        let (remaining, low) = c.two_sum(volume, &c.neg(&withdrawal_upper)?)?;
        self.finite(&remaining, "original donor water after reserved withdrawal")?;
        self.finite(&low, "reserved withdrawal subtraction low")?;
        // A negative low means the rounded subtraction invents available donor water.
        let lower = c.sub(&remaining, &rounding_with_client::<R>(c, &remaining)?)?;
        let remaining_lower = c.maximum(
            &c.where_cond(&c.lt(&low, &zero)?, &lower, &remaining)?,
            &zero,
        )?;
        let face_limit = self.face_depletion_limit(&remaining_lower, outgoing)?;
        let withdrawal_admitted = self.all(&c.le(&withdrawal_upper, volume)?)?;
        let admitted = withdrawal_admitted && dt <= face_limit;

        // This estimate proposes a fresh span. It never modifies the physical input operands.
        let face_outgoing = c.mul_scalar(outgoing, dt)?;
        let demand = self.bound_add(&face_outgoing, &withdrawal_upper)?;
        let active = c.gt(&demand, &zero)?;
        let safe = c.where_cond(&active, &demand, &self.constant(volume, 1.0)?)?;
        let ratio = c.div(volume, &safe)?;
        let limit = c.mul_scalar(&ratio, dt)?;
        let limit = c.where_cond(&active, &limit, &self.constant(volume, f64::INFINITY)?)?;
        let limit = c.min(&limit, &[0], false)?.item::<f64>()?;
        Ok(DonorReservation {
            limit_s: if admitted {
                limit.max(dt).min(face_limit)
            } else {
                limit
            },
            admitted,
            remaining_lower,
        })
    }

    pub fn actual_outgoing_admitted(
        &self,
        remaining_lower: &Tensor<R>,
        face_volume: &Tensor<R>,
    ) -> Result<bool, ChannelError> {
        let c = &self.client;
        let zero = self.constant(remaining_lower, 0.0)?;
        let right_outgoing = c.maximum(&face_volume.narrow(0, 1, self.count())?, &zero)?;
        let left_outgoing = c.maximum(&c.neg(&face_volume.narrow(0, 0, self.count())?)?, &zero)?;
        let (sum, low) = c.two_sum(&right_outgoing, &left_outgoing)?;
        self.finite(&sum, "actual outgoing shared face volumes")?;
        self.finite(&low, "actual outgoing shared face volume low")?;
        let outgoing_upper = self.bound_add(&sum, &c.maximum(&low, &zero)?)?;
        self.all(&c.le(&outgoing_upper, remaining_lower)?)
    }
}
