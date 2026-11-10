//! Outward hydraulic source-admission bounds on configured runtime tensors.

use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, ReduceOps, ScalarOps, TypeConversionOps, UnaryOps,
};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

use crate::soil::SoilError;
use crate::tensor::soil_storage::TensorSoilContext;

pub(crate) struct Bounds<'a, R: Runtime<DType = DType>> {
    pub context: &'a TensorSoilContext<R>,
}

impl<R> Bounds<'_, R>
where
    R: Runtime<DType = DType>,
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + TypeConversionOps<R>
        + UnaryOps<R>,
{
    pub fn select(&self, active: &Tensor<R>, value: &Tensor<R>) -> Result<Tensor<R>, SoilError> {
        let value = self
            .context
            .client
            .where_cond(active, value, &self.context.zero)?;
        self.context
            .finite(&value, "selected source admission operand")?;
        Ok(value)
    }

    pub fn bound(&self, value: &Tensor<R>) -> Result<Tensor<R>, SoilError> {
        let c = &self.context.client;
        let bound = c.add_scalar(
            &c.mul_scalar(&c.abs(value)?, f64::EPSILON)?,
            f64::from_bits(1),
        )?;
        self.context.finite(&bound, "source rounding bound")?;
        Ok(bound)
    }

    pub fn upper(&self, value: &Tensor<R>) -> Result<Tensor<R>, SoilError> {
        self.context.finite(value, "source bound intermediate")?;
        let result = self.context.client.add(value, &self.bound(value)?)?;
        self.context.finite(&result, "outward source bound")?;
        Ok(result)
    }

    pub fn add(&self, a: &Tensor<R>, b: &Tensor<R>) -> Result<Tensor<R>, SoilError> {
        self.upper(&self.context.client.add(a, b)?)
    }

    pub fn mul(&self, a: &Tensor<R>, b: &Tensor<R>) -> Result<Tensor<R>, SoilError> {
        self.upper(&self.context.client.mul(a, b)?)
    }

    pub fn scale(&self, a: &Tensor<R>, scalar: f64) -> Result<Tensor<R>, SoilError> {
        self.upper(&self.context.client.mul_scalar(a, scalar)?)
    }

    pub fn sum(&self, terms: &[Tensor<R>]) -> Result<Tensor<R>, SoilError> {
        let mut sum = self.context.zero.clone();
        for term in terms {
            sum = self.add(&sum, term)?;
        }
        Ok(sum)
    }
}

impl<R> Bounds<'_, R>
where
    R: Runtime<DType = DType>,
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + TypeConversionOps<R>
        + UnaryOps<R>,
{
    pub fn native(
        &self,
        trace: &super::transition::NativeTrace<R>,
        rate: &Tensor<R>,
        depth: &Tensor<R>,
        dt: f64,
    ) -> Result<Tensor<R>, SoilError> {
        let c = &self.context.client;
        let positive = c.gt(rate, &self.context.zero)?;
        let crossing = c.mul(&trace.crossing, &positive)?;
        let normal = c.mul(
            &crossing,
            &c.ge(&trace.remaining_time_raw, &self.context.zero)?,
        )?;
        let negative = c.mul(
            &crossing,
            &c.lt(&trace.remaining_time_raw, &self.context.zero)?,
        )?;
        let ia = self.select(&normal, &trace.raw_availability_rate)?;
        let normal_bound = self.sum(&[
            self.bound(&self.select(&normal, &trace.difference)?)?,
            self.mul(
                &ia,
                &self.bound(&self.select(&normal, &trace.crossing_time)?)?,
            )?,
            self.mul(
                &ia,
                &self.bound(&self.select(&normal, &trace.remaining_time_raw)?)?,
            )?,
            self.bound(&self.select(&normal, &trace.crossing_product)?)?,
            self.bound(&self.select(&normal, &trace.crossing_limit)?)?,
            self.bound(&self.select(&normal, &trace.crossing_increment)?)?,
            self.bound(&c.mul_scalar(&ia, dt)?)?,
            self.scale(&self.bound(&self.select(&normal, rate)?)?, dt)?,
            self.bound(&self.select(&normal, depth)?)?,
        ])?;
        let negative_bound = self.sum(&[
            self.bound(&self.select(&negative, &trace.supply_endpoint)?)?,
            self.bound(&self.select(&negative, &trace.crossing_increment)?)?,
            self.scale(&self.bound(&self.select(&negative, rate)?)?, dt)?,
            self.bound(&self.select(&negative, depth)?)?,
        ])?;
        let saturated = c.mul(&trace.saturated, &positive)?;
        let saturated_bound = self.add(
            &self.scale(&self.bound(&self.select(&saturated, rate)?)?, dt)?,
            &self.bound(&self.select(&saturated, depth)?)?,
        )?;
        let bound = c.where_cond(&normal, &normal_bound, &self.context.zero)?;
        let bound = c.where_cond(&negative, &negative_bound, &bound)?;
        Ok(c.where_cond(&saturated, &saturated_bound, &bound)?)
    }
}
