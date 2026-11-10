use super::{ChannelError, context::Context};
use numr::dtype::DType;
use numr::ops::{BinaryOps, CompareOps, ConditionalOps, IndexingOps, ReduceOps, UnaryOps};
use numr::runtime::{Device, Runtime};
use numr::tensor::Tensor;

pub(super) struct Fields<R: Runtime<DType = DType>> {
    pub area: Tensor<R>,
    pub depth: Tensor<R>,
    pub velocity: Tensor<R>,
    pub discharge: Tensor<R>,
}

impl<R: Runtime<DType = DType>> Context<R>
where
    R::Client: BinaryOps<R> + CompareOps<R> + ConditionalOps<R> + ReduceOps<R> + UnaryOps<R>,
{
    pub fn finite(&self, value: &Tensor<R>, name: &str) -> Result<(), ChannelError> {
        let mask = self
            .client
            .le(&self.client.abs(value)?, &self.constant(value, f64::MAX)?)?;
        if !self.all(&mask)? {
            let index = self.first_rejected(&mask)?;
            let input = value.narrow(0, index, 1)?.item::<f64>()?;
            return Err(ChannelError::numerical(format!(
                "{name}, lane={index}, value={input}: supply finite values and rescale channel inputs"
            )));
        }
        Ok(())
    }

    pub fn all(&self, mask: &Tensor<R>) -> Result<bool, ChannelError> {
        Ok(self.client.min(mask, &[0], false)?.item::<f64>()? == 1.0)
    }

    fn first_rejected(&self, admitted: &Tensor<R>) -> Result<usize, ChannelError> {
        let rejected = self.client.eq(admitted, &self.constant(admitted, 0.0)?)?;
        let selected = self.client.argmax(&rejected, 0, false)?.item::<i64>()?;
        usize::try_from(selected)
            .ok()
            .filter(|index| *index < admitted.numel())
            .ok_or_else(|| ChannelError::numerical(format!(
                "rejected lane={selected}, count={}: inspect the selected runtime's reduction result",
                admitted.numel()
            )))
    }

    pub fn represented(
        &self,
        required: &Tensor<R>,
        value: &Tensor<R>,
        name: &str,
    ) -> Result<(), ChannelError> {
        self.finite(value, name)?;
        let zero = self.constant(value, 0.0)?;
        let admitted = self.client.maximum(
            &self.client.eq(required, &zero)?,
            &self.client.gt(&self.client.abs(value)?, &zero)?,
        )?;
        if !self.all(&admitted)? {
            let index = self.first_rejected(&admitted)?;
            return Err(ChannelError::numerical(format!(
                "{name}, lane={index}, value=0 underflows from a nonzero physical operand: rescale inputs or use a representable interval"
            )));
        }
        Ok(())
    }

    pub fn total(&self, value: &Tensor<R>) -> Result<f64, ChannelError> {
        let total = self.client.sum(value, &[0], false)?.item::<f64>()?;
        if !total.is_finite() {
            return Err(ChannelError::numerical(format!(
                "channel total={total}: rescale water or momentum"
            )));
        }
        Ok(total)
    }

    pub fn fields(
        &self,
        volume: &Tensor<R>,
        momentum: &Tensor<R>,
    ) -> Result<Fields<R>, ChannelError> {
        for (value, name) in [
            (volume, "cell water volume"),
            (momentum, "cell longitudinal momentum"),
        ] {
            if value.shape() != [self.count()]
                || value.dtype() != DType::F64
                || !value.device().is_same(self.length.device())
            {
                return Err(ChannelError::input(
                    "channel state requires equal F64 cell arrays on the configured device: rebuild state with this client",
                ));
            }
            self.finite(value, name)?;
        }
        let c = &self.client;
        let zero = self.constant(volume, 0.0)?;
        let one = self.constant(volume, 1.0)?;
        let wet = c.gt(volume, &zero)?;
        let area = c.div(volume, &self.length)?;
        let depth = c.div(&area, &self.width)?;
        let safe_volume = c.where_cond(&wet, volume, &one)?;
        let velocity = c.div(momentum, &safe_volume)?;
        let discharge = c.div(momentum, &self.length)?;
        let mut admitted = c.ge(volume, &zero)?;
        admitted = c.mul(&admitted, &c.maximum(&wet, &c.eq(momentum, &zero)?)?)?;
        for value in [&velocity, &discharge] {
            admitted = c.mul(
                &admitted,
                &c.maximum(&c.eq(momentum, &zero)?, &c.gt(&c.abs(value)?, &zero)?)?,
            )?;
        }
        for value in [&area, &depth] {
            admitted = c.mul(
                &admitted,
                &c.maximum(&c.eq(volume, &zero)?, &c.gt(value, &zero)?)?,
            )?;
        }
        admitted = c.mul(&admitted, &c.le(&depth, &self.cap)?)?;
        if !self.all(&admitted)? {
            let index = self.first_rejected(&admitted)?;
            let read = |value: &Tensor<R>| value.narrow(0, index, 1)?.item::<f64>();
            return Err(ChannelError::numerical(format!(
                "cell={index}, volume_m3={}, momentum_m4_per_s={}, area_m2={}, depth_m={}, velocity_m_per_s={}, discharge_m3_per_s={}, maximum_depth_m={}: require nonnegative represented water, zero dry momentum, resolved motion, and depth within the bank limit; reduce the interval or correct inputs",
                read(volume)?, read(momentum)?, read(&area)?, read(&depth)?, read(&velocity)?, read(&discharge)?, read(&self.cap)?
            )).at(index));
        }
        for (value, name) in [
            (&area, "channel area"),
            (&depth, "channel depth"),
            (&velocity, "channel velocity"),
            (&discharge, "channel discharge"),
            (&c.add(&self.bed, &depth)?, "channel stage"),
        ] {
            self.finite(value, name)?;
        }
        self.total(volume)?;
        Ok(Fields {
            area,
            depth,
            velocity,
            discharge,
        })
    }
}
