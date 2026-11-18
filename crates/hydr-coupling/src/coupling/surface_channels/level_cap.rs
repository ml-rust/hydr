//! The levelling factor of every link from the plan areas of [`super::levelling`].

use super::SurfaceChannelError;
use super::context::{Context, Ops};
use super::levelling::Levelling;
use crate::coupling::links::LinkError;
use crate::coupling::links::donors::split;
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, IndexingOps, ScalarOps, ScatterReduceOp, ShapeOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Relative shrink of a binding levelling factor, in units of `f64::EPSILON`.
const LEVEL_MARGIN_ULPS: f64 = 16.0;

impl<R: Runtime<DType = DType>> Context<R>
where
    R::Client: Ops<R>,
{
    /// Levelling factor of every link for law volumes `weir`, `[L]`. Unbound links hold exactly
    /// one.
    pub fn level(
        &self,
        weir: &Tensor<R>,
        levelling: &Levelling<R>,
    ) -> Result<Tensor<R>, SurfaceChannelError> {
        let c = &self.client;
        let zero = self.scalar(0.0)?;
        let one = self.scalar(1.0)?;
        let (positive, negative) = split(c, weir, &zero).map_err(LinkError::surface_channel)?;
        let outgoing = self.aggregate(&c.cat(&[&positive, &negative], 0)?)?;
        let incoming = self.aggregate(&c.cat(&[&negative, &positive], 0)?)?;
        let at = |values: &Tensor<R>, index: &Tensor<R>| c.index_select(values, 0, index);
        let forward = c.gt(weir, &zero)?;
        let backward = c.lt(weir, &zero)?;
        let active = c.maximum(&forward, &backward)?;
        let pick = |surface_donor: &Tensor<R>, channel_donor: &Tensor<R>| {
            c.where_cond(&forward, surface_donor, channel_donor)
        };
        let stage = &levelling.stage;
        let donor = pick(&at(stage, &self.link_a)?, &at(stage, &self.link_b)?)?;
        let receiver = pick(&at(stage, &self.link_b)?, &at(stage, &self.link_a)?)?;
        let given = pick(&at(&outgoing, &self.link_a)?, &at(&outgoing, &self.link_b)?)?;
        let taken = pick(&at(&incoming, &self.link_b)?, &at(&incoming, &self.link_a)?)?;
        let given = c.where_cond(&active, &given, &one)?;
        let taken = c.where_cond(&active, &taken, &one)?;
        let cut = pick(&self.cut[1], &self.cut[0])?;
        let donor_area = pick(&levelling.donor_area[0], &levelling.donor_area[1])?;
        let start_area = pick(&levelling.below_area[0], &levelling.below_area[1])?;
        let receiver_area = pick(&levelling.receiver_area[0], &levelling.receiver_area[1])?;
        let below = c.div(&c.mul(&c.sub(&cut, &receiver)?, &start_area)?, &taken)?;
        let floor = c.maximum(&receiver, &cut)?;
        let response = c.add(
            &c.div(&given, &donor_area)?,
            &c.div(&taken, &receiver_area)?,
        )?;
        let level = c.div(&c.sub(&donor, &floor)?, &response)?;
        let passive = c.where_cond(&self.has_below, &c.maximum(&below, &level)?, &level)?;
        let suction = c.div(
            &c.mul(&c.sub(&donor, &self.suction_stop)?, &self.suction_area)?,
            &given,
        )?;
        let theta = c.where_cond(&self.pump, &suction, &passive)?;
        let theta = c.minimum(&theta, &one)?;
        let theta = c.where_cond(&c.gt(&theta, &zero)?, &theta, &zero)?;
        let shrunk = c.mul_scalar(&theta, 1.0 - LEVEL_MARGIN_ULPS * f64::EPSILON)?;
        let theta = c.where_cond(&c.lt(&theta, &one)?, &shrunk, &theta)?;
        let theta = c.where_cond(&active, &theta, &one)?;
        // Each group keeps the smallest factor among links that give from it or take into it.
        let groups = Tensor::<R>::ones(&[self.group_count()], DType::F64, c.device())?;
        let from_surface = c.where_cond(&forward, &theta, &one)?;
        let from_channel = c.where_cond(&backward, &theta, &one)?;
        let least = |index: &Tensor<R>, values: &Tensor<R>| {
            c.scatter_reduce(&groups, 0, index, values, ScatterReduceOp::Min, true)
        };
        let gives = c.minimum(
            &least(&self.link_a, &from_surface)?,
            &least(&self.link_b, &from_channel)?,
        )?;
        let takes = c.minimum(
            &least(&self.link_b, &from_surface)?,
            &least(&self.link_a, &from_channel)?,
        )?;
        let shared = pick(
            &c.minimum(&at(&gives, &self.link_a)?, &at(&takes, &self.link_b)?)?,
            &c.minimum(&at(&gives, &self.link_b)?, &at(&takes, &self.link_a)?)?,
        )?;
        Ok(c.where_cond(&active, &shared, &one)?)
    }
}
