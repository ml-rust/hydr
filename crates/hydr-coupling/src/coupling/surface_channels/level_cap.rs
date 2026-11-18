//! The levelling factor of every link from the plan areas of [`super::levelling`].
//!
//! A link whose cap terms all sit at their bounds is stopped: its factor is exactly zero, and it
//! leaves the shared factors and donor totals of every other link unchanged. See
//! `Context::stopped`.

use super::SurfaceChannelError;
use super::context::{Context, Ops};
use super::levelling::Levelling;
use crate::coupling::links::LinkError;
use crate::coupling::links::donors::split;
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, IndexingOps, ScalarOps, ScatterReduceOp, ShapeOps,
    UnaryOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Relative shrink of a binding levelling factor, in units of `f64::EPSILON`.
const LEVEL_MARGIN_ULPS: f64 = 16.0;

/// Link stages of one direction choice: each `[L]`, picked by the sign of the law volume.
struct Sides<R: Runtime<DType = DType>> {
    /// Pre-exchange end stage of the donor.
    donor: Tensor<R>,
    /// Pre-exchange end stage of the receiver.
    receiver: Tensor<R>,
    /// Cutoff stage of the donor: a pump's suction bound.
    donor_cut: Tensor<R>,
    /// Cutoff stage of the receiver: a weir's crest or a gate's sill.
    receiver_cut: Tensor<R>,
    /// `B = max(H_r, z_r)`: the stage the levelling term keeps the donor above.
    floor: Tensor<R>,
}

impl<R: Runtime<DType = DType>> Context<R>
where
    R::Client: Ops<R>,
{
    /// Head allowance `head_absolute_m + head_relative * max(|a|, |b|)` of stage pairs `a` and
    /// `b`, `[L]`.
    pub(super) fn head_allowance(
        &self,
        a: &Tensor<R>,
        b: &Tensor<R>,
    ) -> Result<Tensor<R>, SurfaceChannelError> {
        let c = &self.client;
        let magnitude = c.maximum(&c.abs(a)?, &c.abs(b)?)?;
        Ok(c.add_scalar(
            &c.mul_scalar(&magnitude, self.head_relative)?,
            self.head_absolute_m,
        )?)
    }

    /// One where `gap` lies at or below the head allowance of stages `a` and `b`, `[L]`.
    fn within(
        &self,
        gap: &Tensor<R>,
        a: &Tensor<R>,
        b: &Tensor<R>,
    ) -> Result<Tensor<R>, SurfaceChannelError> {
        Ok(self.client.le(gap, &self.head_allowance(a, b)?)?)
    }

    /// One where a moving link is stopped at its cap bounds, `[L]`.
    ///
    /// Each cap term ends a stage at a bound. A stage within one head allowance of its bound
    /// already stands there, so the term keeps nothing:
    /// - Suction: the donor within the allowance above its suction bound `z_s`.
    /// - Level: the donor within the allowance above `B = max(H_r, z_r)`.
    /// - Below: the receiver within the allowance below its cutoff `z_r`.
    ///
    /// A pump stops on its suction term. A weir or gate stops when both its below and level
    /// terms stop, an orifice when its level term stops. A stopped link otherwise keeps a
    /// volume set by a stage gap that no shorter interval shrinks. That volume can sit below
    /// the resolution of the receiving storage, which then rejects every retry.
    fn stopped(&self, sides: &Sides<R>) -> Result<Tensor<R>, SurfaceChannelError> {
        let c = &self.client;
        let Sides {
            donor,
            receiver,
            donor_cut,
            receiver_cut,
            floor,
        } = sides;
        let level = self.within(&c.sub(donor, floor)?, donor, floor)?;
        let below = self.within(&c.sub(receiver_cut, receiver)?, receiver_cut, receiver)?;
        let suction = self.within(&c.sub(donor, donor_cut)?, donor, donor_cut)?;
        let passive = c.where_cond(&self.has_below, &c.mul(&below, &level)?, &level)?;
        Ok(c.where_cond(&self.pump, &suction, &passive)?)
    }

    /// Levelling factor of every link for law volumes `weir`, then the part of it other links
    /// impose, each `[L]`.
    ///
    /// Unbound links hold exactly one. Stopped links hold exactly zero. The imposed factor is
    /// the shared factor where it lies below the link's own factor, and one elsewhere.
    pub fn level(
        &self,
        weir: &Tensor<R>,
        levelling: &Levelling<R>,
    ) -> Result<(Tensor<R>, Tensor<R>), SurfaceChannelError> {
        let c = &self.client;
        let zero = self.scalar(0.0)?;
        let one = self.scalar(1.0)?;
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
        let cut = pick(&self.cut[1], &self.cut[0])?;
        let floor = c.maximum(&receiver, &cut)?;
        let sides = Sides {
            donor_cut: pick(&self.cut[0], &self.cut[1])?,
            receiver_cut: cut,
            donor,
            receiver,
            floor,
        };
        let stopped = c.mul(&active, &self.stopped(&sides)?)?;
        // Stopped links move no water, so they join no donor or receiver total.
        let moving = c.where_cond(&stopped, &zero, weir)?;
        let (positive, negative) = split(c, &moving, &zero).map_err(LinkError::surface_channel)?;
        let outgoing = self.aggregate(&c.cat(&[&positive, &negative], 0)?)?;
        let incoming = self.aggregate(&c.cat(&[&negative, &positive], 0)?)?;
        let Sides {
            donor,
            receiver,
            receiver_cut: cut,
            floor,
            ..
        } = &sides;
        let given = pick(&at(&outgoing, &self.link_a)?, &at(&outgoing, &self.link_b)?)?;
        let taken = pick(&at(&incoming, &self.link_b)?, &at(&incoming, &self.link_a)?)?;
        let given = c.where_cond(&active, &given, &one)?;
        let taken = c.where_cond(&active, &taken, &one)?;
        let donor_area = pick(&levelling.donor_area[0], &levelling.donor_area[1])?;
        let start_area = pick(&levelling.below_area[0], &levelling.below_area[1])?;
        let receiver_area = pick(&levelling.receiver_area[0], &levelling.receiver_area[1])?;
        let below = c.div(&c.mul(&c.sub(cut, receiver)?, &start_area)?, &taken)?;
        let response = c.add(
            &c.div(&given, &donor_area)?,
            &c.div(&taken, &receiver_area)?,
        )?;
        let level = c.div(&c.sub(donor, floor)?, &response)?;
        let passive = c.where_cond(&self.has_below, &c.maximum(&below, &level)?, &level)?;
        let suction = c.div(
            &c.mul(&c.sub(donor, &self.suction_stop)?, &self.suction_area)?,
            &given,
        )?;
        let theta = c.where_cond(&self.pump, &suction, &passive)?;
        let theta = c.minimum(&theta, &one)?;
        let theta = c.where_cond(&c.gt(&theta, &zero)?, &theta, &zero)?;
        let shrunk = c.mul_scalar(&theta, 1.0 - LEVEL_MARGIN_ULPS * f64::EPSILON)?;
        let theta = c.where_cond(&c.lt(&theta, &one)?, &shrunk, &theta)?;
        let theta = c.where_cond(&active, &theta, &one)?;
        // A stopped link binds no other link.
        let theta = c.where_cond(&stopped, &one, &theta)?;
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
        let shared = c.where_cond(&active, &shared, &one)?;
        let imposed = c.where_cond(&c.lt(&shared, &theta)?, &shared, &one)?;
        let imposed = c.where_cond(&stopped, &one, &imposed)?;
        Ok((c.where_cond(&stopped, &zero, &shared)?, imposed))
    }
}

#[cfg(test)]
mod tests {
    use super::super::SurfaceChannelBatch;
    use super::super::pools::scenario::{
        PoolForcing, advance, build, check, declaration, link, long_weir, pond, pool, stages,
        volumes, with_nodes,
    };
    use super::super::structures::scenario::{
        HEAD_ALLOWANCE, Tanks, TestResult, client, pump_law, receipt,
    };
    use crate::structures::{DirectionPolicy, StructureOperatingPoint};

    /// Requested duration of every interval, in s.
    const DT: f64 = 0.5;

    /// Suction bound of every pump, in m.
    const SUCTION_M: f64 = 0.1;

    /// Pump from pool `sump` into pool `tank`, running at reference speed.
    fn pump() -> TestResult<super::super::structures::SurfaceChannelStructure> {
        Ok(link(
            "pump",
            pump_law(SUCTION_M, None)?,
            DirectionPolicy::AtoBOnly,
            StructureOperatingPoint::PumpRunning { speed_ratio: 1.0 },
            (pool("sump"), 0.0),
            (pool("tank"), 0.0),
        ))
    }

    /// A running pump drains a 1 m² sump at 0.5 m into a tank of `tank_m2` at 0.5 m.
    ///
    /// Frozen criteria, fixed before the first run:
    /// - Every one of 40 intervals accepts and passes its mass bound, every pairing, and the
    ///   composite balance.
    /// - No pool volume is negative, and the sump stage never falls below `z_s`.
    /// - The suction cap binds in some interval.
    /// - Every later interval moves `0.0` bit for bit with level factor `0.0`.
    /// - The final sump stage lies in `[z_s, z_s + allowance(z_s)]`.
    fn drains_to_its_suction_band(tank_m2: f64) -> TestResult {
        let pools = declaration(
            vec![
                pond("sump", 0.0, 1.0, 2.0)?,
                pond("tank", 0.0, tank_m2, 2.0)?,
            ],
            false,
            Vec::new(),
        );
        let tanks = Tanks::at(0.0, 0.0);
        let mut batch = build(tanks, pools, &[0.5, 0.5 * tank_m2], vec![pump()?])?;
        let none = PoolForcing::none();
        let mut capped = false;
        for index in 0..40 {
            batch = advance(&batch, DT, &none)?;
            check(&batch, &[0.0, 0.0])?;
            assert!(volumes(&batch)?.iter().all(|&v| v >= 0.0), "{index}");
            let sump = stages(&batch)?[0];
            assert!(sump >= SUCTION_M, "interval {index}: sump stage={sump:e}");
            let link = receipt(&batch, "pump")?;
            if capped {
                assert_eq!(link.signed.to_bits(), 0.0f64.to_bits(), "{index}: {link:?}");
                assert_eq!(link.level.to_bits(), 0.0f64.to_bits(), "{index}: {link:?}");
            }
            capped |= link.level < 1.0;
        }
        assert!(capped, "the suction cap never bound");
        let sump = stages(&batch)?[0];
        let band = HEAD_ALLOWANCE + HEAD_ALLOWANCE * SUCTION_M;
        assert!(sump <= SUCTION_M + band, "sump stage={sump:e}");
        Ok(())
    }

    #[test]
    fn a_pump_into_a_wide_tank_rests_in_its_suction_band() -> TestResult {
        drains_to_its_suction_band(8.0)
    }

    #[test]
    fn a_pump_into_an_equal_tank_rests_in_its_suction_band() -> TestResult {
        drains_to_its_suction_band(1.0)
    }

    /// A 1 m² sump at 0.5 m, an 8 m² tank at 0.5 m, and an 8 m² upper pool at 1.5 m.
    fn shared_tank() -> TestResult<super::super::pools::SurfaceChannelPools> {
        Ok(declaration(
            vec![
                pond("sump", 0.0, 1.0, 2.0)?,
                pond("tank", 0.0, 8.0, 2.0)?,
                pond("upper", 0.0, 8.0, 2.0)?,
            ],
            false,
            Vec::new(),
        ))
    }

    /// Weir from pool `upper` into pool `tank`, crest 1 m, 1 m long.
    fn weir() -> TestResult<super::super::structures::SurfaceChannelStructure> {
        Ok(link(
            "weir",
            long_weir(1.0, 1.0)?,
            DirectionPolicy::Bidirectional,
            StructureOperatingPoint::Passive,
            (pool("upper"), 0.0),
            (pool("tank"), 0.0),
        ))
    }

    /// The pump of [`a_stopped_pump_binds_no_link_at_its_receiver`] with `retry_limit = 0`: its
    /// suction cap binds while the weir shares its tank, so the cap's factor reaches the weir.
    ///
    /// Frozen criteria, fixed before the first run:
    /// - Every one of 16 intervals of 0.5 s converges on its first attempt and ends at its
    ///   requested end.
    /// - Every interval passes its mass bound, every pairing, and the composite balance.
    /// - In some interval the pump's cap binds: `0 < level < 1`, and the weir takes the same
    ///   factor.
    #[test]
    fn a_shared_cap_factor_converges_without_a_retry() -> TestResult {
        let tanks = Tanks::at(0.0, 0.0);
        let (mut setup, initial) = with_nodes(
            tanks.parts()?,
            shared_tank()?,
            &[0.5, 4.0, 12.0],
            vec![pump()?, weir()?],
        );
        setup.config.retry_limit = 0;
        let mut batch = SurfaceChannelBatch::new(client(), setup, initial)?;
        let none = PoolForcing::none();
        let mut shared = false;
        for index in 0..16 {
            let requested = batch.time() + DT;
            batch = advance(&batch, DT, &none)?;
            check(&batch, &[0.0, 0.0, 0.0])?;
            let step = batch.last_step()?.ok_or("an accepted interval")?;
            assert_eq!(step.retries, 0, "{index}");
            assert_eq!(step.end_s, requested, "{index}");
            let pump = receipt(&batch, "pump")?;
            if 0.0 < pump.level && pump.level < 1.0 {
                assert_eq!(receipt(&batch, "weir")?.level, pump.level, "{index}");
                shared = true;
            }
        }
        assert!(shared, "the pump's cap never bound");
        Ok(())
    }

    /// A weir from a high pool fills the pump's tank while the pump drains its sump to suction.
    ///
    /// Frozen criteria, fixed before the first run:
    /// - Every one of 16 intervals accepts and passes its mass bound, every pairing, and the
    ///   composite balance.
    /// - The pump stops in some interval: it moves `0.0` bit for bit.
    /// - In every interval where the pump stops, the weir keeps level factor `1.0` and moves
    ///   water into the tank.
    #[test]
    fn a_stopped_pump_binds_no_link_at_its_receiver() -> TestResult {
        let pools = shared_tank()?;
        let tanks = Tanks::at(0.0, 0.0);
        let mut batch = build(tanks, pools, &[0.5, 4.0, 12.0], vec![pump()?, weir()?])?;
        let none = PoolForcing::none();
        let mut stopped = false;
        for index in 0..16 {
            batch = advance(&batch, DT, &none)?;
            check(&batch, &[0.0, 0.0, 0.0])?;
            let pump = receipt(&batch, "pump")?;
            if pump.signed.to_bits() == 0.0f64.to_bits() {
                stopped = true;
                let weir = receipt(&batch, "weir")?;
                assert_eq!(weir.level, 1.0, "{index}: {weir:?}");
                assert!(weir.signed > 0.0, "{index}: {weir:?}");
            }
        }
        assert!(stopped, "the pump never stopped");
        Ok(())
    }
}
