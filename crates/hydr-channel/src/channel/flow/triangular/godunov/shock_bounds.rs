use super::super::{
    envelope::{lower, upper},
    wave::SideBounds,
};
use crate::channel::flow::{ChannelError, context::Context, geometry::TriangleGeometry};
use numr::{
    dtype::DType,
    ops::{
        BinaryOps, CompareOps, ConditionalOps, CorrectlyRoundedOps, ReduceOps, ScalarOps, ShapeOps,
        UnaryOps,
    },
    runtime::Runtime,
    tensor::Tensor,
};
use solvr::optimize::roots::BatchedBisectionAlgorithms;

impl<R: Runtime<DType = DType>> Context<R>
where
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + CorrectlyRoundedOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + ShapeOps<R>
        + UnaryOps<R>
        + BatchedBisectionAlgorithms<R>,
{
    pub(super) fn star_shock_bounds(
        &self,
        geometry: &TriangleGeometry<R>,
        side: &SideBounds<R>,
        lo: &Tensor<R>,
        hi: &Tensor<R>,
        right: bool,
    ) -> Result<(Tensor<R>, Tensor<R>), ChannelError> {
        let c = &self.client;
        let zero = self.constant(lo, 0.0)?;
        let magnitude =
            |h: &Tensor<R>, hk: &Tensor<R>, up: bool| -> Result<Tensor<R>, ChannelError> {
                let round = |x: &Tensor<R>| {
                    if up {
                        upper::<R>(c, x)
                    } else {
                        lower::<R>(c, x)
                    }
                };
                let opposite = |x: &Tensor<R>| {
                    if up {
                        lower::<R>(c, x)
                    } else {
                        upper::<R>(c, x)
                    }
                };
                let h = c.maximum(h, hk)?;
                let rlo = c.maximum(&lower::<R>(c, &c.div(hk, &h)?)?, &zero)?;
                let rhi = upper::<R>(c, &c.div(hk, &h)?)?;
                let r = if up { &rhi } else { &rlo };
                let numerator =
                    round(&c.add_scalar(&round(&c.add(r, &round(&c.mul(r, r)?)?)?)?, 1.0)?)?;
                let denominator = opposite(&c.add_scalar(if up { &rlo } else { &rhi }, 1.0)?)?;
                let shape = c.maximum(&round(&c.div(&numerator, &denominator)?)?, &zero)?;
                let three = self.constant(lo, 3.0)?;
                let scale = round(&c.div(
                    &round(&c.mul(
                        if up {
                            &geometry.root_gravity_upper
                        } else {
                            &geometry.root_gravity_lower
                        },
                        &c.maximum(&round(&c.sqrt(&h)?)?, &zero)?,
                    )?)?,
                    &opposite(&c.sqrt(&three)?)?,
                )?)?;
                Ok(c.maximum(
                    &round(&c.mul(
                        &round(&c.mul(&scale, &round(&c.div(&h, hk)?)?)?)?,
                        &c.maximum(&round(&c.sqrt(&shape)?)?, &zero)?,
                    )?)?,
                    &zero,
                )?)
            };
        // Every shock magnitude exceeds its reference sound speed, including zero-strength limits.
        let ml = c.where_cond(
            &c.gt(lo, &side.upper_depth)?,
            &magnitude(lo, &side.upper_depth, false)?,
            &side.lower_sound,
        )?;
        let mh = magnitude(hi, &side.lower_depth, true)?;
        let (l, h) = if right {
            (
                lower::<R>(c, &c.add(&side.lower_velocity, &ml)?)?,
                upper::<R>(c, &c.add(&side.upper_velocity, &mh)?)?,
            )
        } else {
            (
                lower::<R>(c, &c.sub(&side.lower_velocity, &mh)?)?,
                upper::<R>(c, &c.sub(&side.upper_velocity, &ml)?)?,
            )
        };
        self.finite(&l, "triangular shock lower speed")?;
        self.finite(&h, "triangular shock upper speed")?;
        Ok((l, h))
    }
}

#[cfg(test)]
mod tests {
    use crate::channel::flow::{
        ChannelConfig, ChannelInitialState, ChannelReachDefinition,
        context::Context,
        geometry::{ChannelGeometry, TensorGeometry, TriangularChannelCell},
        triangular::wave::SideBounds,
    };
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};

    #[test]
    fn overlapping_star_and_reference_depths_enclose_zero_strength_shock_speeds()
    -> Result<(), Box<dyn std::error::Error>> {
        let client = CpuClient::new(CpuDevice::new())
            .with_parallelism(ParallelismConfig::new(Some(1), None));
        let context = Context::<CpuRuntime>::new(
            client,
            ChannelReachDefinition {
                vertical_datum: "test datum".into(),
                provenance: "triangular shock-speed enclosure".into(),
                geometry: ChannelGeometry::PrismaticTriangular {
                    left_bank_slope_h_per_v: 1.0,
                    right_bank_slope_h_per_v: 1.0,
                    bed_elevation_m: 0.0,
                    cells: vec![TriangularChannelCell {
                        id: "cell".into(),
                        length_m: 1.0,
                        maximum_depth_m: 2.0,
                        manning_n: 0.0,
                    }],
                },
                lateral_bindings: vec![],
                config: ChannelConfig {
                    gravity: 8.0,
                    ..ChannelConfig::default()
                },
            },
            ChannelInitialState {
                time_s: 0.0,
                volume_m3: vec![1.0],
                longitudinal_momentum_m4_per_s: vec![0.0],
            },
        )?;
        let TensorGeometry::Triangular(geometry) = &context.geometry else {
            return Err("triangular geometry required".into());
        };
        let side = SideBounds {
            lower_signal: context.tensor(&[-8.0 * 2.0_f64.sqrt()])?,
            upper_signal: context.tensor(&[8.0 * 2.0_f64.sqrt()])?,
            lower_depth: context.tensor(&[1.0])?,
            upper_depth: context.tensor(&[2.0])?,
            lower_sound: context.tensor(&[2.0])?,
            upper_sound: context.tensor(&[2.0 * 2.0_f64.sqrt()])?,
            lower_velocity: context.tensor(&[0.0])?,
            upper_velocity: context.tensor(&[0.0])?,
        };
        let lower_depth = context.tensor(&[1.0])?;
        let upper_depth = context.tensor(&[2.0])?;
        let (left_lower, left_upper) =
            context.star_shock_bounds(geometry, &side, &lower_depth, &upper_depth, false)?;
        let (right_lower, right_upper) =
            context.star_shock_bounds(geometry, &side, &lower_depth, &upper_depth, true)?;
        let left_lower = left_lower.item::<f64>()?;
        let left_upper = left_upper.item::<f64>()?;
        let right_lower = right_lower.item::<f64>()?;
        let right_upper = right_upper.item::<f64>()?;
        // Rankine-Hugoniot uses A=h² and p=8h³/3. Equal depths use the sound-speed limit.
        for reference_depth_m in [1.0_f64, 1.25, 1.5, 2.0] {
            for star_depth_m in [1.0_f64, 1.25, 1.5, 2.0] {
                if star_depth_m < reference_depth_m {
                    continue;
                }
                let speed_m_per_s = if star_depth_m == reference_depth_m {
                    (4.0 * reference_depth_m).sqrt()
                } else {
                    let reference_area_m2 = reference_depth_m.powi(2);
                    let star_area_m2 = star_depth_m.powi(2);
                    let pressure_jump_m4_per_s2 =
                        (8.0 / 3.0) * (star_depth_m.powi(3) - reference_depth_m.powi(3));
                    (pressure_jump_m4_per_s2 * star_area_m2
                        / (reference_area_m2 * (star_area_m2 - reference_area_m2)))
                        .sqrt()
                };
                assert!(right_lower <= speed_m_per_s && speed_m_per_s <= right_upper);
                assert!(left_lower <= -speed_m_per_s && -speed_m_per_s <= left_upper);
            }
        }
        Ok(())
    }
}
