use super::values;
use crate::channel::flow::{ChannelBoundary, ChannelForcingInterval};

pub(in super::super) fn forcing(a: &ChannelForcingInterval, b: &ChannelForcingInterval) -> bool {
    values(&[a.start_s, a.end_s], &[b.start_s, b.end_s])
        && boundary(&a.left, &b.left)
        && boundary(&a.right, &b.right)
        && a.lateral_inputs.len() == b.lateral_inputs.len()
        && a.lateral_inputs
            .iter()
            .zip(&b.lateral_inputs)
            .all(|(a, b)| a.same_bits(b))
}

fn boundary(a: &ChannelBoundary, b: &ChannelBoundary) -> bool {
    match (a, b) {
        (ChannelBoundary::Wall, ChannelBoundary::Wall) => true,
        (ChannelBoundary::Stage { elevation_m: a }, ChannelBoundary::Stage { elevation_m: b }) => {
            a.to_bits() == b.to_bits()
        }
        (
            ChannelBoundary::QuiescentReservoir {
                elevation_m: ah,
                entrance_loss_coefficient: ak,
            },
            ChannelBoundary::QuiescentReservoir {
                elevation_m: bh,
                entrance_loss_coefficient: bk,
            },
        ) => values(&[*ah, *ak], &[*bh, *bk]),
        (
            ChannelBoundary::Discharge {
                discharge_m3_per_s: a,
            },
            ChannelBoundary::Discharge {
                discharge_m3_per_s: b,
            },
        ) => a.to_bits() == b.to_bits(),
        (
            ChannelBoundary::Exterior {
                depth_m: ah,
                velocity_m_per_s: au,
            },
            ChannelBoundary::Exterior {
                depth_m: bh,
                velocity_m_per_s: bu,
            },
        ) => values(&[*ah, *au], &[*bh, *bu]),
        (ChannelBoundary::FluxPort(a), ChannelBoundary::FluxPort(b)) => a.same_bits(b),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_forcing_identity_retains_policy_target_bits() {
        let forcing = ChannelForcingInterval {
            start_s: 0.0,
            end_s: 1.0,
            left: ChannelBoundary::Stage { elevation_m: 0.0 },
            right: ChannelBoundary::Discharge {
                discharge_m3_per_s: 0.0,
            },
            lateral_inputs: vec![],
        };
        assert!(super::forcing(&forcing, &forcing));
        for index in 0..4 {
            let mut changed = forcing.clone();
            match index {
                0 => changed.left = ChannelBoundary::Stage { elevation_m: -0.0 },
                1 => {
                    changed.right = ChannelBoundary::Discharge {
                        discharge_m3_per_s: -0.0,
                    }
                }
                2 => {
                    changed.left = ChannelBoundary::Discharge {
                        discharge_m3_per_s: 0.0,
                    }
                }
                _ => changed.right = ChannelBoundary::Stage { elevation_m: 0.0 },
            }
            assert!(!super::forcing(&forcing, &changed));
        }
    }

    #[test]
    fn reservoir_forcing_identity_retains_head_coefficient_and_policy_bits() {
        let forcing = ChannelForcingInterval {
            start_s: 0.0,
            end_s: 1.0,
            left: ChannelBoundary::QuiescentReservoir {
                elevation_m: 0.0,
                entrance_loss_coefficient: 0.0,
            },
            right: ChannelBoundary::Wall,
            lateral_inputs: vec![],
        };
        assert!(super::forcing(&forcing, &forcing));
        for index in 0..3 {
            let mut changed = forcing.clone();
            changed.left = match index {
                0 => ChannelBoundary::QuiescentReservoir {
                    elevation_m: -0.0,
                    entrance_loss_coefficient: 0.0,
                },
                1 => ChannelBoundary::QuiescentReservoir {
                    elevation_m: 0.0,
                    entrance_loss_coefficient: -0.0,
                },
                _ => ChannelBoundary::Stage { elevation_m: 0.0 },
            };
            assert!(!super::forcing(&forcing, &changed));
        }
    }
}
