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
    let client =
        CpuClient::new(CpuDevice::new()).with_parallelism(ParallelismConfig::new(Some(1), None));
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
