//! Channel cross-section epochs with bit-identical wet area, discharge, and volume conservation.

mod apply;
mod schedule;

pub use schedule::{
    ChannelSectionEpoch, ChannelSectionSchedule, active_channel_section_epochs, active_geometry_at,
    check_channel_section_schedules, geometry_digest_bits, next_channel_section_event_after,
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::channel::{
        ChannelConfig, ChannelError, ChannelGeometry, ChannelInitialState, ChannelReach,
        ChannelReachDefinition, RectangularChannelCell,
    };
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};

    fn rect_geom(width_m: f64, max_depth_m: f64) -> ChannelGeometry {
        ChannelGeometry::Rectangular {
            cells: vec![RectangularChannelCell {
                id: "c0".into(),
                length_m: 2.0,
                width_m,
                bed_elevation_m: 0.0,
                maximum_depth_m: max_depth_m,
                manning_n: 0.0,
            }],
        }
    }

    #[test]
    fn section_epoch_conserves_area_and_rejects_above_new_bank() -> Result<(), ChannelError> {
        let client = CpuClient::new(CpuDevice::new());
        let def = ChannelReachDefinition {
            vertical_datum: "datum".into(),
            provenance: "synthetic".into(),
            geometry: rect_geom(4.0, 2.0),
            lateral_bindings: Vec::new(),
            config: ChannelConfig::default(),
        };
        let narrowed = rect_geom(2.0, 2.0);
        let schedule = ChannelSectionSchedule {
            schedule_id: "narrow".into(),
            reach_id: "r".into(),
            epochs: vec![ChannelSectionEpoch {
                time_s: 1.0,
                geometry: narrowed.clone(),
                evidence: "Channel constriction".into(),
            }],
        };
        check_channel_section_schedules(&[("r", &def)], std::slice::from_ref(&schedule))?;
        assert_eq!(
            next_channel_section_event_after(std::slice::from_ref(&schedule), 0.0),
            Some(1.0)
        );
        assert_eq!(
            active_channel_section_epochs(std::slice::from_ref(&schedule), 1.0),
            vec![1]
        );

        // Volume = 4.0 m3 (area = 2.0 m2): depth rises from 0.5 m (width 4) to 1.0 m (width 2).
        let reach = ChannelReach::<CpuRuntime>::new(
            client.clone(),
            def.clone(),
            ChannelInitialState {
                time_s: 0.0,
                volume_m3: vec![4.0],
                longitudinal_momentum_m4_per_s: vec![1.0],
            },
        )?;
        let updated = reach.coupling_with_geometry(narrowed.clone())?;
        let state = updated.state()?;
        assert_eq!(state[0].volume_m3.to_bits(), 4.0_f64.to_bits());
        assert_eq!(
            state[0].longitudinal_momentum_m4_per_s.to_bits(),
            1.0_f64.to_bits()
        );
        assert_eq!(state[0].depth_m, 1.0);

        // Volume = 10.0 m3 (area = 5.0 m2): depth on width 2 would be 2.5 m > bank limit 2.0 m -> rejects.
        let full = ChannelReach::<CpuRuntime>::new(
            client,
            def,
            ChannelInitialState {
                time_s: 0.0,
                volume_m3: vec![10.0],
                longitudinal_momentum_m4_per_s: vec![0.0],
            },
        )?;
        assert!(full.coupling_with_geometry(narrowed).is_err());
        Ok(())
    }
}
