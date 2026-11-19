//! Flux port admission, depletion, coupling reservation, restart and tensor operands.

use super::common::{Reach, Shape, TestResult, check_water, definition, initial, reach};
use super::flux_port::{SHAPES, advance, held, invariant, port, uniform_ports, uniform_reach};
use hydr::channel::{
    ChannelBoundary, ChannelError, ChannelGeometry, ChannelSurveyedConfig, ChannelTensorBoundary,
    ChannelTensorFluxPort, ChannelTensorForcingInterval, CrossSectionDefinition, CrossSectionPoint,
    SurveyedChannelCell,
};
use numr::runtime::RuntimeClient;
use numr::tensor::Tensor;

/// Rejection text of one step, which must leave the reach unchanged.
fn rejection(
    origin: &Reach,
    left: &ChannelBoundary,
    right: &ChannelBoundary,
) -> Result<(ChannelError, String), Box<dyn std::error::Error>> {
    let before = origin.snapshot_data()?;
    let end = origin.time() + 0.25;
    let Err(error) = origin.step(end, &held(origin, end, left, right)) else {
        return Err("the port was admitted".into());
    };
    assert_eq!(origin.snapshot_data()?, before, "rejection changed state");
    let message = error.to_string();
    Ok((error, message))
}

#[test]
fn port_speed_below_the_end_cell_characteristic_rejects() -> TestResult {
    for shape in SHAPES {
        let h = 1.0;
        let u = 0.5;
        let slow = port(
            0.0,
            shape.pressure(h),
            u + 0.5 * shape.sound(h),
            Some(invariant(shape, h)),
        )?;
        let origin = uniform_reach(shape, 3, h, u)?;
        for (left, right, cell) in [
            (&slow, &ChannelBoundary::Wall, 0),
            (&ChannelBoundary::Wall, &slow, 2),
        ] {
            let (error, message) = rejection(&origin, left, right)?;
            assert!(
                matches!(error, ChannelError::InvalidInput { cell: Some(found), .. } if found == cell),
                "{error:?}"
            );
            for fragment in [
                "is below the end cell |u|+c",
                &format!("speed={}", u + 0.5 * shape.sound(h)),
                "supply a signal-speed bound of at least the end cell characteristic speed",
            ] {
                assert!(message.contains(fragment), "{message} lacks {fragment}");
            }
        }
        // The exact characteristic speed is admitted.
        let exact = port(
            0.0,
            shape.pressure(h),
            u + shape.sound(h),
            Some(invariant(shape, h)),
        )?;
        advance(origin, 1, 0.25, &exact, &ChannelBoundary::Wall)?;
    }
    Ok(())
}

#[test]
fn export_from_a_dry_end_cell_rejects() -> TestResult {
    for shape in SHAPES {
        let export = port(0.5, 0.0, 1.0, None)?;
        let origin = uniform_reach(shape, 2, 0.0, 0.0)?;
        for (left, right, side) in [
            (&export, &ChannelBoundary::Wall, "left"),
            (&ChannelBoundary::Wall, &export, "right"),
        ] {
            let (error, message) = rejection(&origin, left, right)?;
            assert!(
                matches!(error, ChannelError::InvalidInput { .. }),
                "{error:?}"
            );
            let expected = format!(
                "flux port {side} exports water=0.5 m³/s from a dry end cell: supply zero or entering water at a dry end"
            );
            assert!(message.contains(&expected), "{message}");
        }
    }
    Ok(())
}

/// Two cells of a horizontal trapezoid survey at unit depth, at rest.
fn surveyed_reach() -> Result<Reach, Box<dyn std::error::Error>> {
    let shape = Shape::Rectangle {
        gravity: 9.81,
        width: 2.0,
    };
    let mut layout = definition(shape, 2, 4.0, 4.0, 1.0);
    layout.geometry = ChannelGeometry::PrismaticSurveyed {
        profile: CrossSectionDefinition {
            id: "trapezoid".into(),
            vertical_datum: layout.vertical_datum.clone(),
            points: [(-5.0, 4.0), (-1.0, 0.0), (1.0, 0.0), (5.0, 4.0)]
                .into_iter()
                .map(|(station_m, elevation_m)| CrossSectionPoint {
                    station_m,
                    elevation_m,
                })
                .collect(),
            maximum_stage_m: 4.0,
            profile_provenance: "Explicit analytical trapezoid survey".into(),
        },
        cells: (0..2)
            .map(|i| SurveyedChannelCell {
                id: format!("cell-{i}"),
                length_m: 4.0,
                manning_n: 0.0,
            })
            .collect(),
        numerical: ChannelSurveyedConfig::default(),
    };
    // Unit depth on the 2 m base with unit side slopes holds 3 m² per metre.
    let volume = 4.0 * 3.0;
    reach(
        layout,
        hydr::channel::ChannelInitialState {
            time_s: 0.0,
            volume_m3: vec![volume; 2],
            longitudinal_momentum_m4_per_s: vec![0.0; 2],
        },
    )
}

#[test]
fn surveyed_reaches_admit_standalone_flux_ports_and_reject_coordinated_ones() -> TestResult {
    let origin = surveyed_reach()?;
    let closed = port(0.0, 0.0, 10.0, Some(10.0))?;
    let wall = ChannelBoundary::Wall;
    for (left, right) in [(&closed, &wall), (&wall, &closed)] {
        advance(origin.clone(), 1, 0.25, left, right)?;
        let Err(error) = origin.bind_coupling_host(0.0, 0.25, Some(left), Some(right), &[]) else {
            return Err("a coordinated surveyed port was admitted".into());
        };
        let message = error.to_string();
        assert!(
            message.contains("surveyed coordinator endpoints") && message.contains("FluxPort"),
            "{message}"
        );
    }
    Ok(())
}

#[test]
fn export_counts_in_the_depletion_limit_and_the_coupling_reservation() -> TestResult {
    for shape in SHAPES {
        // A shallow cell that one export drains within one requested interval.
        let h = 0.01;
        let (length, export) = (100.0, 1.0);
        let layout = definition(shape, 1, length, 10.0, 100.0);
        let origin = reach(layout, initial(shape, &[length], &[h], &[0.0]))?;
        let volume = origin.state()?[0].volume_m3;
        let drain = port(export, shape.pressure(h), shape.sound(h) + 1.0, None)?;
        let wall = ChannelBoundary::Wall;
        let forcing = origin.bind_coupling_host(0.0, 10.0, Some(&wall), Some(&drain), &[])?;
        let reservation = origin.coupling_reservation(&forcing)?;
        let outgoing = reservation.outgoing_rate_m3_per_s.try_to_vec::<f64>()?;
        assert_eq!(outgoing, [export], "{shape:?}");
        assert_eq!(
            reservation.face_depletion_limit_s.to_bits(),
            (volume / export).to_bits(),
            "{shape:?}"
        );
        let next = origin.step(10.0, &held(&origin, 10.0, &wall, &drain))?.next;
        check_water(&next)?;
        let receipt = next.last_step()?.ok_or("missing depletion receipt")?;
        assert_eq!(
            receipt.depletion_limit_s.to_bits(),
            (volume / export).to_bits()
        );
        assert!(receipt.retries >= 1, "{shape:?}: {receipt:?}");
        assert!(next.time() <= receipt.depletion_limit_s);
        assert!(next.state()?[0].volume_m3 >= 0.0);
    }
    Ok(())
}

#[test]
fn port_forcing_survives_snapshot_import_and_restore() -> TestResult {
    for shape in SHAPES {
        let h = 1.0;
        let u = 0.5 * shape.sound(h);
        let [left, right] = uniform_ports(shape, h, u)?;
        let origin = uniform_reach(shape, 3, h, u)?;
        let current = advance(origin.clone(), 2, 0.25, &left, &right)?;
        let saved = current.snapshot_data()?;
        let interval = saved
            .last_interval
            .as_ref()
            .ok_or("missing port interval")?;
        assert_eq!(interval.forcing.left, left);
        assert_eq!(interval.forcing.right, right);
        let cold = origin.import_snapshot(&saved)?;
        let trusted = origin.restore(&current.snapshot())?;
        assert_eq!(cold.snapshot_data()?, saved);
        assert_eq!(trusted.snapshot_data()?, saved);
        let expected = advance(current, 1, 0.25, &left, &right)?.snapshot_data()?;
        assert_eq!(
            advance(cold, 1, 0.25, &left, &right)?.snapshot_data()?,
            expected
        );
        assert_eq!(
            advance(trusted, 1, 0.25, &left, &right)?.snapshot_data()?,
            expected
        );
        // A forged port operand rejects by identity of the replayed interval.
        let mut forged = saved.clone();
        if let Some(last) = forged.last_interval.as_mut() {
            last.forcing.left = port(-1.0, 1.0, 10.0, Some(10.0))?;
        }
        assert!(origin.import_snapshot(&forged).is_err(), "{shape:?}");
    }
    Ok(())
}

#[test]
fn tensor_ports_match_host_ports_bit_for_bit() -> TestResult {
    let client = crate::runtime::client();
    let value = |v: f64| Tensor::from_slice(&[v], &[1], client.device());
    for shape in SHAPES {
        let h = 1.0;
        let u = 0.5 * shape.sound(h);
        let [left, right] = uniform_ports(shape, h, u)?;
        let tensor = |boundary: &ChannelBoundary| -> Result<ChannelTensorBoundary<_>, Box<dyn std::error::Error>> {
            let ChannelBoundary::FluxPort(port) = boundary else {
                return Err("expected a FluxPort".into());
            };
            Ok(ChannelTensorBoundary::FluxPort(Box::new(ChannelTensorFluxPort::new(
                value(port.outward_water_rate_m3_per_s())?,
                value(port.outward_momentum_rate_m4_per_s2())?,
                value(port.signal_speed_m_per_s())?,
                port.exterior_invariant_m_per_s().map(value).transpose()?,
                port.provenance(),
            )?)))
        };
        let origin = uniform_reach(shape, 3, h, u)?;
        let host = origin.step(0.25, &held(&origin, 0.25, &left, &right))?.next;
        let native = origin
            .step_tensor(
                0.25,
                &ChannelTensorForcingInterval {
                    start_s: 0.0,
                    end_s: 0.25,
                    left: tensor(&left)?,
                    right: tensor(&right)?,
                    lateral_inputs: Vec::new(),
                },
            )?
            .next;
        assert_eq!(native.snapshot_data()?, host.snapshot_data()?);
    }
    // A port operand of the wrong shape rejects at construction.
    let wide = Tensor::from_slice(&[0.0, 0.0], &[2], client.device())?;
    let Err(error) =
        ChannelTensorFluxPort::new(wide, value(0.0)?, value(1.0)?, None, "test provider")
    else {
        return Err("a two-value port operand was admitted".into());
    };
    assert!(
        error
            .to_string()
            .contains("flux port outward water rate shape=[2]"),
        "{error}"
    );
    Ok(())
}
