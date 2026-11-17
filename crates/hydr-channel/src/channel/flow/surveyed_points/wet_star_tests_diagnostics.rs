use super::{frozen, support};
use frozen::{Result, hydraulic};
use numr::{dtype::DType, runtime::Runtime};

pub(super) fn check_diagnostics<R>(
    pair: &crate::channel::flow::surveyed_points::paired_sources::PairedSources<'_, R>,
    domain: &crate::channel::flow::surveyed_points::wet_star_types::WetStarDomain<R>,
) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: support::DepthClient<R>,
{
    use crate::channel::flow::surveyed::error::checked;
    hydraulic(checked(
        pair.geometry.client(),
        "wet star diagnostic identities",
        |a| {
            let stage = pair.select_rows(&domain.parent_rows, &domain.query)?;
            for endpoint in &domain.endpoints {
                let area = a.expansion_interval(&endpoint.area)?;
                for (lower, upper) in area
                    .lower()
                    .try_to_vec::<f64>()?
                    .into_iter()
                    .zip(area.upper().try_to_vec::<f64>()?)
                {
                    assert!(lower <= upper);
                }
                for (side, diagnostic) in endpoint.sides.iter().enumerate() {
                    let source_velocity = if side == 0 {
                        &stage.velocity.left
                    } else {
                        &stage.velocity.right
                    };
                    let source_discharge = if side == 0 {
                        &stage.velocity.input().left_discharge
                    } else {
                        &stage.velocity.input().right_discharge
                    };
                    let source_area = if side == 0 {
                        &stage.velocity.input().left_area
                    } else {
                        &stage.velocity.input().right_area
                    };
                    for (actual, expected) in [
                        (&diagnostic.source_area, source_area),
                        (&diagnostic.source_velocity, source_velocity),
                        (&diagnostic.source_discharge, source_discharge),
                    ] {
                        assert_eq!(
                            actual.high().try_to_vec::<f64>()?,
                            expected.high().try_to_vec::<f64>()?
                        );
                        assert_eq!(
                            actual.low().try_to_vec::<f64>()?,
                            expected.low().try_to_vec::<f64>()?
                        );
                        assert_eq!(
                            actual.abs_error().try_to_vec::<f64>()?,
                            expected.abs_error().try_to_vec::<f64>()?
                        );
                    }
                    for value in [
                        &diagnostic.area_difference,
                        &diagnostic.velocity_change,
                        &diagnostic.discharge_change,
                    ] {
                        assert!(
                            value
                                .abs_error()
                                .try_to_vec::<f64>()?
                                .iter()
                                .all(|radius| radius.is_finite() && *radius >= 0.0)
                        );
                    }
                    let unchanged = diagnostic.unchanged.try_to_vec::<f64>()?;
                    let actual = diagnostic.discharge.high().try_to_vec::<f64>()?;
                    let captured = diagnostic.source_discharge.high().try_to_vec::<f64>()?;
                    for (index, unchanged) in unchanged.into_iter().enumerate() {
                        if unchanged != 0.0 {
                            assert_eq!(actual[index], captured[index]);
                        }
                    }
                }
            }
            Ok(())
        },
    ))?;
    let velocity = frozen::endpoints(&domain.velocity)?;
    let discharge = frozen::endpoints(&domain.discharge)?;
    for side in 0..2 {
        let side_velocity = frozen::endpoints(&domain.side_velocity[side])?;
        let side_discharge = frozen::endpoints(&domain.side_discharge[side])?;
        for index in 0..velocity.len() {
            assert!(
                side_velocity[index][0] <= velocity[index][0]
                    && velocity[index][1] <= side_velocity[index][1]
            );
            assert!(
                side_discharge[index][0] <= discharge[index][0]
                    && discharge[index][1] <= side_discharge[index][1]
            );
        }
        let range = &domain.side_operands[side];
        for value in [
            &range.area_difference,
            &range.velocity_change,
            &range.discharge_change,
            &range.discharge,
        ] {
            assert!(
                value
                    .abs_error()
                    .try_to_vec::<f64>()?
                    .iter()
                    .all(|x| x.is_finite() && *x >= 0.0)
            );
        }
        assert_eq!(
            frozen::endpoints(&frozen::bounds(pair.geometry.client(), &range.discharge)?)?,
            side_discharge
        );
    }
    Ok(())
}

pub(super) fn check_frozen_differences<R>(
    pair: &crate::channel::flow::surveyed_points::paired_sources::PairedSources<'_, R>,
    domain: &crate::channel::flow::surveyed_points::wet_star_types::WetStarDomain<R>,
) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: support::DepthClient<R>,
{
    // Independent rational star bounds minus exact captured source high+low, projected outward.
    // Source rationals match references.json physical_input_verbatim. Parent order matches frozen_star_states.
    let differences: [[[[u64; 2]; 2]; 2]; 4] = [
        [
            [
                [0xbff0000000000000, 0xbfefffffffffffff],
                [0x3fffffffffffffff, 0x4000000000000000],
            ],
            [
                [0xbff0000000000000, 0xbfefffffffffffff],
                [0xc000000000000000, 0xbfffffffffffffff],
            ],
        ],
        [[[0, 0], [0, 0]], [[0, 0], [0, 0]]],
        [
            [
                [0xbcc2263845ea350e, 0xbcc2263845ea350d],
                [0x3cc8025eff7c99be, 0x3cc8025eff7c99bf],
            ],
            [
                [0xbc8207bfde3a47e7, 0xbc8207bfde3a47e6],
                [0xbc87da100836641c, 0xbc87da100836641b],
            ],
        ],
        [
            [
                [0x3aefffffffffffff, 0x3af0000000000000],
                [0xbaea20bd700c2c3e, 0xbaea20bd700c2c3d],
            ],
            [
                [0xbaf0000000000001, 0xbaf0000000000000],
                [0xbaea20bd700c2c3e, 0xbaea20bd700c2c3d],
            ],
        ],
    ];
    let parents = domain.parent_rows.try_to_vec::<i64>()?;
    for (side, operands) in domain.side_operands.iter().enumerate() {
        let area = frozen::endpoints(&frozen::bounds(
            pair.geometry.client(),
            &operands.area_difference,
        )?)?;
        let discharge = frozen::endpoints(&frozen::bounds(
            pair.geometry.client(),
            &operands.discharge_change,
        )?)?;
        for (row, &parent) in parents.iter().enumerate() {
            let truth = differences[usize::try_from(parent)?][side];
            for (field, (actual, expected)) in [(area[row], truth[0]), (discharge[row], truth[1])]
                .into_iter()
                .enumerate()
            {
                let expected = expected.map(f64::from_bits);
                assert!(
                    actual[0] <= expected[0] && expected[1] <= actual[1],
                    "parent={parent}, side={side}, difference={field}: {actual:?} excludes {expected:?}"
                );
            }
        }
    }
    Ok(())
}
