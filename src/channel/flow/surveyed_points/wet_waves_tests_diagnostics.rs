use super::{Result, WetWaveDomain, frozen, support};
use numr::{dtype::DType, runtime::Runtime};

pub(super) fn check<R>(domain: &WetWaveDomain<R>) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: support::DepthClient<R>,
{
    let n = domain.star.parent_rows.numel();
    assert_eq!(domain.star.invocation_owner.numel(), n);
    assert_eq!(domain.star.query.query.original_rows.numel(), n);
    for mask in [
        &domain.regions.left_exterior,
        &domain.regions.left_fan,
        &domain.regions.star,
        &domain.regions.right_fan,
        &domain.regions.right_exterior,
    ] {
        assert_eq!(mask.numel(), n);
        assert!(
            mask.try_to_vec::<f64>()?
                .iter()
                .all(|&x| x == 0.0 || x == 1.0)
        );
    }
    for sound in &domain.endpoint_sound {
        for bounds in frozen::endpoints(&sound.physical)? {
            assert!(0.0 <= bounds[0] && bounds[0] <= bounds[1]);
        }
        assert!(
            sound
                .retained
                .abs_error()
                .try_to_vec::<f64>()?
                .iter()
                .all(|&x| x.is_finite() && x >= 0.0)
        );
    }
    for side in &domain.sides {
        for mask in [
            &side.shock_possible,
            &side.rarefaction_possible,
            &side.equality_possible,
            &side.shock_actual,
            &side.shock_conditional,
            &side.shock_stationary,
            &side.equality_stationary,
            &side.valid,
        ] {
            assert!(
                mask.try_to_vec::<f64>()?
                    .iter()
                    .all(|&x| x == 0.0 || x == 1.0)
            );
        }
        let actual = side.shock_actual.try_to_vec::<f64>()?;
        let conditional = side.shock_conditional.try_to_vec::<f64>()?;
        let possible = side.shock_possible.try_to_vec::<f64>()?;
        for row in 0..n {
            assert!(actual[row] + conditional[row] <= possible[row]);
        }
        for value in [
            &side.source_sound,
            &side.shock_offset,
            &side.shock_speed,
            &side.head,
            &side.tail,
        ] {
            let available = value.available.try_to_vec::<f64>()?;
            let bounds = frozen::endpoints(&value.physical)?;
            for (row, bounds) in bounds.into_iter().enumerate() {
                assert!(bounds[0].is_finite() && bounds[1].is_finite() && bounds[0] <= bounds[1]);
                assert!(available[row] == 0.0 || available[row] == 1.0);
            }
            assert!(
                value
                    .retained
                    .abs_error()
                    .try_to_vec::<f64>()?
                    .iter()
                    .all(|&x| x.is_finite() && x >= 0.0)
            );
        }
        for (mask, speed) in [
            (&side.shock_stationary, &side.shock_speed),
            (&side.equality_stationary, &side.head),
        ] {
            let stationary = mask.try_to_vec::<f64>()?;
            let available = speed.available.try_to_vec::<f64>()?;
            let bounds = frozen::endpoints(&speed.physical)?;
            let retained = frozen::components(&speed.retained)?;
            for row in 0..n {
                if stationary[row] != 0.0 {
                    assert_eq!(available[row], 1.0);
                    assert_eq!(bounds[row], [0.0, 0.0]);
                    assert!(retained[row].map(f64::from_bits).iter().all(|&x| x == 0.0));
                }
            }
        }
    }
    for endpoint in &domain.star.endpoints {
        for side in 0..2 {
            let far = endpoint.waves.far_rarefaction[side].try_to_vec::<f64>()?;
            if let Some(local) = &endpoint.waves.local[side] {
                for &row in &local.rows.try_to_vec::<i64>()? {
                    assert_eq!(far[usize::try_from(row)?], 0.0);
                }
                let classification = &local.endpoint.classification;
                assert!(
                    classification
                        .dry_source
                        .try_to_vec::<f64>()?
                        .iter()
                        .all(|&v| v == 0.0)
                );
                for mask in [
                    &classification.shock,
                    &classification.rarefaction,
                    &classification.equality,
                    &classification.uncertain,
                ] {
                    assert_eq!(mask.numel(), local.rows.numel());
                }
                assert_eq!(
                    local.endpoint.equality_limit.available.numel(),
                    local.rows.numel()
                );
            }
        }
    }
    Ok(())
}
