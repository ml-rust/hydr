use super::fixtures::PROFILES;
use super::support::{Client, Result, batch, expansion};
use crate::channel::flow::surveyed::error::{HydraulicError, HydraulicResult};
use crate::channel::section::CrossSectionError;
use numr::dtype::DType;
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

fn invalid<T>(result: HydraulicResult<T>) -> Result {
    let error = result
        .err()
        .ok_or("Source capture accepts an invalid selected point")?;
    assert!(
        matches!(
            error,
            HydraulicError::Geometry(CrossSectionError::InvalidInput { .. })
        ),
        "{error:?}"
    );
    Ok(())
}

pub(super) fn run<R>(client: R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: Client<R>,
{
    let section = batch::<R>(&client)?;
    let owner = Tensor::<R>::from_slice(&[0i64], &[1], client.device())?;
    let one = 1.0f64.to_bits();
    let baseline = [[one, 0, 0], [0; 3], [one, 0, 0], [0; 3]];
    let make = |points: [[u64; 3]; 4]| {
        points
            .iter()
            .map(|point| expansion::<R>(&client, &[*point], false))
            .collect::<Result<Vec<_>>>()
    };
    let input = make(baseline)?;
    for owner_value in [-1, i64::try_from(PROFILES.len())?] {
        let invalid_owner = Tensor::<R>::from_slice(&[owner_value], &[1], client.device())?;
        invalid(section.capture_riemann_points(
            &invalid_owner,
            &input[0],
            &input[1],
            &input[2],
            &input[3],
        ))?;
    }
    let scalar_owner = Tensor::<R>::from_slice(&[0i64], &[], client.device())?;
    let float_owner = Tensor::<R>::from_slice(&[0.0f64], &[1], client.device())?;
    for invalid_owner in [&scalar_owner, &float_owner] {
        invalid(section.capture_riemann_points(
            invalid_owner,
            &input[0],
            &input[1],
            &input[2],
            &input[3],
        ))?;
    }
    let extra = expansion::<R>(&client, &[[one, 0, 0]; 2], false)?;
    invalid(section.capture_riemann_points(&owner, &extra, &input[1], &input[2], &input[3]))?;
    for field in 0..4 {
        let mut points = baseline;
        points[field][2] = 1;
        let input = make(points)?;
        invalid(
            section.capture_riemann_points(&owner, &input[0], &input[1], &input[2], &input[3]),
        )?;
    }
    for area_field in [0, 2] {
        let mut points = baseline;
        points[area_field] = [0, (-f64::from_bits(1)).to_bits(), 0];
        let input = make(points)?;
        invalid(
            section.capture_riemann_points(&owner, &input[0], &input[1], &input[2], &input[3]),
        )?;
        for discharge in [f64::from_bits(1), -f64::from_bits(1)] {
            let mut points = baseline;
            points[area_field] = [0; 3];
            points[area_field + 1] = [0, discharge.to_bits(), 0];
            let input = make(points)?;
            invalid(
                section.capture_riemann_points(&owner, &input[0], &input[1], &input[2], &input[3]),
            )?;
        }
    }
    Ok(())
}
