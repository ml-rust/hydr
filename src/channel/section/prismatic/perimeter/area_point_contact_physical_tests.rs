use super::local_contact;
use crate::channel::section::prismatic::query_input;
use crate::channel::section::{
    CrossSectionBatch, CrossSectionDefinition, CrossSectionPoint, PrismaticRecoveryConfig,
};
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, HypotOps, MidpointOps, NextRepresentableOps, ScalarOps, TensorOps,
    TwoSumOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;
use solvr::common::arithmetic::{Interval, with_interval_arithmetic};

#[path = "../../../../../tests/prismatic_perimeter_area_points/contact_data.rs"]
mod contact_data;

type Result<T = ()> = std::result::Result<T, Box<dyn std::error::Error>>;

fn check_contact<R: Runtime<DType = DType>>(
    actual: &Interval<R>,
    expected: [u64; 2],
    allowance: u64,
    label: &str,
) -> Result {
    let lower = actual.lower().item::<f64>()?;
    let upper = actual.upper().item::<f64>()?;
    let reference = expected.map(f64::from_bits);
    let allowance = f64::from_bits(allowance);
    assert!(lower.is_finite() && upper.is_finite(), "{label}");
    assert!(
        0.0 < lower && lower <= reference[0] && reference[1] <= upper,
        "{label}: [{lower},{upper}], reference={reference:?}"
    );
    let error = (lower - reference[1])
        .abs()
        .max((upper - reference[0]).abs());
    assert!(
        allowance > 0.0 && allowance <= 1e-10 * reference[1],
        "{label}: allowance={allowance}"
    );
    assert!(
        error <= allowance,
        "{label}: error={error}, allowance={allowance}"
    );
    Ok(())
}

fn run<R>(client: R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + HypotOps<R>
        + MidpointOps<R>
        + RuntimeClient<R>,
{
    for case in contact_data::PRIVATE_CONTACT_CASES {
        let distance =
            Tensor::<R>::from_slice(&[f64::from_bits(case.distance)], &[1], client.device())?;
        let length =
            Tensor::<R>::from_slice(&[f64::from_bits(case.bank_length)], &[1], client.device())?;
        let thickness =
            Tensor::<R>::from_slice(&[f64::from_bits(case.thickness)], &[1], client.device())?;
        let contact = with_interval_arithmetic::<R, _, _, _>(
            &client,
            DType::F64,
            "frozen retained contact control",
            |a| {
                let distance = a.point(&distance)?;
                let length = a.point(&length)?;
                let thickness = a.point(&thickness)?;
                Ok(local_contact(a, &distance, &length, &thickness))
            },
        )?;
        let contact = match contact {
            Ok(contact) => contact,
            Err(query_input::QueryError::Geometry(error)) => return Err(error.into()),
            Err(query_input::QueryError::Arithmetic(error)) => return Err(error.into()),
        };
        check_contact(&contact, case.expected, case.allowance, case.id)?;
    }

    let case = &contact_data::BINARY_REMAINING_CASE;
    let definition = CrossSectionDefinition {
        id: case.profile_id.into(),
        vertical_datum: "Frozen original-coordinate contact datum".into(),
        points: case
            .coordinates
            .iter()
            .map(|point| CrossSectionPoint {
                station_m: f64::from_bits(point[0]),
                elevation_m: f64::from_bits(point[1]),
            })
            .collect(),
        maximum_stage_m: f64::from_bits(case.cap_stage),
        profile_provenance: "Frozen original-segment remaining-contact reference".into(),
    };
    let section =
        CrossSectionBatch::<R>::new(client.clone(), vec![definition])?.certify_prismatic()?;
    let perimeter = section.certify_perimeter()?;
    let owners = Tensor::<R>::from_slice(&[0_i64], &[1], client.device())?;
    let high = Tensor::<R>::from_slice(&[f64::from_bits(case.input_high)], &[1], client.device())?;
    let low = Tensor::<R>::from_slice(&[f64::from_bits(case.input_low)], &[1], client.device())?;
    let radius = Tensor::<R>::zeros(&[1], DType::F64, client.device())?;
    let area = with_interval_arithmetic::<R, _, _, _>(
        &client,
        DType::F64,
        "physical remaining-contact input",
        |a| a.expansion_bounds(&high, &low, &radius),
    )?;
    let geometry =
        section.recover_area_points_tensor(&owners, &area, &PrismaticRecoveryConfig::default())?;
    let coordinates = geometry.band_coordinates();
    let candidate = coordinates
        .candidate_mask()
        .narrow(1, case.band_index, 1)?
        .item::<f64>()?;
    assert_eq!(candidate, 1.0, "{}", case.id);
    let remaining = query_input::checked(
        &geometry.context,
        "physical retained remaining contact",
        |a| {
            let band = |a: &mut query_input::Arithmetic<'_, R>, value: &Interval<R>| {
                a.bounds(
                    &value
                        .lower()
                        .narrow(1, case.band_index, 1)?
                        .squeeze(Some(1)),
                    &value
                        .upper()
                        .narrow(1, case.band_index, 1)?
                        .squeeze(Some(1)),
                )
            };
            let distance = band(a, coordinates.depth_below_end_m())?;
            let thickness = band(a, &perimeter.contact.thickness)?;
            let left = band(a, &perimeter.contact.left)?;
            let right = band(a, &perimeter.contact.right)?;
            let left = local_contact(a, &distance, &left, &thickness)?;
            let right = local_contact(a, &distance, &right, &thickness)?;
            Ok(a.add(&left, &right)?)
        },
    )?;
    check_contact(&remaining, case.expected, case.allowance, case.id)?;

    let retained_lower = coordinates
        .depth_below_end_m()
        .lower()
        .narrow(1, case.band_index, 1)?
        .item::<f64>()?;
    assert!(retained_lower > 0.0, "{}", case.id);
    let end_depth = geometry
        .context
        .end_depth
        .upper()
        .narrow(1, case.band_index, 1)?
        .squeeze(Some(1));
    let reconstructed = client.sub(&end_depth, geometry.depth_m().upper())?;
    assert!(
        reconstructed.item::<f64>()? <= 0.0,
        "{}: absolute-depth subtraction cannot certify positive remaining distance",
        case.id
    );
    Ok(())
}

#[test]
fn cpu_physical_contact_retains_positive_distance_below_band_end() -> Result {
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};
    run::<CpuRuntime>(
        CpuClient::new(CpuDevice::new()).with_parallelism(ParallelismConfig::new(Some(1), None)),
    )
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_physical_contact_retains_positive_distance_below_band_end() -> Result {
    use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
    run::<CudaRuntime>(CudaClient::new(CudaDevice::new(0))?)
}
