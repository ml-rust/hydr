use super::local_contact;
use crate::channel::section::CrossSectionError;
use crate::channel::section::prismatic::query_input::QueryError;
use numr::dtype::DType;
use numr::ops::{CompareOps, NextRepresentableOps, ScalarOps, TensorOps, TwoSumOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;
use solvr::common::arithmetic::{ArithmeticError, Interval, with_interval_arithmetic};

type Result = std::result::Result<(), Box<dyn std::error::Error>>;

fn check_contact<R: Runtime<DType = DType>>(
    contact: &Interval<R>,
    reference: &[[f64; 2]],
) -> Result {
    let lower = contact.lower().try_to_vec::<f64>()?;
    let upper = contact.upper().try_to_vec::<f64>()?;
    assert_eq!(lower.len(), reference.len());
    assert_eq!(upper.len(), reference.len());
    for (index, [reference_lower, reference_upper]) in reference.iter().copied().enumerate() {
        assert!(lower[index].is_finite() && upper[index].is_finite());
        assert!(lower[index] <= reference_lower && reference_upper <= upper[index]);
        if reference_lower == 0.0 && reference_upper == 0.0 {
            assert_eq!(lower[index], 0.0);
            assert_eq!(upper[index], 0.0);
        } else {
            assert!(lower[index] > 0.0);
            let allowance = 1e-10 * reference_lower;
            assert!(reference_lower - lower[index] <= allowance);
            assert!(upper[index] - reference_upper <= allowance);
        }
    }
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
        + RuntimeClient<R>,
{
    let huge = f64::from_bits(0x6570_0000_0000_0000);
    let tiny = f64::from_bits(0x1a70_0000_0000_0000);
    let two_tiny = f64::from_bits(0x1a80_0000_0000_0000);
    let distance = Tensor::<R>::from_slice(&[tiny, tiny, 0.0], &[3], client.device())?;
    let bank_length = Tensor::<R>::from_slice(&[huge; 3], &[3], client.device())?;
    let thickness = Tensor::<R>::from_slice(&[huge; 3], &[3], client.device())?;
    let contacts = with_interval_arithmetic::<R, _, _, _>(
        &client,
        DType::F64,
        "retained endpoint contact range",
        |a| {
            let distance = a.point(&distance)?;
            let bank_length = a.point(&bank_length)?;
            let thickness = a.point(&thickness)?;
            let contacts = (|| {
                let left = local_contact(a, &distance, &bank_length, &thickness)?;
                let right = local_contact(a, &distance, &bank_length, &thickness)?;
                let sum = a.add(&left, &right)?;
                Ok::<_, QueryError>((left, right, sum))
            })();
            Ok(contacts)
        },
    )?;
    let (left, right, sum) = match contacts {
        Ok(contacts) => contacts,
        Err(QueryError::Geometry(error)) => return Err(error.into()),
        Err(QueryError::Arithmetic(error)) => return Err(error.into()),
    };
    // Rows retain lower-end distance, upper-end remaining distance, and exact zero.
    let per_bank = [[tiny, tiny], [tiny, tiny], [0.0, 0.0]];
    check_contact(&left, &per_bank)?;
    check_contact(&right, &per_bank)?;
    check_contact(
        &sum,
        &[[two_tiny, two_tiny], [two_tiny, two_tiny], [0.0, 0.0]],
    )?;

    let distance = Tensor::<R>::from_slice(&[0.25, 0.5, 0.125, 1.0], &[4], client.device())?;
    let bank_length = Tensor::<R>::from_slice(&[1.0, 4.0, 0.5, 8.0], &[4], client.device())?;
    let thickness = Tensor::<R>::from_slice(&[0.5, 2.0, 0.25, 4.0], &[4], client.device())?;
    let contact = with_interval_arithmetic::<R, _, _, _>(
        &client,
        DType::F64,
        "mixed contact thickness ranges",
        |a| {
            let distance = a.point(&distance)?;
            let bank_length = a.point(&bank_length)?;
            let thickness = a.point(&thickness)?;
            Ok(local_contact(a, &distance, &bank_length, &thickness))
        },
    )?;
    let contact = match contact {
        Ok(contact) => contact,
        Err(QueryError::Geometry(error)) => return Err(error.into()),
        Err(QueryError::Arithmetic(error)) => return Err(error.into()),
    };
    check_contact(
        &contact,
        &[[0.5, 0.5], [1.0, 1.0], [0.25, 0.25], [2.0, 2.0]],
    )?;

    let distance = Tensor::<R>::from_slice(&[0.25], &[1], client.device())?;
    let bank_length = Tensor::<R>::from_slice(&[2.0], &[1], client.device())?;
    let lower = Tensor::<R>::from_slice(&[0.5], &[1], client.device())?;
    let upper = Tensor::<R>::from_slice(&[2.0], &[1], client.device())?;
    let contact = with_interval_arithmetic::<R, _, _, _>(
        &client,
        DType::F64,
        "contact thickness crossing one",
        |a| {
            let distance = a.point(&distance)?;
            let bank_length = a.point(&bank_length)?;
            let thickness = a.bounds(&lower, &upper)?;
            Ok(local_contact(a, &distance, &bank_length, &thickness))
        },
    )?;
    let contact = match contact {
        Ok(contact) => contact,
        Err(QueryError::Geometry(error)) => return Err(error.into()),
        Err(QueryError::Arithmetic(error)) => return Err(error.into()),
    };
    check_contact(&contact, &[[0.25, 1.0]])?;

    for (lower, upper) in [(0.0, 0.0), (0.0, 1.0), (-0.5, 0.5), (-1.0, -0.5)] {
        let lower = Tensor::<R>::from_slice(&[lower], &[1], client.device())?;
        let upper = Tensor::<R>::from_slice(&[upper], &[1], client.device())?;
        let result = with_interval_arithmetic::<R, _, _, _>(
            &client,
            DType::F64,
            "nonpositive contact thickness",
            |a| {
                let distance = a.point(&distance)?;
                let bank_length = a.point(&bank_length)?;
                let thickness = a.bounds(&lower, &upper)?;
                Ok(local_contact(a, &distance, &bank_length, &thickness))
            },
        );
        assert!(matches!(
            result,
            Err(ArithmeticError::Range { .. })
                | Ok(Err(QueryError::Arithmetic(ArithmeticError::Range { .. })))
                | Ok(Err(QueryError::Geometry(
                    CrossSectionError::Numerical { .. }
                )))
        ));
    }
    Ok(())
}

#[test]
fn cpu_local_contact_retains_endpoint_distances_across_thickness_ranges() -> Result {
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};
    run::<CpuRuntime>(
        CpuClient::new(CpuDevice::new()).with_parallelism(ParallelismConfig::new(Some(1), None)),
    )
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_local_contact_retains_endpoint_distances_across_thickness_ranges() -> Result {
    use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
    run::<CudaRuntime>(CudaClient::new(CudaDevice::new(0))?)
}
