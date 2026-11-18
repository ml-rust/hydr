use super::super::query_input::{Arithmetic, QueryResult};
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, NextRepresentableOps, ScalarOps, TensorOps, TwoSumOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;
use solvr::common::arithmetic::Interval;

#[cfg(test)]
#[path = "../../../../../tests/prismatic_perimeter_area_points/contact_data.rs"]
mod contact_data;

/// Enclose partial bank contact using retained distance and the physical support `0 <= distance <= thickness`.
pub(super) fn local_contact<R>(
    a: &mut Arithmetic<'_, R>,
    distance: &Interval<R>,
    bank_length: &Interval<R>,
    thickness: &Interval<R>,
) -> QueryResult<Interval<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    let c = a.client();
    let lower = point_contact(a, distance.lower(), bank_length.lower(), thickness.upper())?;
    let upper_distance = c.minimum(distance.upper(), thickness.lower())?;
    let upper = point_contact(a, &upper_distance, bank_length.upper(), thickness.lower())?;
    // Each bank length is at least its vertical thickness, proving contact >= distance.
    Ok(a.bounds(
        &c.maximum(lower.lower(), distance.lower())?,
        &c.minimum(upper.upper(), bank_length.upper())?,
    )?)
}

fn point_contact<R>(
    a: &mut Arithmetic<'_, R>,
    distance: &Tensor<R>,
    length: &Tensor<R>,
    thickness: &Tensor<R>,
) -> QueryResult<Interval<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    let c = a.client();
    let zero = Tensor::zeros(distance.shape(), DType::F64, c.device())?;
    let one = Tensor::ones(distance.shape(), DType::F64, c.device())?;
    let large = c.ge(thickness, &one)?;
    let small = c.eq(&large, &zero)?;
    let large_h = a.point(&c.where_cond(&large, distance, &zero)?)?;
    let large_length = a.point(&c.where_cond(&large, length, &zero)?)?;
    let large_thickness = c.where_cond(&large, thickness, &one)?;
    let ratio = a.div_positive(&large_length, &large_thickness)?;
    let large_contact = a.mul(&large_h, &ratio)?;
    let small_h = a.point(&c.where_cond(&small, distance, &zero)?)?;
    let small_length = a.point(&c.where_cond(&small, length, &zero)?)?;
    let small_thickness = c.where_cond(&small, thickness, &one)?;
    let fraction = a.div_positive(&small_h, &small_thickness)?;
    let fraction = a.bounds(
        &c.maximum(fraction.lower(), &zero)?,
        &c.minimum(fraction.upper(), &one)?,
    )?;
    let small_contact = a.mul(&fraction, &small_length)?;
    let lower = c.where_cond(&large, large_contact.lower(), small_contact.lower())?;
    let upper = c.where_cond(&large, large_contact.upper(), small_contact.upper())?;
    let is_zero = c.eq(distance, &zero)?;
    Ok(a.bounds(
        &c.where_cond(&is_zero, &zero, &c.maximum(&lower, &zero)?)?,
        &c.where_cond(&is_zero, &zero, &c.minimum(&upper, length)?)?,
    )?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::channel::section::CrossSectionError;
    use crate::channel::section::prismatic::query_input::QueryError;
    use solvr::common::arithmetic::{ArithmeticError, with_interval_arithmetic};

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
            CpuClient::new(CpuDevice::new())
                .with_parallelism(ParallelismConfig::new(Some(1), None)),
        )
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_local_contact_retains_endpoint_distances_across_thickness_ranges() -> Result {
        use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
        run::<CudaRuntime>(CudaClient::new(CudaDevice::new(0))?)
    }
}

#[cfg(test)]
mod physical_tests {
    use super::contact_data;
    use super::*;
    use crate::channel::section::prismatic::query_input;
    use crate::channel::section::{
        CrossSectionBatch, CrossSectionDefinition, CrossSectionPoint, PrismaticRecoveryConfig,
    };
    use numr::ops::{HypotOps, MidpointOps};
    use solvr::common::arithmetic::with_interval_arithmetic;

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
            let length = Tensor::<R>::from_slice(
                &[f64::from_bits(case.bank_length)],
                &[1],
                client.device(),
            )?;
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
        let high =
            Tensor::<R>::from_slice(&[f64::from_bits(case.input_high)], &[1], client.device())?;
        let low =
            Tensor::<R>::from_slice(&[f64::from_bits(case.input_low)], &[1], client.device())?;
        let radius = Tensor::<R>::zeros(&[1], DType::F64, client.device())?;
        let area = with_interval_arithmetic::<R, _, _, _>(
            &client,
            DType::F64,
            "physical remaining-contact input",
            |a| a.expansion_bounds(&high, &low, &radius),
        )?;
        let geometry = section.recover_area_points_tensor(
            &owners,
            &area,
            &PrismaticRecoveryConfig::default(),
        )?;
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
            CpuClient::new(CpuDevice::new())
                .with_parallelism(ParallelismConfig::new(Some(1), None)),
        )
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_physical_contact_retains_positive_distance_below_band_end() -> Result {
        use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
        run::<CudaRuntime>(CudaClient::new(CudaDevice::new(0))?)
    }
}
