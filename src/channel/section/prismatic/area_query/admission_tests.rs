use super::fixtures::INVALID;
use super::test_support::{Client, Result, hydraulic, input, query};
use crate::channel::flow::surveyed::error::{self, HydraulicError};
use crate::channel::section::{CrossSectionError, HydraulicGeometry, PrismaticSectionBatch};
use numr::dtype::DType;
use numr::runtime::Runtime;
use std::collections::BTreeSet;

pub(super) fn run<R>(client: &R::Client, section: &PrismaticSectionBatch<R>) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: Client<R>,
{
    assert_eq!(INVALID.len(), 6);
    assert_eq!(
        INVALID
            .iter()
            .map(|case| case.id)
            .collect::<BTreeSet<_>>()
            .len(),
        6
    );
    let geometry = HydraulicGeometry::from_batch(section);
    for case in INVALID {
        let malformed = case.cause == "invalid_parameter_metadata";
        let input = input::<R>(client, section, &[case.chart], false, malformed)?;
        let (stage, error) = hydraulic(error::checked::<R, _>(
            client,
            "private chart admission",
            |a| {
                let bands = geometry.gather(a, &input.owners)?;
                match query(a, &bands, &input) {
                    Err(error) => Ok(("chart", error)),
                    Ok(query) => match query.prove_support(a, &bands) {
                        Err(error) => Ok(("support", error)),
                        Ok(_) => Err(HydraulicError::Numerical {
                            context: format!("Private chart {} accepts unsupported input", case.id),
                        }),
                    },
                }
            },
        ))?;
        assert!(
            matches!(
                error,
                HydraulicError::Geometry(CrossSectionError::InvalidInput { .. })
            ),
            "{}, cause={}: {error:?}",
            case.id,
            case.cause
        );
        let expected_stage = if case.cause == "invalid_physical_support" {
            "support"
        } else {
            "chart"
        };
        assert_eq!(stage, expected_stage, "{}, cause={}", case.id, case.cause);
    }
    Ok(())
}
