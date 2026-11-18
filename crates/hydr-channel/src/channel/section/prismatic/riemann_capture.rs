use super::batch::PrismaticSectionBatch;
use crate::channel::flow::surveyed::{
    error::{HydraulicError, HydraulicResult},
    types::CapturedInput,
};
use crate::channel::section::CrossSectionError;
use numr::ops::{
    BinaryOps, CompareOps, NextRepresentableOps, ScalarOps, TensorOps, TwoSumOps,
    TypeConversionOps, UnaryOps,
};
use numr::{
    dtype::DType,
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::{
    integrate::CertifiedSimpsonAlgorithms,
    optimize::roots::{CertifiedBisectionAlgorithms, CertifiedUpperBracketAlgorithms},
};

impl<R> PrismaticSectionBatch<R>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>
        + CertifiedSimpsonAlgorithms<R>
        + CertifiedBisectionAlgorithms<R>
        + CertifiedUpperBracketAlgorithms<R>,
{
    pub(super) fn capture_riemann(
        &self,
        owners: &Tensor<R>,
        left_area: &Tensor<R>,
        left_discharge: &Tensor<R>,
        right_area: &Tensor<R>,
        right_discharge: &Tensor<R>,
    ) -> HydraulicResult<CapturedInput<R>> {
        let c = &self.context.source.client;
        let capture = |source: &Tensor<R>| -> HydraulicResult<Tensor<R>> {
            let copy = Tensor::empty(source.shape(), source.dtype(), c.device())?;
            c.copy_into(&copy, source)?;
            Ok(copy)
        };
        let input = CapturedInput {
            owners: capture(owners)?,
            left_area: capture(left_area)?,
            left_discharge: capture(left_discharge)?,
            right_area: capture(right_area)?,
            right_discharge: capture(right_discharge)?,
        };
        if owners.numel() == 0 {
            return Ok(input);
        }
        let zero = &self.context.source.zero;
        let count = i64::try_from(self.context.source.definitions.len())
            .map_err(|_| input_error("section count exceeds I64. Split the section batch"))?;
        let zero_owner = Tensor::from_slice(&[0_i64], &[], c.device())?;
        let limit = Tensor::from_slice(&[count], &[], c.device())?;
        let invalid_owner = c.maximum(
            &c.lt(&input.owners, &zero_owner)?,
            &c.ge(&input.owners, &limit)?,
        )?;
        if let Some(query) = self
            .context
            .source
            .rejected_lane(&c.cast(&invalid_owner, DType::F64)?)?
        {
            let owner = input.owners.narrow(0, query, 1)?.item::<i64>()?;
            return Err(input_error(format!(
                "owner={owner}, query={query}, section_count={count}: supply an owner in 0..{count}"
            )));
        }
        for (name, area, discharge) in [
            ("left", &input.left_area, &input.left_discharge),
            ("right", &input.right_area, &input.right_discharge),
        ] {
            let finite_error = |value: &Tensor<R>| -> HydraulicResult<Tensor<R>> {
                Ok(c.maximum(
                    &c.cast(&c.isnan(value)?, DType::F64)?,
                    &c.cast(&c.isinf(value)?, DType::F64)?,
                )?)
            };
            let invalid = c.maximum(
                &finite_error(area)?,
                &c.maximum(
                    &finite_error(discharge)?,
                    &c.maximum(
                        &c.lt(area, zero)?,
                        &c.mul(&c.eq(area, zero)?, &c.ne(discharge, zero)?)?,
                    )?,
                )?,
            )?;
            if let Some(query) = self.context.source.rejected_lane(&invalid)? {
                let owner = input.owners.narrow(0, query, 1)?.item::<i64>()?;
                let area = area.narrow(0, query, 1)?.item::<f64>()?;
                let discharge = discharge.narrow(0, query, 1)?.item::<f64>()?;
                return Err(input_error(format!(
                    "{name} area={area:?}, discharge={discharge:?}, query={query}, section_owner={owner}: supply finite nonnegative area and finite discharge, with zero discharge at zero area"
                )));
            }
        }
        Ok(input)
    }
}

pub(super) fn input_error(context: impl Into<String>) -> HydraulicError {
    CrossSectionError::InvalidInput {
        section: None,
        context: context.into(),
    }
    .into()
}
