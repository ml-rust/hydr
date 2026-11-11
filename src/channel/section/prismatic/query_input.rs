use super::super::CrossSectionError;
use super::{
    config,
    context::{Context, arithmetic_error},
};
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, NextRepresentableOps, ScalarOps, TensorOps, TwoSumOps,
    TypeConversionOps, UnaryOps,
};
use numr::runtime::{Device, Runtime, RuntimeClient};
use numr::tensor::Tensor;
use solvr::common::arithmetic::{
    ArithmeticError, Interval, IntervalContext, with_interval_arithmetic,
};

pub(super) type Arithmetic<'a, R> = IntervalContext<'a, R, <R as Runtime>::Client>;
pub(super) type QueryResult<T> = Result<T, QueryError>;

pub(super) enum QueryError {
    Geometry(CrossSectionError),
    Arithmetic(ArithmeticError),
}

impl From<CrossSectionError> for QueryError {
    fn from(error: CrossSectionError) -> Self {
        Self::Geometry(error)
    }
}

impl From<ArithmeticError> for QueryError {
    fn from(error: ArithmeticError) -> Self {
        Self::Arithmetic(error)
    }
}

impl From<numr::error::Error> for QueryError {
    fn from(error: numr::error::Error) -> Self {
        Self::Arithmetic(error.into())
    }
}

pub(super) struct Inputs<R: Runtime<DType = DType>> {
    pub owners: Tensor<R>,
    pub values: Tensor<R>,
    pub wet: Tensor<R>,
    pub dry: Tensor<R>,
}

pub(super) fn metadata<R: Runtime<DType = DType>>(
    context: &Context<R>,
    owners: &Tensor<R>,
    values: &Tensor<R>,
) -> Result<usize, CrossSectionError> {
    let device = context.source.client.device();
    if owners.ndim() != 1
        || owners.dtype() != DType::I64
        || values.shape() != owners.shape()
        || values.dtype() != DType::F64
        || !owners.device().is_same(device)
        || !values.device().is_same(device)
    {
        return Err(CrossSectionError::input(format!(
            "prismatic owners shape={:?}, dtype={:?}, device={}, values shape={:?}, dtype={:?}, device={}: supply matching rank-one I64 owners and F64 operands on device={}",
            owners.shape(),
            owners.dtype(),
            owners.device().id(),
            values.shape(),
            values.dtype(),
            values.device().id(),
            device.id()
        )));
    }
    let count = owners.numel();
    config::check_work(count, context.maximum_bands)?;
    count
        .checked_mul(std::mem::size_of::<f64>())
        .ok_or_else(|| {
            CrossSectionError::input("prismatic input bytes overflow usize. Split the query batch")
        })?;
    Ok(count)
}

pub(super) fn checked<'a, R, T, F>(
    context: &'a Context<R>,
    stage: &'static str,
    build: F,
) -> Result<T, CrossSectionError>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
    F: FnOnce(&mut Arithmetic<'a, R>) -> QueryResult<T>,
{
    let mut rejection = None;
    let result = with_interval_arithmetic::<R, _, _, _>(
        &context.source.client,
        DType::F64,
        stage,
        |arithmetic| match build(arithmetic) {
            Ok(value) => Ok(Some(value)),
            Err(QueryError::Arithmetic(error)) => Err(error),
            Err(QueryError::Geometry(error)) => {
                rejection = Some(error);
                Ok(None)
            }
        },
    )
    .map_err(arithmetic_error)?;
    if let Some(error) = rejection {
        return Err(error);
    }
    result.ok_or_else(|| {
        CrossSectionError::numerical(
            "prismatic arithmetic returned no query. Rebuild the complete query",
        )
    })
}

pub(super) fn capture<R>(
    context: &Context<R>,
    owners: &Tensor<R>,
    values: &Tensor<R>,
    quantity: &'static str,
) -> QueryResult<Inputs<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    let c = &context.source.client;
    let owned_owners = Tensor::empty(owners.shape(), DType::I64, c.device())?;
    let owned_values = Tensor::empty(values.shape(), DType::F64, c.device())?;
    c.copy_into(&owned_owners, owners)?;
    c.copy_into(&owned_values, values)?;
    if owners.numel() == 0 {
        return Ok(Inputs {
            owners: owned_owners,
            values: owned_values,
            wet: Tensor::zeros(&[0], DType::F64, c.device())?,
            dry: Tensor::zeros(&[0], DType::F64, c.device())?,
        });
    }
    let zero_owner = Tensor::from_slice(&[0_i64], &[], c.device())?;
    let section_count = i64::try_from(context.source.definitions.len()).map_err(|_| {
        CrossSectionError::input(
            "prismatic section count exceeds I64 owners. Split the section batch",
        )
    })?;
    let limit = Tensor::from_slice(&[section_count], &[], c.device())?;
    let invalid_owner = c.maximum(
        &c.lt(&owned_owners, &zero_owner)?,
        &c.ge(&owned_owners, &limit)?,
    )?;
    if let Some(index) = context
        .source
        .rejected_lane(&c.cast(&invalid_owner, DType::F64)?)?
    {
        let owner = owned_owners.narrow(0, index, 1)?.item::<i64>()?;
        return Err(CrossSectionError::input(format!(
            "prismatic owner={owner}, query={index}, section_count={section_count}: supply an owner in 0..{section_count}"
        )).into());
    }
    let invalid_value = c.maximum(
        &c.maximum(
            &c.cast(&c.isnan(&owned_values)?, DType::F64)?,
            &c.cast(&c.isinf(&owned_values)?, DType::F64)?,
        )?,
        &c.lt(&owned_values, &context.source.zero)?,
    )?;
    if let Some(index) = context.source.rejected_lane(&invalid_value)? {
        let value = owned_values.narrow(0, index, 1)?.item::<f64>()?;
        let owner = source_owner(context, &owned_owners, index)?;
        return Err(CrossSectionError::input(format!(
            "prismatic {quantity}={value:?}, bits=0x{:016x}, query={index}: supply finite nonnegative operands",
            value.to_bits()
        )).at(owner).into());
    }
    let wet = c.gt(&owned_values, &context.source.zero)?;
    let dry = c.eq(&owned_values, &context.source.zero)?;
    Ok(Inputs {
        owners: owned_owners,
        values: owned_values,
        wet,
        dry,
    })
}

pub(super) fn reject_cap<R>(
    context: &Context<R>,
    inputs: &Inputs<R>,
    mask: &Tensor<R>,
    cap: &Interval<R>,
    quantity: &str,
    numerical: bool,
) -> QueryResult<()>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    if let Some(index) = context.source.rejected_lane(mask)? {
        let owner = source_owner(context, &inputs.owners, index)?;
        let value = inputs.values.narrow(0, index, 1)?.item::<f64>()?;
        let lower = cap.lower().narrow(0, index, 1)?.item::<f64>()?;
        let upper = cap.upper().narrow(0, index, 1)?.item::<f64>()?;
        let condition = if numerical {
            "cap membership is unresolved"
        } else {
            "the requested operand exceeds surveyed support"
        };
        let action = if numerical {
            "Keep the original operand and review profile conditioning"
        } else {
            "Check the original operand units or extend the surveyed profile"
        };
        let message = format!(
            "prismatic requested_{quantity}={value:?}, certified_cap_{quantity}=[{lower:?},{upper:?}], query={index}: {condition}. {action}"
        );
        let error = if numerical {
            CrossSectionError::numerical(message)
        } else {
            CrossSectionError::input(message)
        };
        return Err(error.at(owner).into());
    }
    Ok(())
}

pub(super) fn reject<R>(
    context: &Context<R>,
    owners: &Tensor<R>,
    mask: &Tensor<R>,
    numerical: bool,
    requirement: &str,
) -> QueryResult<()>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    if let Some(index) = context.source.rejected_lane(mask)? {
        let owner = source_owner(context, owners, index)?;
        let message = format!("prismatic query={index}: {requirement}");
        let error = if numerical {
            CrossSectionError::numerical(message)
        } else {
            CrossSectionError::input(message)
        };
        return Err(error.at(owner).into());
    }
    Ok(())
}

fn source_owner<R: Runtime<DType = DType>>(
    context: &Context<R>,
    owners: &Tensor<R>,
    index: usize,
) -> QueryResult<usize> {
    let original = owners.narrow(0, index, 1)?.item::<i64>()?;
    let count = context.source.definitions.len();
    usize::try_from(original).ok().filter(|owner| *owner < count).ok_or_else(|| {
        CrossSectionError::numerical(format!(
            "prismatic captured owner={original}, query={index}, section_count={count}: keep query tensors immutable and rebuild the query"
        )).into()
    })
}
