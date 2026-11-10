//! Flat tensor-record selection and explicit projection.

use super::CpuActuatorContext;
use crate::cpu::CpuTensor;
use crate::structures::StructureError;
use numr::ops::ConditionalOps;
use std::sync::Arc;

pub(crate) trait TensorRecord: Clone {
    type Values;
    fn zero(context: &CpuActuatorContext) -> Self;
    fn select(
        context: &CpuActuatorContext,
        mask: &CpuTensor,
        yes: &Self,
        no: &Self,
    ) -> Result<Self, StructureError>;
    fn project(&self) -> Result<Self::Values, StructureError>;
    fn from_values(
        context: &CpuActuatorContext,
        values: &Self::Values,
    ) -> Result<Self, StructureError>;
}
impl TensorRecord for CpuTensor {
    type Values = Vec<f64>;
    fn zero(context: &CpuActuatorContext) -> Self {
        context.zero.clone()
    }
    fn select(
        context: &CpuActuatorContext,
        mask: &CpuTensor,
        yes: &Self,
        no: &Self,
    ) -> Result<Self, StructureError> {
        Ok(context.structures.client.where_cond(mask, yes, no)?)
    }
    fn project(&self) -> Result<Self::Values, StructureError> {
        Ok(self.try_to_vec::<f64>()?)
    }
    fn from_values(
        context: &CpuActuatorContext,
        values: &Self::Values,
    ) -> Result<Self, StructureError> {
        context.tensor(values)
    }
}
// Indirection changes ownership only. Each operation retains the record's original field order.
impl<T: TensorRecord> TensorRecord for Arc<T> {
    type Values = T::Values;
    fn zero(context: &CpuActuatorContext) -> Self {
        Arc::new(T::zero(context))
    }
    fn select(
        context: &CpuActuatorContext,
        mask: &CpuTensor,
        yes: &Self,
        no: &Self,
    ) -> Result<Self, StructureError> {
        Ok(Arc::new(T::select(
            context,
            mask,
            yes.as_ref(),
            no.as_ref(),
        )?))
    }
    fn project(&self) -> Result<Self::Values, StructureError> {
        self.as_ref().project()
    }
    fn from_values(
        context: &CpuActuatorContext,
        values: &Self::Values,
    ) -> Result<Self, StructureError> {
        Ok(Arc::new(T::from_values(context, values)?))
    }
}
macro_rules! record {
    ($name:ident, $values:ident { $($field:ident: $ty:ty),* $(,)? }) => {
        #[derive(Clone, Debug)]
        pub(crate) struct $name { $(pub $field: $ty,)* }
        pub(crate) struct $values { $(pub $field: <$ty as $crate::cpu::structures::actuator::record::TensorRecord>::Values,)* }
        impl $crate::cpu::structures::actuator::record::TensorRecord for $name {
            type Values = $values;
            fn zero(context: &$crate::cpu::structures::actuator::CpuActuatorContext) -> Self {
                Self { $($field: <$ty as $crate::cpu::structures::actuator::record::TensorRecord>::zero(context),)* }
            }
            fn select(context: &$crate::cpu::structures::actuator::CpuActuatorContext, mask: &$crate::cpu::CpuTensor, yes: &Self, no: &Self) -> Result<Self, $crate::structures::StructureError> {
                Ok(Self { $($field: <$ty as $crate::cpu::structures::actuator::record::TensorRecord>::select(context, mask, &yes.$field, &no.$field)?,)* })
            }
            fn from_values(context: &$crate::cpu::structures::actuator::CpuActuatorContext, values: &Self::Values) -> Result<Self, $crate::structures::StructureError> {
                Ok(Self { $($field: <$ty as $crate::cpu::structures::actuator::record::TensorRecord>::from_values(context, &values.$field)?,)* })
            }
            fn project(&self) -> Result<Self::Values, $crate::structures::StructureError> {
                Ok($values { $($field: self.$field.project()?,)* })
            }
        }
    };
}
pub(crate) use record;
