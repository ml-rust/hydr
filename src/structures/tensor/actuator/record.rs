//! Flat tensor-record selection and explicit projection.

use super::TensorActuatorContext;
use crate::structures::StructureError;
use numr::ops::ConditionalOps;
use numr::{dtype::DType, runtime::Runtime, tensor::Tensor};
use std::sync::Arc;

pub(crate) trait TensorRecord<R: Runtime<DType = DType>>: Clone {
    type Values;
    fn zero(context: &TensorActuatorContext<R>) -> Self;
    fn select(
        context: &TensorActuatorContext<R>,
        mask: &Tensor<R>,
        yes: &Self,
        no: &Self,
    ) -> Result<Self, StructureError>
    where
        R::Client: ConditionalOps<R>;
    fn project(&self) -> Result<Self::Values, StructureError>;
    fn from_values(
        context: &TensorActuatorContext<R>,
        values: &Self::Values,
    ) -> Result<Self, StructureError>;
}
impl<R: Runtime<DType = DType>> TensorRecord<R> for Tensor<R> {
    type Values = Vec<f64>;
    fn zero(context: &TensorActuatorContext<R>) -> Self {
        context.zero.clone()
    }
    fn select(
        context: &TensorActuatorContext<R>,
        mask: &Tensor<R>,
        yes: &Self,
        no: &Self,
    ) -> Result<Self, StructureError>
    where
        R::Client: ConditionalOps<R>,
    {
        Ok(context.structures.client.where_cond(mask, yes, no)?)
    }
    fn project(&self) -> Result<Self::Values, StructureError> {
        Ok(self.try_to_vec::<f64>()?)
    }
    fn from_values(
        context: &TensorActuatorContext<R>,
        values: &Self::Values,
    ) -> Result<Self, StructureError> {
        context.tensor(values)
    }
}
// Indirection changes ownership only. Each operation retains the record's original field order.
impl<R: Runtime<DType = DType>, T: TensorRecord<R>> TensorRecord<R> for Arc<T> {
    type Values = T::Values;
    fn zero(context: &TensorActuatorContext<R>) -> Self {
        Arc::new(T::zero(context))
    }
    fn select(
        context: &TensorActuatorContext<R>,
        mask: &Tensor<R>,
        yes: &Self,
        no: &Self,
    ) -> Result<Self, StructureError>
    where
        R::Client: ConditionalOps<R>,
    {
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
        context: &TensorActuatorContext<R>,
        values: &Self::Values,
    ) -> Result<Self, StructureError> {
        Ok(Arc::new(T::from_values(context, values)?))
    }
}
macro_rules! record {
    ($name:ident<$runtime:ident>, $values:ident { $($field:ident: $ty:ty),* $(,)? }) => {
        #[derive(Clone, Debug)]
        pub(crate) struct $name<$runtime: numr::runtime::Runtime<DType = numr::dtype::DType>> {
            $(pub $field: $ty,)*
        }
        impl<$runtime: numr::runtime::Runtime<DType = numr::dtype::DType>>
            $crate::structures::tensor::actuator::record::TensorRecord<$runtime> for $name<$runtime>
        {
            type Values = $values;
            fn zero(context: &$crate::structures::tensor::actuator::TensorActuatorContext<$runtime>) -> Self {
                Self { $($field: <$ty as $crate::structures::tensor::actuator::record::TensorRecord<$runtime>>::zero(context),)* }
            }
            fn select(
                context: &$crate::structures::tensor::actuator::TensorActuatorContext<$runtime>,
                mask: &numr::tensor::Tensor<$runtime>, yes: &Self, no: &Self,
            ) -> Result<Self, $crate::structures::StructureError>
            where $runtime::Client: numr::ops::ConditionalOps<$runtime>
            {
                Ok(Self { $($field: <$ty as $crate::structures::tensor::actuator::record::TensorRecord<$runtime>>::select(context, mask, &yes.$field, &no.$field)?,)* })
            }
            fn from_values(
                context: &$crate::structures::tensor::actuator::TensorActuatorContext<$runtime>, values: &Self::Values,
            ) -> Result<Self, $crate::structures::StructureError> {
                Ok(Self { $($field: <$ty as $crate::structures::tensor::actuator::record::TensorRecord<$runtime>>::from_values(context, &values.$field)?,)* })
            }
            fn project(&self) -> Result<Self::Values, $crate::structures::StructureError> {
                Ok($values { $($field: self.$field.project()?,)* })
            }
        }
    };
}
pub(crate) use record;
