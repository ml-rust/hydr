//! Persistent runtime soil parameters and full native state.

use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, IndexingOps, ReduceOps, ScalarOps, TypeConversionOps,
    UnaryOps,
};
use numr::runtime::{Device, Runtime, RuntimeClient};
use numr::tensor::Tensor;

use crate::soil::{ModifiedGreenAmptParameters, NativeSoilState, SoilError};

#[doc(hidden)]
pub struct TensorSoilContext<R: Runtime<DType = DType>> {
    pub client: R::Client,
    pub parameters: Vec<ModifiedGreenAmptParameters>,
    pub suction: Tensor<R>,
    pub ks: Tensor<R>,
    pub imdmax: Tensor<R>,
    pub lu: Tensor<R>,
    pub capacity: Tensor<R>,
    pub deficit_ceiling: Tensor<R>,
    pub zero: Tensor<R>,
    pub one: Tensor<R>,
}

#[derive(Debug, Clone)]
pub(crate) struct TensorSoilState<R: Runtime<DType = DType>> {
    pub imd: Tensor<R>,
    pub f: Tensor<R>,
    pub fu: Tensor<R>,
    pub sat: Tensor<R>,
    pub t: Tensor<R>,
}

impl<R> TensorSoilContext<R>
where
    R: Runtime<DType = DType>,
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + TypeConversionOps<R>
        + UnaryOps<R>,
{
    #[doc(hidden)]
    pub fn new(
        client: R::Client,
        parameters: &[ModifiedGreenAmptParameters],
    ) -> Result<Self, SoilError> {
        if parameters.is_empty() {
            return Err(SoilError::input("batch requires at least one soil cell"));
        }
        let device = client.device().clone();
        if let Some(error) = crate::tensor::probe_f64::<R, _>(&client)? {
            return Err(SoilError::input(error.message("soil")));
        }
        if let Some(error) = crate::tensor::probe_correctly_rounded::<R>() {
            return Err(SoilError::input(error.message("soil")));
        }
        let tensor = |values: Vec<f64>| Tensor::<R>::from_slice(&values, &[values.len()], &device);
        let context = Self {
            client,
            parameters: parameters.to_vec(),
            suction: tensor(parameters.iter().map(|p| p.suction).collect())?,
            ks: tensor(parameters.iter().map(|p| p.conductivity).collect())?,
            imdmax: tensor(parameters.iter().map(|p| p.deficit).collect())?,
            lu: tensor(parameters.iter().map(|p| p.lu).collect())?,
            capacity: tensor(parameters.iter().map(|p| p.capacity).collect())?,
            deficit_ceiling: tensor(parameters.iter().map(|p| p.deficit_ceiling).collect())?,
            zero: tensor(vec![0.0; parameters.len()])?,
            one: tensor(vec![1.0; parameters.len()])?,
        };
        Ok(context)
    }

    pub(crate) fn tensor(&self, values: &[f64]) -> Result<Tensor<R>, SoilError> {
        Ok(Tensor::<R>::from_slice(
            values,
            &[values.len()],
            self.zero.device(),
        )?)
    }

    pub(crate) fn any(&self, mask: &Tensor<R>) -> Result<bool, SoilError> {
        Ok(self.client.any(mask, &[], false)?.item::<f64>()? != 0.0)
    }

    pub(crate) fn finite(&self, value: &Tensor<R>, quantity: &str) -> Result<(), SoilError> {
        if value.dtype() != DType::F64 || !value.device().is_same(self.client.device()) {
            return Err(SoilError::input(format!(
                "{quantity}: require F64 tensors on the configured runtime client device"
            )));
        }
        let invalid = self.client.maximum(
            &self
                .client
                .cast(&self.client.isnan(value)?, value.dtype())?,
            &self
                .client
                .cast(&self.client.isinf(value)?, value.dtype())?,
        )?;
        self.reject_mask(
            &invalid,
            quantity,
            &[("value", value)],
            &[],
            "Rescale inputs and parameters",
        )
    }

    #[doc(hidden)]
    pub fn reject_mask(
        &self,
        mask: &Tensor<R>,
        quantity: &str,
        fields: &[(&str, &Tensor<R>)],
        scalars: &[(&str, f64)],
        action: &str,
    ) -> Result<(), SoilError> {
        if !self.any(mask)? {
            return Ok(());
        }
        let selected = self.client.argmax(mask, 0, false)?.item::<i64>()?;
        let lane = usize::try_from(selected).ok().filter(|lane| *lane < mask.numel()).ok_or_else(|| {
            SoilError::numerical(format!(
                "quantity={quantity}: rejection mask contradicts its reduction. Inspect runtime comparison kernels"
            ))
        })?;
        if mask.narrow(0, lane, 1)?.item::<f64>()? == 0.0 {
            return Err(SoilError::numerical(format!(
                "quantity={quantity}: rejection mask contradicts its reduction. Inspect runtime comparison kernels"
            )));
        }
        let parameters = self.parameters.get(lane).ok_or_else(|| {
            SoilError::numerical(format!(
                "quantity={quantity}, cell={lane}: parameter support differs. Rebuild the soil batch"
            ))
        })?;
        let mut context = format!("quantity={quantity}, cell={lane}");
        for (name, tensor) in fields {
            if lane >= tensor.numel() {
                return Err(SoilError::numerical(format!(
                    "quantity={quantity}, field={name}, cell={lane}: diagnostic support differs. Rebuild the soil batch"
                )));
            }
            let value = tensor.narrow(0, lane, 1)?.item::<f64>()?;
            context.push_str(&format!(", {name}={value:.17e}"));
        }
        for (name, value) in scalars {
            context.push_str(&format!(", {name}={value:.17e}"));
        }
        Err(
            SoilError::numerical(format!("{context}, parameters={parameters:?}: {action}"))
                .at_cell(lane),
        )
    }

    pub(crate) fn check_projection_lengths(
        &self,
        fields: &[(&str, usize)],
    ) -> Result<(), SoilError> {
        let expected = self.parameters.len();
        for &(quantity, actual) in fields {
            if actual != expected {
                return Err(SoilError::numerical(format!(
                    "quantity={quantity}, length={actual}: require {expected} soil cells. Inspect tensor output shapes"
                )));
            }
        }
        Ok(())
    }
}

impl<R> TensorSoilState<R>
where
    R: Runtime<DType = DType>,
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + TypeConversionOps<R>
        + UnaryOps<R>,
{
    pub(crate) fn from_native(
        context: &TensorSoilContext<R>,
        states: &[NativeSoilState],
    ) -> Result<Self, SoilError> {
        if states.len() != context.parameters.len() {
            return Err(SoilError::input(format!(
                "state count={}: require {} cells",
                states.len(),
                context.parameters.len()
            )));
        }
        for (index, (&state, &parameters)) in states.iter().zip(&context.parameters).enumerate() {
            state
                .check(parameters, index)
                .map_err(|error| error.at_cell(index))?;
        }
        let tensor = |values: Vec<f64>| context.tensor(&values);
        Ok(Self {
            imd: tensor(states.iter().map(|s| s.imd).collect())?,
            f: tensor(states.iter().map(|s| s.f_ft).collect())?,
            fu: tensor(states.iter().map(|s| s.fu_ft).collect())?,
            sat: tensor(
                states
                    .iter()
                    .map(|s| if s.saturated { 1.0 } else { 0.0 })
                    .collect(),
            )?,
            t: tensor(states.iter().map(|s| s.t_s).collect())?,
        })
    }

    pub(crate) fn project(
        &self,
        context: &TensorSoilContext<R>,
    ) -> Result<Vec<NativeSoilState>, SoilError> {
        let imd = self.imd.try_to_vec::<f64>()?;
        let f = self.f.try_to_vec::<f64>()?;
        let fu = self.fu.try_to_vec::<f64>()?;
        let sat = self.sat.try_to_vec::<f64>()?;
        let t = self.t.try_to_vec::<f64>()?;
        context.check_projection_lengths(&[
            ("IMD", imd.len()),
            ("F (ft)", f.len()),
            ("Fu (ft)", fu.len()),
            ("Sat", sat.len()),
            ("T (s)", t.len()),
        ])?;
        Ok(imd
            .into_iter()
            .zip(f)
            .zip(fu)
            .zip(sat)
            .zip(t)
            .map(|((((imd, f_ft), fu_ft), sat), t_s)| NativeSoilState {
                imd,
                f_ft,
                fu_ft,
                saturated: sat == 1.0,
                t_s,
            })
            .collect())
    }

    pub(crate) fn check(&self, context: &TensorSoilContext<R>) -> Result<(), SoilError> {
        for (quantity, value) in [
            ("IMD", &self.imd),
            ("F (ft)", &self.f),
            ("Fu (ft)", &self.fu),
            ("Sat", &self.sat),
            ("T (s)", &self.t),
        ] {
            context.finite(value, quantity)?;
        }
        let c = &context.client;
        let outside = c.maximum(
            &c.lt(&self.imd, &context.zero)?,
            &c.gt(&self.imd, &context.deficit_ceiling)?,
        )?;
        let outside = c.maximum(
            &outside,
            &c.maximum(
                &c.lt(&self.fu, &context.zero)?,
                &c.gt(&self.fu, &context.capacity)?,
            )?,
        )?;
        context.reject_mask(
            &outside,
            "native IMD/Fu range",
            &[
                ("IMD", &self.imd),
                ("IMD ceiling", &context.deficit_ceiling),
                ("Fu (ft)", &self.fu),
                ("capacity (ft)", &context.capacity),
            ],
            &[],
            "Inspect moisture deficit and upper-zone capacity inputs",
        )
    }
}

impl<R: Runtime<DType = DType>> std::fmt::Debug for TensorSoilContext<R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TensorSoilContext").finish_non_exhaustive()
    }
}
