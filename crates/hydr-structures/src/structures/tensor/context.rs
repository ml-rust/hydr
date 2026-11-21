//! Configured persistent concrete-law masks and sum-of-logs coefficient caches.
use numr::ops::BinaryOps;
use numr::ops::CompareOps;
use numr::ops::CorrectlyRoundedOps;
use numr::ops::IndexingOps;
use numr::ops::ReduceOps;
use numr::ops::ScalarOps;
use numr::ops::TypeConversionOps;
use numr::ops::UnaryOps;
use numr::{
    dtype::DType,
    runtime::{Runtime, RuntimeClient},
};

use crate::structures::{
    DirectionPolicy, StructureElement, StructureError, StructureInputErrorKind, StructureLaw,
};
use numr::tensor::Tensor;

use std::collections::BTreeSet;

pub(crate) struct TensorStructureContext<R: Runtime<DType = DType>> {
    pub culvert: Option<Box<dyn super::culvert::CulvertLaw<R>>>,
    pub gate: Option<super::gate::TensorGateContext<R>>,
    pub pump: Option<super::pump::TensorPumpContext<R>>,
    pub elements: Vec<StructureElement>,
    pub client: R::Client,
    pub device: R::Device,
    pub zero: Tensor<R>,
    pub one: Tensor<R>,
    pub orifice: Tensor<R>,
    pub weir: Tensor<R>,
    pub crown: Tensor<R>,
    pub crest: Tensor<R>,
    pub allow_a: Tensor<R>,
    pub allow_b: Tensor<R>,
    pub log_orifice_scale: Tensor<R>,
    pub log_weir_scale: Tensor<R>,
    pub log_gravity: Tensor<R>,
    pub log_two: Tensor<R>,
}
impl<R: Runtime<DType = DType>> TensorStructureContext<R> {
    /// Checks every law except the culvert, which rejects.
    pub(crate) fn new(
        client: R::Client,
        elements: Vec<StructureElement>,
    ) -> Result<Self, StructureError>
    where
        R::Client: numr::ops::BinaryOps<R>
            + numr::ops::CompareOps<R>
            + numr::ops::CorrectlyRoundedOps<R>
            + numr::ops::ReduceOps<R>
            + numr::ops::ScalarOps<R>
            + numr::ops::TypeConversionOps<R>
            + numr::ops::UnaryOps<R>,
    {
        Self::build(client, elements, false)
    }

    /// Checks every law and uploads the culvert rows.
    pub(crate) fn with_culverts(
        client: R::Client,
        elements: Vec<StructureElement>,
    ) -> Result<Self, StructureError>
    where
        R::Client: crate::structures::CulvertClient<R>,
    {
        let mut context = Self::build(client, elements, true)?;
        context.culvert = super::culvert::TensorCulvertLaw::new(&context)?
            .map(|law| Box::new(law) as Box<dyn super::culvert::CulvertLaw<R>>);
        Ok(context)
    }

    fn build(
        client: R::Client,
        elements: Vec<StructureElement>,
        culverts: bool,
    ) -> Result<Self, StructureError>
    where
        R::Client: numr::ops::BinaryOps<R>
            + numr::ops::CompareOps<R>
            + numr::ops::CorrectlyRoundedOps<R>
            + numr::ops::ReduceOps<R>
            + numr::ops::ScalarOps<R>
            + numr::ops::TypeConversionOps<R>
            + numr::ops::UnaryOps<R>,
    {
        if elements.is_empty() {
            return Err(StructureError::input(
                StructureInputErrorKind::Definition,
                "structure batch requires at least one explicit law",
            ));
        }
        let mut ids = BTreeSet::new();
        let n = elements.len();
        let (
            mut orifice,
            mut weir,
            mut crown,
            mut crest,
            mut allow_a,
            mut allow_b,
            mut cd,
            mut area,
            mut cw,
            mut length,
            mut gravity,
        ) = (
            Vec::with_capacity(n),
            Vec::with_capacity(n),
            Vec::with_capacity(n),
            Vec::with_capacity(n),
            Vec::with_capacity(n),
            Vec::with_capacity(n),
            Vec::with_capacity(n),
            Vec::with_capacity(n),
            Vec::with_capacity(n),
            Vec::with_capacity(n),
            Vec::with_capacity(n),
        );
        for (index, e) in elements.iter().enumerate() {
            for (value, field) in [
                (&e.id, "structure.id"),
                (&e.endpoint_a, "endpoint_a"),
                (&e.endpoint_b, "endpoint_b"),
            ] {
                if value.trim().is_empty() || value.chars().any(char::is_control) {
                    return Err(StructureError::input(StructureInputErrorKind::Identity,format!("{field}={value:?}: supply explicit nonempty identity without control characters")).at(index));
                }
            }
            if !ids.insert(&e.id) || e.endpoint_a == e.endpoint_b {
                return Err(StructureError::input(StructureInputErrorKind::Identity,format!("structure={:?},endpoints={:?}/{:?}: require unique structure IDs and distinct endpoints",e.id,e.endpoint_a,e.endpoint_b)).at(index));
            }
            if e.datum.trim().is_empty() || e.datum.chars().any(char::is_control) {
                return Err(StructureError::input(
                    StructureInputErrorKind::Datum,
                    "supply explicit common vertical datum",
                )
                .at(index));
            }
            let symmetric = match &e.law {
                StructureLaw::Culvert(parameters) => {
                    if !culverts {
                        return Err(StructureError::input(
                            StructureInputErrorKind::Definition,
                            "a culvert law needs certified roots and quadrature on the runtime. Build the batch with StructureBatch::with_culverts",
                        )
                        .at(index));
                    }
                    orifice.push(0.0);
                    weir.push(0.0);
                    crown.push(0.0);
                    crest.push(0.0);
                    cd.push(1.0);
                    area.push(1.0);
                    cw.push(1.0);
                    length.push(1.0);
                    gravity.push(1.0);
                    culvert_ends(parameters, e.direction).map_err(|error| error.at(index))?;
                    true
                }
                StructureLaw::HydrostaticRectangularGate(parameters) => {
                    orifice.push(0.0);
                    weir.push(0.0);
                    crown.push(0.0);
                    crest.push(0.0);
                    cd.push(1.0);
                    area.push(1.0);
                    cw.push(1.0);
                    length.push(1.0);
                    gravity.push(1.0);
                    parameters.applicability().symmetric_reversal
                }
                StructureLaw::Pump(_) => {
                    orifice.push(0.0);
                    weir.push(0.0);
                    crown.push(0.0);
                    crest.push(0.0);
                    cd.push(1.0);
                    area.push(1.0);
                    cw.push(1.0);
                    length.push(1.0);
                    gravity.push(1.0);
                    if e.direction == DirectionPolicy::Bidirectional {
                        return Err(StructureError::input(
                            StructureInputErrorKind::Definition,
                            "pump requires fixed A-to-B or B-to-A orientation",
                        )
                        .at(index));
                    }
                    false
                }
                StructureLaw::SubmergedOrifice(p) => {
                    orifice.push(1.0);
                    weir.push(0.0);
                    crown.push(p.crown_m());
                    crest.push(0.0);
                    cd.push(p.discharge_coefficient());
                    area.push(p.area_m2());
                    cw.push(1.0);
                    length.push(1.0);
                    gravity.push(p.gravity_m_per_s2());
                    p.applicability().symmetric_reversal
                }
                StructureLaw::SuppressedRectangularWeir(p) => {
                    orifice.push(0.0);
                    weir.push(1.0);
                    crown.push(0.0);
                    crest.push(p.crest_m());
                    cd.push(1.0);
                    area.push(1.0);
                    cw.push(p.discharge_coefficient());
                    length.push(p.crest_length_m());
                    gravity.push(1.0);
                    p.applicability().symmetric_reversal
                }
            };
            if e.direction == DirectionPolicy::Bidirectional && !symmetric {
                return Err(StructureError::input(
                    StructureInputErrorKind::Definition,
                    "bidirectional flow requires explicit symmetric coefficient applicability",
                )
                .at(index));
            }
            allow_a.push(if e.direction == DirectionPolicy::BtoAOnly {
                0.0
            } else {
                1.0
            });
            allow_b.push(if e.direction == DirectionPolicy::AtoBOnly {
                0.0
            } else {
                1.0
            });
        }
        drop(ids);
        let device = client.device().clone();
        if let Some(error) = crate::tensor::probe_f64::<R, _>(&client)? {
            return Err(StructureError::input(
                StructureInputErrorKind::Definition,
                error.message("structures"),
            ));
        }
        if let Some(error) = crate::tensor::probe_correctly_rounded::<R>() {
            return Err(StructureError::input(
                StructureInputErrorKind::Definition,
                error.message("structures"),
            ));
        }
        let tensor = |v: &[f64]| Tensor::<R>::from_slice(v, &[v.len()], &device);
        let one = tensor(&vec![1.0; n])?;
        let log_orifice_scale = client.add(
            &client.cr_log(&tensor(&cd)?)?,
            &client.cr_log(&tensor(&area)?)?,
        )?;
        let log_weir_scale = client.add(
            &client.cr_log(&tensor(&cw)?)?,
            &client.cr_log(&tensor(&length)?)?,
        )?;
        let context = Self {
            culvert: None,
            gate: super::gate::TensorGateContext::new(&elements, &client, &device)?,
            pump: super::pump::TensorPumpContext::new(&elements, &device)?,
            zero: tensor(&vec![0.0; n])?,
            orifice: tensor(&orifice)?,
            weir: tensor(&weir)?,
            crown: tensor(&crown)?,
            crest: tensor(&crest)?,
            allow_a: tensor(&allow_a)?,
            allow_b: tensor(&allow_b)?,
            log_gravity: client.cr_log(&tensor(&gravity)?)?,
            log_two: client.cr_log(&client.mul_scalar(&one, 2.0)?)?,
            log_orifice_scale,
            log_weir_scale,
            one,
            elements,
            client,
            device,
        };
        for value in [
            &context.log_orifice_scale,
            &context.log_weir_scale,
            &context.log_gravity,
            &context.log_two,
        ] {
            context.finite(value, "cached positive coefficient/geometry logarithm")?;
        }
        if let Some(gate) = &context.gate {
            context.finite(&gate.log_scale, "cached gate coefficient/width logarithm")?;
            context.finite(&gate.log_root_gravity, "cached gate gravity logarithm")?;
        }
        Ok(context)
    }
    pub(crate) fn tensor(&self, values: &[f64]) -> Result<Tensor<R>, StructureError> {
        Ok(Tensor::<R>::from_slice(
            values,
            &[values.len()],
            &self.device,
        )?)
    }
    pub(crate) fn any(&self, mask: &Tensor<R>) -> Result<bool, StructureError>
    where
        R::Client: numr::ops::ReduceOps<R>,
    {
        Ok(self.client.any(mask, &[], false)?.item::<f64>()? != 0.0)
    }
    fn lane(&self, mask: &Tensor<R>) -> Result<Option<usize>, StructureError>
    where
        R::Client: numr::ops::ReduceOps<R> + numr::ops::IndexingOps<R>,
    {
        if !self.any(mask)? {
            return Ok(None);
        }
        let rejected = self.client.argmax(mask, 0, false)?.item::<i64>()?;
        let index = usize::try_from(rejected)
            .ok()
            .filter(|&index| index < mask.numel() && index < self.elements.len())
            .ok_or_else(|| {
                StructureError::numerical("structure rejection mask contradicts reduction")
            })?;
        Ok(Some(index))
    }
    pub(crate) fn reject(&self, mask: &Tensor<R>, quantity: &str) -> Result<(), StructureError>
    where
        R::Client: numr::ops::ReduceOps<R> + numr::ops::IndexingOps<R>,
    {
        if let Some(index) = self.lane(mask)? {
            return Err(StructureError::numerical(format!("structure={:?},{quantity}: rescale supported heads/parameters or select a representable held interval",self.elements[index].id)).at(index));
        }
        Ok(())
    }
    pub(crate) fn reject_support(
        &self,
        mask: &Tensor<R>,
        quantity: &str,
    ) -> Result<(), StructureError>
    where
        R::Client: numr::ops::ReduceOps<R> + numr::ops::IndexingOps<R>,
    {
        if let Some(index) = self.lane(mask)? {
            return Err(StructureError::input(StructureInputErrorKind::UnsupportedRegime,format!("structure={:?},{quantity}: supply complete support in this law's admitted physical regime",self.elements[index].id)).at(index));
        }
        Ok(())
    }
    pub(crate) fn reject_input(
        &self,
        mask: &Tensor<R>,
        kind: StructureInputErrorKind,
        quantity: &str,
    ) -> Result<(), StructureError>
    where
        R::Client: numr::ops::ReduceOps<R> + numr::ops::IndexingOps<R>,
    {
        if let Some(index) = self.lane(mask)? {
            return Err(StructureError::input(
                kind,
                format!(
                    "structure={:?},{quantity}: supply an admitted explicit operating point",
                    self.elements[index].id
                ),
            )
            .at(index));
        }
        Ok(())
    }
    pub(crate) fn finite(&self, value: &Tensor<R>, quantity: &str) -> Result<(), StructureError>
    where
        R::Client: numr::ops::BinaryOps<R>
            + numr::ops::ReduceOps<R>
            + numr::ops::IndexingOps<R>
            + numr::ops::TypeConversionOps<R>
            + numr::ops::UnaryOps<R>,
    {
        self.reject(
            &self.client.maximum(
                &self
                    .client
                    .cast(&self.client.isnan(value)?, value.dtype())?,
                &self
                    .client
                    .cast(&self.client.isinf(value)?, value.dtype())?,
            )?,
            quantity,
        )
    }
    pub(crate) fn positive(&self, value: &Tensor<R>, quantity: &str) -> Result<(), StructureError>
    where
        R::Client: numr::ops::BinaryOps<R>
            + numr::ops::CompareOps<R>
            + numr::ops::ReduceOps<R>
            + numr::ops::TypeConversionOps<R>
            + numr::ops::UnaryOps<R>,
    {
        self.finite(value, quantity)?;
        self.reject(&self.client.le(value, &self.zero)?, quantity)
    }
}

/// Checks that a culvert declares an end for each admitted direction and no other.
fn culvert_ends(
    parameters: &crate::structures::CulvertParameters,
    direction: DirectionPolicy,
) -> Result<(), StructureError> {
    let ends = parameters.ends();
    let (a, b) = (ends.a.is_some(), ends.b.is_some());
    let expected = match direction {
        DirectionPolicy::Bidirectional => (true, true),
        DirectionPolicy::AtoBOnly => (true, false),
        DirectionPolicy::BtoAOnly => (false, true),
    };
    if (a, b) != expected {
        return Err(StructureError::input(
            StructureInputErrorKind::Definition,
            format!(
                "direction={direction:?},end_a={a},end_b={b}: declare the culvert end of each admitted direction and no other. Bidirectional needs both ends, AtoBOnly the A end, BtoAOnly the B end"
            ),
        ));
    }
    Ok(())
}

impl<R: Runtime<DType = DType>> std::fmt::Debug for TensorStructureContext<R> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("TensorStructureContext")
            .field("elements", &self.elements)
            .finish_non_exhaustive()
    }
}
