//! Configured persistent concrete-law masks and sum-of-logs coefficient caches.

use crate::cpu::CpuTensor;
use crate::structures::{
    DirectionPolicy, StructureElement, StructureError, StructureInputErrorKind, StructureLaw,
};
use numr::ops::{BinaryOps, CompareOps, ReduceOps, ScalarOps, TypeConversionOps, UnaryOps};
use numr::runtime::cpu::{CpuClient, CpuDevice, ParallelismConfig};
use std::collections::BTreeSet;

#[derive(Debug)]
pub(crate) struct CpuStructureContext {
    pub gate: Option<super::gate::CpuGateContext>,
    pub pump: Option<super::pump::CpuPumpContext>,
    pub elements: Vec<StructureElement>,
    pub client: CpuClient,
    pub device: CpuDevice,
    pub zero: CpuTensor,
    pub one: CpuTensor,
    pub orifice: CpuTensor,
    pub weir: CpuTensor,
    pub crown: CpuTensor,
    pub crest: CpuTensor,
    pub allow_a: CpuTensor,
    pub allow_b: CpuTensor,
    pub log_orifice_scale: CpuTensor,
    pub log_weir_scale: CpuTensor,
    pub log_gravity: CpuTensor,
    pub log_two: CpuTensor,
}
impl CpuStructureContext {
    pub(crate) fn new(elements: Vec<StructureElement>) -> Result<Self, StructureError> {
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
        let device = CpuDevice::new();
        let client =
            CpuClient::new(device.clone()).with_parallelism(ParallelismConfig::new(Some(1), None));
        if let Some(error) = crate::cpu::arithmetic::probe_f64(&client, &device)? {
            return Err(StructureError::input(
                StructureInputErrorKind::Definition,
                error.message("structures"),
            ));
        }
        let tensor = |v: &[f64]| CpuTensor::from_slice(v, &[v.len()], &device);
        let one = tensor(&vec![1.0; n])?;
        let log_orifice_scale =
            client.add(&client.log(&tensor(&cd)?)?, &client.log(&tensor(&area)?)?)?;
        let log_weir_scale =
            client.add(&client.log(&tensor(&cw)?)?, &client.log(&tensor(&length)?)?)?;
        let context = Self {
            gate: super::gate::CpuGateContext::new(&elements, &client, &device)?,
            pump: super::pump::CpuPumpContext::new(&elements, &device)?,
            zero: tensor(&vec![0.0; n])?,
            orifice: tensor(&orifice)?,
            weir: tensor(&weir)?,
            crown: tensor(&crown)?,
            crest: tensor(&crest)?,
            allow_a: tensor(&allow_a)?,
            allow_b: tensor(&allow_b)?,
            log_gravity: client.log(&tensor(&gravity)?)?,
            log_two: client.log(&client.mul_scalar(&one, 2.0)?)?,
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
    pub(crate) fn tensor(&self, values: &[f64]) -> Result<CpuTensor, StructureError> {
        Ok(CpuTensor::from_slice(
            values,
            &[values.len()],
            &self.device,
        )?)
    }
    pub(crate) fn any(&self, mask: &CpuTensor) -> Result<bool, StructureError> {
        Ok(self.client.any(mask, &[], false)?.item::<f64>()? != 0.0)
    }
    fn lane(&self, mask: &CpuTensor) -> Result<Option<usize>, StructureError> {
        if !self.any(mask)? {
            return Ok(None);
        }
        let flags = mask.try_to_vec::<f64>()?;
        let index = flags.iter().position(|v| *v != 0.0).ok_or_else(|| {
            StructureError::numerical("structure rejection mask contradicts reduction")
        })?;
        Ok(Some(index))
    }
    pub(crate) fn reject(&self, mask: &CpuTensor, quantity: &str) -> Result<(), StructureError> {
        if let Some(index) = self.lane(mask)? {
            return Err(StructureError::numerical(format!("structure={:?},{quantity}: rescale supported heads/parameters or select a representable held interval",self.elements[index].id)).at(index));
        }
        Ok(())
    }
    pub(crate) fn reject_support(
        &self,
        mask: &CpuTensor,
        quantity: &str,
    ) -> Result<(), StructureError> {
        if let Some(index) = self.lane(mask)? {
            return Err(StructureError::input(StructureInputErrorKind::UnsupportedRegime,format!("structure={:?},{quantity}: supply complete support in this law's admitted physical regime",self.elements[index].id)).at(index));
        }
        Ok(())
    }
    pub(crate) fn reject_input(
        &self,
        mask: &CpuTensor,
        kind: StructureInputErrorKind,
        quantity: &str,
    ) -> Result<(), StructureError> {
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
    pub(crate) fn finite(&self, value: &CpuTensor, quantity: &str) -> Result<(), StructureError> {
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
    pub(crate) fn positive(&self, value: &CpuTensor, quantity: &str) -> Result<(), StructureError> {
        self.finite(value, quantity)?;
        self.reject(&self.client.le(value, &self.zero)?, quantity)
    }
}
