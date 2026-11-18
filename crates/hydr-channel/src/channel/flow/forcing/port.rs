//! Device-resident flux port operands.

use super::super::{
    ChannelError, ChannelFluxPort,
    flux_port::{check_port_provenance, check_port_values},
};
use super::admission::check_scalar;
use numr::dtype::DType;
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Flux port operands held as F64 tensors of shape `[1]` on the reach device.
///
/// The fields match [`ChannelFluxPort`]. Binding checks every value on the reach device.
#[derive(Clone, Debug)]
pub struct ChannelTensorFluxPort<R: Runtime<DType = DType>> {
    outward_water_rate_m3_per_s: Tensor<R>,
    outward_momentum_rate_m4_per_s2: Tensor<R>,
    signal_speed_m_per_s: Tensor<R>,
    exterior_invariant_m_per_s: Option<Tensor<R>>,
    provenance: String,
}

impl<R: Runtime<DType = DType>> ChannelTensorFluxPort<R> {
    /// Keeps port operands after checking each shape, dtype and the provenance text.
    ///
    /// Binding to a reach checks the values, as [`ChannelFluxPort::new`] does.
    pub fn new(
        outward_water_rate_m3_per_s: Tensor<R>,
        outward_momentum_rate_m4_per_s2: Tensor<R>,
        signal_speed_m_per_s: Tensor<R>,
        exterior_invariant_m_per_s: Option<Tensor<R>>,
        provenance: impl Into<String>,
    ) -> Result<Self, ChannelError> {
        let port = Self {
            outward_water_rate_m3_per_s,
            outward_momentum_rate_m4_per_s2,
            signal_speed_m_per_s,
            exterior_invariant_m_per_s,
            provenance: provenance.into(),
        };
        for (name, tensor) in port.operands() {
            if tensor.shape() != [1] || tensor.dtype() != DType::F64 {
                return Err(ChannelError::input(format!(
                    "{name} shape={:?}, dtype={:?}: supply an F64 scalar tensor of shape [1]",
                    tensor.shape(),
                    tensor.dtype()
                )));
            }
        }
        check_port_provenance(&port.provenance)?;
        Ok(port)
    }

    /// Water leaving the reach through the end face, in m³/s. A negative rate enters.
    pub fn outward_water_rate_m3_per_s(&self) -> &Tensor<R> {
        &self.outward_water_rate_m3_per_s
    }

    /// Normal momentum flux along the outward normal, including normal pressure, in m⁴/s².
    pub fn outward_momentum_rate_m4_per_s2(&self) -> &Tensor<R> {
        &self.outward_momentum_rate_m4_per_s2
    }

    /// Conservative absolute bound on the face signal speeds, in m/s.
    pub fn signal_speed_m_per_s(&self) -> &Tensor<R> {
        &self.signal_speed_m_per_s
    }

    /// Into-reach Riemann invariant `u + φ` of the exterior water, in m/s. `None` when dry.
    pub fn exterior_invariant_m_per_s(&self) -> Option<&Tensor<R>> {
        self.exterior_invariant_m_per_s.as_ref()
    }

    /// Provider, compatible state, and basis of the speed bound.
    pub fn provenance(&self) -> &str {
        &self.provenance
    }

    /// Named operands in a fixed order. The invariant comes last when present.
    pub(in crate::channel::flow) fn operands(&self) -> Vec<(&'static str, &Tensor<R>)> {
        let mut operands = vec![
            (
                "flux port outward water rate",
                &self.outward_water_rate_m3_per_s,
            ),
            (
                "flux port outward momentum rate",
                &self.outward_momentum_rate_m4_per_s2,
            ),
            ("flux port signal speed", &self.signal_speed_m_per_s),
        ];
        if let Some(invariant) = &self.exterior_invariant_m_per_s {
            operands.push(("flux port exterior invariant", invariant));
        }
        operands
    }

    pub(in crate::channel::flow) fn from_host(
        client: &R::Client,
        port: &ChannelFluxPort,
    ) -> Result<Self, ChannelError> {
        let make = |value: f64| Tensor::from_slice(&[value], &[1], client.device());
        Ok(Self {
            outward_water_rate_m3_per_s: make(port.outward_water_rate_m3_per_s())?,
            outward_momentum_rate_m4_per_s2: make(port.outward_momentum_rate_m4_per_s2())?,
            signal_speed_m_per_s: make(port.signal_speed_m_per_s())?,
            exterior_invariant_m_per_s: port.exterior_invariant_m_per_s().map(make).transpose()?,
            provenance: port.provenance().to_owned(),
        })
    }

    pub(in crate::channel::flow) fn project(&self) -> Result<ChannelFluxPort, ChannelError> {
        ChannelFluxPort::new(
            self.outward_water_rate_m3_per_s.item::<f64>()?,
            self.outward_momentum_rate_m4_per_s2.item::<f64>()?,
            self.signal_speed_m_per_s.item::<f64>()?,
            self.exterior_invariant_m_per_s
                .as_ref()
                .map(|value| value.item::<f64>())
                .transpose()?,
            self.provenance.clone(),
        )
    }

    /// Checks placement and every value of the held operands.
    pub(in crate::channel::flow) fn check(&self, client: &R::Client) -> Result<(), ChannelError> {
        for (name, tensor) in self.operands() {
            check_scalar(client, tensor, name)?;
        }
        check_port_values(
            self.outward_water_rate_m3_per_s.item::<f64>()?,
            self.outward_momentum_rate_m4_per_s2.item::<f64>()?,
            self.signal_speed_m_per_s.item::<f64>()?,
            self.exterior_invariant_m_per_s
                .as_ref()
                .map(|value| value.item::<f64>())
                .transpose()?,
        )?;
        check_port_provenance(&self.provenance)
    }

    /// Copies every operand into storage the reach owns.
    pub(in crate::channel::flow) fn into_owned(
        self,
        detach: impl Fn(&Tensor<R>, &str) -> Result<Tensor<R>, ChannelError>,
    ) -> Result<Self, ChannelError> {
        Ok(Self {
            outward_water_rate_m3_per_s: detach(
                &self.outward_water_rate_m3_per_s,
                "flux port outward water rate",
            )?,
            outward_momentum_rate_m4_per_s2: detach(
                &self.outward_momentum_rate_m4_per_s2,
                "flux port outward momentum rate",
            )?,
            signal_speed_m_per_s: detach(&self.signal_speed_m_per_s, "flux port signal speed")?,
            exterior_invariant_m_per_s: self
                .exterior_invariant_m_per_s
                .as_ref()
                .map(|value| detach(value, "flux port exterior invariant"))
                .transpose()?,
            provenance: self.provenance,
        })
    }
}
