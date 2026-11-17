use super::TensorTopology;
use super::boundary::TensorBoundaries;
use super::exterior::TensorExterior;
use crate::contract::{Axis, BoundaryForcing, EngineError, ForcingInterval, ForcingSchedule};
use crate::model::{CellState, Domain, FlowState};
use numr::dtype::DType;
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;
use std::collections::BTreeMap;

#[derive(Clone, Debug)]
pub(crate) struct TensorState<R: Runtime<DType = DType>> {
    pub h: Tensor<R>,
    pub qx: Tensor<R>,
    pub qy: Tensor<R>,
}

impl<R: Runtime<DType = DType>> TensorState<R> {
    pub(crate) fn from_flow(device: &R::Device, state: &FlowState) -> Result<Self, EngineError> {
        let cells = state.cells();
        let tensor = |values: Vec<f64>| {
            Tensor::<R>::from_slice(&values, &[cells.len()], device).map_err(EngineError::from)
        };
        Ok(Self {
            h: tensor(cells.iter().map(|cell| cell.h).collect())?,
            qx: tensor(cells.iter().map(|cell| cell.qx).collect())?,
            qy: tensor(cells.iter().map(|cell| cell.qy).collect())?,
        })
    }

    pub(crate) fn to_flow(&self, domain: &Domain) -> Result<FlowState, EngineError> {
        let h = self.h.try_to_vec::<f64>()?;
        let qx = self.qx.try_to_vec::<f64>()?;
        let qy = self.qy.try_to_vec::<f64>()?;
        let count = domain.grid().cell_count();
        if h.len() != count || qx.len() != count || qy.len() != count {
            return Err(EngineError::rejected(
                "tensor state",
                format!(
                    "tensor lengths [{}, {}, {}]: require {count} cells",
                    h.len(),
                    qx.len(),
                    qy.len()
                ),
            ));
        }
        let cells = h
            .into_iter()
            .zip(qx)
            .zip(qy)
            .map(|((h, qx), qy)| CellState { h, qx, qy })
            .collect();
        Ok(FlowState::new(domain, cells)?)
    }
}

pub(crate) struct TensorDomain<R: Runtime<DType = DType>> {
    pub client: R::Client,
    pub bed: Tensor<R>,
    pub manning: Tensor<R>,
    pub active: Tensor<R>,
    pub sources: Vec<TensorSources<R>>,
    pub topology: TensorTopology<R>,
    pub zero: Tensor<R>,
}

#[derive(Debug)]
pub(crate) struct TensorSources<R: Runtime<DType = DType>> {
    pub rain: Tensor<R>,
    pub potential_evaporation: Tensor<R>,
    pub withdrawal: Tensor<R>,
    pub port_mask: Tensor<R>,
    pub open_mask: Tensor<R>,
    pub trace_mask: Tensor<R>,
    pub outward_sign: Tensor<R>,
    pub characteristic: TensorBoundaries<R>,
    pub exterior: TensorExterior<R>,
    pub port_flux: [Tensor<R>; 3],
    pub port_speed: Tensor<R>,
}

impl<R: Runtime<DType = DType>> TensorDomain<R> {
    pub(crate) fn new(
        client: R::Client,
        domain: &Domain,
        forcing: &ForcingSchedule,
    ) -> Result<Self, EngineError> {
        let device = client.device();
        let topology = TensorTopology::new(domain, device)?;
        let sources = forcing
            .intervals()
            .iter()
            .map(|interval| TensorSources::new(domain, &topology, interval, device))
            .collect::<Result<_, _>>()?;
        let shape = [domain.grid().cell_count()];
        Ok(Self {
            client: client.clone(),
            bed: Tensor::<R>::from_slice(domain.bed(), &shape, device)?,
            manning: Tensor::<R>::from_slice(domain.manning(), &shape, device)?,
            sources,
            active: Tensor::<R>::from_slice(
                &domain
                    .active()
                    .iter()
                    .map(|active| if *active { 1.0 } else { 0.0 })
                    .collect::<Vec<_>>(),
                &shape,
                device,
            )?,
            topology,
            zero: Tensor::<R>::zeros(&shape, DType::F64, device)?,
        })
    }
}

impl<R: Runtime<DType = DType>> TensorSources<R> {
    fn new(
        domain: &Domain,
        topology: &TensorTopology<R>,
        forcing: &ForcingInterval,
        device: &R::Device,
    ) -> Result<Self, EngineError> {
        let boundaries: BTreeMap<_, _> = forcing
            .boundaries
            .iter()
            .map(|value| (value.face(), value))
            .collect();
        let count = topology.faces.len();
        let mut mask = vec![0.0; count];
        let mut open_mask = vec![0.0; count];
        let mut trace_mask = vec![0.0; count];
        let mut outward_sign = vec![0.0; count];
        let mut flux = [vec![0.0; count], vec![0.0; count], vec![0.0; count]];
        let mut speed = vec![0.0; count];
        let (dx, dy) = domain.grid().spacing();
        for (index, support) in topology.faces.iter().enumerate() {
            if let Some(value) = boundaries.get(&support.face) {
                let sign = if support.left.is_some() { 1.0 } else { -1.0 };
                let width = if support.face.axis == Axis::X { dy } else { dx };
                open_mask[index] = 1.0;
                outward_sign[index] = sign;
                let BoundaryForcing::Flux(port) = value else {
                    if matches!(
                        value,
                        BoundaryForcing::Stage(_) | BoundaryForcing::Discharge(_)
                    ) {
                        trace_mask[index] = 1.0;
                    }
                    continue;
                };
                for (component, rate) in [
                    port.water_rate,
                    port.momentum_rate[0],
                    port.momentum_rate[1],
                ]
                .into_iter()
                .enumerate()
                {
                    let value = sign * rate / width;
                    if !value.is_finite() || (rate != 0.0 && value == 0.0) {
                        return Err(EngineError::rejected(
                            "tensor port flux",
                            format!(
                                "face={:?}, rate={rate}, width={width}: rescale port rate or grid spacing",
                                support.face
                            ),
                        ));
                    }
                    flux[component][index] = value;
                }
                mask[index] = 1.0;
                speed[index] = port.signal_speed;
            }
        }
        let tensor = |values: &[f64]| {
            Tensor::<R>::from_slice(values, &[values.len()], device).map_err(EngineError::from)
        };
        Ok(Self {
            rain: tensor(&forcing.rain)?,
            potential_evaporation: tensor(&forcing.potential_evaporation)?,
            withdrawal: tensor(&forcing.withdrawal)?,
            port_mask: tensor(&mask)?,
            open_mask: tensor(&open_mask)?,
            trace_mask: tensor(&trace_mask)?,
            outward_sign: tensor(&outward_sign)?,
            characteristic: TensorBoundaries::new(domain, topology, forcing, device)?,
            exterior: TensorExterior::new(domain, topology, forcing, device)?,
            port_flux: [tensor(&flux[0])?, tensor(&flux[1])?, tensor(&flux[2])?],
            port_speed: tensor(&speed)?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contract::{BoundaryConditions, BoundaryFace, FluxPort};
    use crate::grid::CartesianGrid;
    use crate::model::SpatialReference;
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};

    #[test]
    fn cached_ports_preserve_outward_rates_and_pressure() -> Result<(), Box<dyn std::error::Error>>
    {
        let domain = Domain::new(
            CartesianGrid::new(1, 1, 2.0, 4.0)?,
            vec![0.0],
            vec![0.0],
            vec![true],
            SpatialReference::new("projected".into(), "datum".into(), [0.0; 2])?,
        )?;
        let face = BoundaryFace {
            axis: Axis::X,
            column: 0,
            row: 0,
        };
        let boundaries = BoundaryConditions::with_open_faces(&domain, vec![face])?;
        let forcing = ForcingSchedule::new(
            &domain,
            &boundaries,
            vec![ForcingInterval {
                start: 0.0,
                end: 1.0,
                rain: vec![0.1],
                potential_evaporation: vec![0.0],
                withdrawal: vec![0.2],
                boundaries: vec![FluxPort {
                    face,
                    water_rate: -8.0,
                    momentum_rate: [12.0, -4.0],
                    signal_speed: 5.0,
                    provenance: "declared complete flux".into(),
                }]
                .into_iter()
                .map(Into::into)
                .collect(),
            }],
        )?;
        let cached = TensorDomain::<CpuRuntime>::new(
            CpuClient::new(CpuDevice::new())
                .with_parallelism(ParallelismConfig::new(Some(1), None)),
            &domain,
            &forcing,
        )?;
        let source = &cached.sources[0];
        assert_eq!(
            source.port_mask.try_to_vec::<f64>()?,
            vec![1.0, 0.0, 0.0, 0.0]
        );
        assert_eq!(
            source.port_flux[0].try_to_vec::<f64>()?,
            vec![2.0, 0.0, 0.0, 0.0]
        );
        assert_eq!(
            source.port_flux[1].try_to_vec::<f64>()?,
            vec![-3.0, 0.0, 0.0, 0.0]
        );
        assert_eq!(
            source.port_flux[2].try_to_vec::<f64>()?,
            vec![1.0, 0.0, 0.0, 0.0]
        );
        assert_eq!(
            source.port_speed.try_to_vec::<f64>()?,
            vec![5.0, 0.0, 0.0, 0.0]
        );
        let flow = FlowState::new(
            &domain,
            vec![CellState {
                h: 1.0,
                qx: 2.0,
                qy: -3.0,
            }],
        )?;
        assert_eq!(
            TensorState::<CpuRuntime>::from_flow(&CpuDevice::new(), &flow)?.to_flow(&domain)?,
            flow
        );
        Ok(())
    }
}
