use super::CpuTopology;
use crate::model::{CellState, Domain, FlowState};
use crate::solver::{Axis, EngineError, ForcingInterval, ForcingSchedule};
use numr::dtype::DType;
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};
use numr::tensor::Tensor;
use std::collections::BTreeMap;

pub(crate) type CpuTensor = Tensor<CpuRuntime>;

#[derive(Clone, Debug)]
pub(crate) struct CpuState {
    pub h: CpuTensor,
    pub qx: CpuTensor,
    pub qy: CpuTensor,
}

impl CpuState {
    pub(crate) fn from_flow(state: &FlowState) -> Result<Self, EngineError> {
        let device = CpuDevice::new();
        let cells = state.cells();
        let tensor = |values: Vec<f64>| {
            CpuTensor::from_slice(&values, &[cells.len()], &device).map_err(EngineError::from)
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
                "CPU state",
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

#[derive(Debug)]
pub(crate) struct CpuDomain {
    pub client: CpuClient,
    pub bed: CpuTensor,
    pub manning: CpuTensor,
    pub sources: Vec<CpuSources>,
    pub topology: CpuTopology,
    pub zero: CpuTensor,
}

#[derive(Debug)]
pub(crate) struct CpuSources {
    pub rain: CpuTensor,
    pub withdrawal: CpuTensor,
    pub port_mask: CpuTensor,
    pub port_flux: [CpuTensor; 3],
    pub port_speed: CpuTensor,
    pub boundary_in_rate: f64,
    pub boundary_out_rate: f64,
}

impl CpuDomain {
    pub(crate) fn new(domain: &Domain, forcing: &ForcingSchedule) -> Result<Self, EngineError> {
        let device = CpuDevice::new();
        let topology = CpuTopology::new(domain, &device)?;
        let sources = forcing
            .intervals()
            .iter()
            .map(|interval| CpuSources::new(domain, &topology, interval, &device))
            .collect::<Result<_, _>>()?;
        let shape = [domain.grid().cell_count()];
        Ok(Self {
            client: CpuClient::new(device.clone())
                .with_parallelism(ParallelismConfig::new(Some(1), None)),
            bed: CpuTensor::from_slice(domain.bed(), &shape, &device)?,
            manning: CpuTensor::from_slice(domain.manning(), &shape, &device)?,
            sources,
            topology,
            zero: CpuTensor::zeros(&shape, DType::F64, &device)?,
        })
    }
}

impl CpuSources {
    fn new(
        domain: &Domain,
        topology: &CpuTopology,
        forcing: &ForcingInterval,
        device: &CpuDevice,
    ) -> Result<Self, EngineError> {
        let ports: BTreeMap<_, _> = forcing.ports.iter().map(|port| (port.face, port)).collect();
        let count = topology.faces.len();
        let mut mask = vec![0.0; count];
        let mut flux = [vec![0.0; count], vec![0.0; count], vec![0.0; count]];
        let mut speed = vec![0.0; count];
        let mut boundary_in_rate = 0.0;
        let mut boundary_out_rate = 0.0;
        let (dx, dy) = domain.grid().spacing();
        for (index, support) in topology.faces.iter().enumerate() {
            if let Some(port) = ports.get(&support.face) {
                let sign = if support.left.is_some() { 1.0 } else { -1.0 };
                let width = if support.face.axis == Axis::X { dy } else { dx };
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
                            "CPU port flux",
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
                boundary_in_rate += (-port.water_rate).max(0.0);
                boundary_out_rate += port.water_rate.max(0.0);
            }
        }
        if !boundary_in_rate.is_finite() || !boundary_out_rate.is_finite() {
            return Err(EngineError::rejected(
                "CPU port flux",
                "total boundary rate overflows: reduce prescribed rates",
            ));
        }
        let tensor = |values: &[f64]| {
            CpuTensor::from_slice(values, &[values.len()], device).map_err(EngineError::from)
        };
        Ok(Self {
            rain: tensor(&forcing.rain)?,
            withdrawal: tensor(&forcing.withdrawal)?,
            port_mask: tensor(&mask)?,
            port_flux: [tensor(&flux[0])?, tensor(&flux[1])?, tensor(&flux[2])?],
            port_speed: tensor(&speed)?,
            boundary_in_rate,
            boundary_out_rate,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::CartesianGrid;
    use crate::model::SpatialReference;
    use crate::solver::{BoundaryConditions, BoundaryFace, FluxPort};

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
        let boundaries = BoundaryConditions::with_ports(&domain, vec![face])?;
        let forcing = ForcingSchedule::new(
            &domain,
            &boundaries,
            vec![ForcingInterval {
                start: 0.0,
                end: 1.0,
                rain: vec![0.1],
                withdrawal: vec![0.2],
                ports: vec![FluxPort {
                    face,
                    water_rate: -8.0,
                    momentum_rate: [12.0, -4.0],
                    signal_speed: 5.0,
                    provenance: "declared complete flux".into(),
                }],
            }],
        )?;
        let cached = CpuDomain::new(&domain, &forcing)?;
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
        assert_eq!(source.boundary_in_rate, 8.0);
        assert_eq!(source.boundary_out_rate, 0.0);
        let flow = FlowState::new(
            &domain,
            vec![CellState {
                h: 1.0,
                qx: 2.0,
                qy: -3.0,
            }],
        )?;
        assert_eq!(CpuState::from_flow(&flow)?.to_flow(&domain)?, flow);
        Ok(())
    }
}
