//! Per-trial port ghost states from reach end states.

use super::super::Solver;
use super::ports::{PortLayout, SurfacePorts};
use crate::contract::EngineError;
use crate::numerics::TensorFaceState;
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, IndexingOps, ReduceOps, ScalarOps, UnaryOps,
};
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use std::sync::Arc;

/// Per-trial ghost states on every port face, from the reach ends' accepted starts.
#[doc(hidden)]
#[derive(Clone)]
pub struct PortGhosts<R: Runtime<DType = DType>> {
    layout: Arc<PortLayout>,
    pub(in crate::solver) state: TensorFaceState<R>,
}

impl<R: Runtime<DType = DType>> PortGhosts<R> {
    /// Returns the ghost depth on each port face lane, in m.
    pub fn face_depth_m(&self) -> &Tensor<R> {
        &self.state.h
    }

    /// Returns the ghost normal discharge per width along each face's positive axis, in m²/s.
    pub fn face_normal_discharge_m2_per_s(&self) -> &Tensor<R> {
        &self.state.q_normal
    }

    /// Returns the ghost tangential discharge per width on each face lane, in m²/s. It is zero.
    pub fn face_tangential_discharge_m2_per_s(&self) -> &Tensor<R> {
        &self.state.q_tangent
    }

    /// Returns the ghost bed elevation on each face lane, in m.
    pub fn face_bed_m(&self) -> &Tensor<R> {
        &self.state.bed
    }

    /// Rejects ghosts built for another port set.
    pub(in crate::solver) fn check(&self, ports: &SurfacePorts<R>) -> Result<(), EngineError> {
        if Arc::ptr_eq(&self.layout, ports.layout()) {
            return Ok(());
        }
        Err(EngineError::rejected(
            "port ghosts",
            "ghosts belong to another port set: build the ghosts from these ports",
        ))
    }
}

impl<R: Runtime<DType = DType>> Solver<R>
where
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + IndexingOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + UnaryOps<R>,
{
    /// Builds ghost states for `ports` from per-port reach end states.
    ///
    /// `depth_m` holds the end cell depth `A / b`. `discharge_per_width_m2_per_s` holds
    /// `Q / b` along the increasing reach coordinate. Both have one F64 lane per port.
    ///
    /// # Errors
    /// Rejects ports from another engine, a wrong shape or dtype, nonfinite values, a negative
    /// depth, and discharge at zero depth.
    #[doc(hidden)]
    pub fn coupling_port_ghosts(
        &self,
        ports: &SurfacePorts<R>,
        depth_m: &Tensor<R>,
        discharge_per_width_m2_per_s: &Tensor<R>,
    ) -> Result<PortGhosts<R>, EngineError> {
        ports.check(self)?;
        let reject = |reason: String| EngineError::rejected("port ghosts", reason);
        for (name, tensor) in [
            ("depth_m", depth_m),
            ("discharge_per_width_m2_per_s", discharge_per_width_m2_per_s),
        ] {
            if tensor.shape() != [ports.len()] || tensor.dtype() != DType::F64 {
                return Err(reject(format!(
                    "{name}: shape={:?}, dtype={:?}, ports={}: supply one F64 lane per port",
                    tensor.shape(),
                    tensor.dtype(),
                    ports.len()
                )));
            }
        }
        let client = &self.context.client;
        let depth = depth_m;
        let discharge = discharge_per_width_m2_per_s;
        let zero = client.mul_scalar(depth, 0.0)?;
        let finite = |values: &Tensor<R>| -> Result<Tensor<R>, EngineError> {
            Ok(client.le(&client.abs(values)?, &client.add_scalar(&zero, f64::MAX)?)?)
        };
        let mut admitted = client.mul(&finite(depth)?, &finite(discharge)?)?;
        admitted = client.mul(&admitted, &client.ge(depth, &zero)?)?;
        admitted = client.mul(
            &admitted,
            &client.maximum(&client.gt(depth, &zero)?, &client.eq(discharge, &zero)?)?,
        )?;
        if client.min(&admitted, &[0], false)?.item::<f64>()? != 1.0 {
            let flags = admitted.try_to_vec::<f64>()?;
            let depth = depth.try_to_vec::<f64>()?;
            let discharge = discharge.try_to_vec::<f64>()?;
            let port = flags.iter().position(|flag| *flag != 1.0).unwrap_or(0);
            return Err(reject(format!(
                "port={port}, depth_m={:?}, discharge_per_width_m2_per_s={:?}: supply finite nonnegative depth and zero discharge at zero depth",
                depth.get(port),
                discharge.get(port)
            )));
        }
        Ok(PortGhosts {
            layout: Arc::clone(ports.layout()),
            state: ports.faces.ghost_states(client, depth, discharge)?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::super::fixture::{TestResult, ported_grid};
    use super::super::{PortReachEnd, SurfacePortSpec};
    use crate::contract::{Axis, BoundaryFace};
    use numr::runtime::cpu::{CpuDevice, CpuRuntime};
    use numr::tensor::Tensor;

    fn x_face(column: usize, row: usize) -> BoundaryFace {
        BoundaryFace {
            axis: Axis::X,
            column,
            row,
        }
    }

    fn y_face(column: usize, row: usize) -> BoundaryFace {
        BoundaryFace {
            axis: Axis::Y,
            column,
            row,
        }
    }

    fn port(faces: Vec<BoundaryFace>, bed: f64, end: PortReachEnd) -> SurfacePortSpec {
        SurfacePortSpec {
            faces,
            ghost_bed_m: bed,
            reach_end: end,
        }
    }

    #[test]
    fn ghost_states_follow_face_side_and_reach_end() -> TestResult {
        let solver = ported_grid()?;
        // West edge: ghost left. East edge: ghost right. Below the inactive cell: ghost right.
        let ports = solver.coupling_ports(&[
            port(vec![x_face(0, 1), x_face(0, 0)], -0.25, PortReachEnd::Right),
            port(vec![x_face(3, 0)], 0.5, PortReachEnd::Right),
            port(vec![y_face(1, 1)], 1.0, PortReachEnd::Left),
        ])?;
        assert_eq!(ports.len(), 3);
        assert_eq!(ports.face_lanes(), 4);
        assert_eq!(ports.faces(0), Some(&[x_face(0, 0), x_face(0, 1)][..]));
        assert_eq!(ports.ghost_left(0), Some(true));
        assert_eq!(ports.ghost_left(1), Some(false));
        assert_eq!(ports.ghost_left(2), Some(false));
        let device = CpuDevice::new();
        let lanes = |values: &[f64]| Tensor::<CpuRuntime>::from_slice(values, &[3], &device);
        let ghosts = solver.coupling_port_ghosts(
            &ports,
            &lanes(&[0.75, 0.0, 2.5])?,
            &lanes(&[0.3, 0.0, -1.5])?,
        )?;
        let bits = |tensor: &Tensor<CpuRuntime>| -> Result<Vec<u64>, Box<dyn std::error::Error>> {
            Ok(tensor
                .try_to_vec::<f64>()?
                .iter()
                .map(|value| value.to_bits())
                .collect())
        };
        let expected =
            |values: &[f64]| -> Vec<u64> { values.iter().map(|value| value.to_bits()).collect() };
        assert_eq!(
            bits(ghosts.face_depth_m())?,
            expected(&[0.75, 0.75, 0.0, 2.5])
        );
        // A right end over a left ghost keeps the sign, as a left end over a right ghost does.
        // A right end over a right ghost negates, so its zero discharge turns to `-0.0`.
        assert_eq!(
            bits(ghosts.face_normal_discharge_m2_per_s())?,
            expected(&[0.3, 0.3, -0.0, -1.5])
        );
        assert_eq!(
            bits(ghosts.face_tangential_discharge_m2_per_s())?,
            expected(&[0.0; 4])
        );
        assert_eq!(
            bits(ghosts.face_bed_m())?,
            expected(&[-0.25, -0.25, 0.5, 1.0])
        );

        // A left end over a left ghost negates, as a right end over a right ghost does.
        let one = |value: f64| Tensor::<CpuRuntime>::from_slice(&[value], &[1], &device);
        for spec in [
            port(vec![x_face(0, 0)], 0.0, PortReachEnd::Left),
            port(vec![x_face(3, 1)], 0.0, PortReachEnd::Right),
        ] {
            let flipped = solver.coupling_ports(&[spec])?;
            let ghosts = solver.coupling_port_ghosts(&flipped, &one(1.0)?, &one(0.4)?)?;
            assert_eq!(
                bits(ghosts.face_normal_discharge_m2_per_s())?,
                expected(&[-0.4])
            );
        }
        Ok(())
    }

    #[test]
    fn ghost_values_reject() -> TestResult {
        let solver = ported_grid()?;
        let spec = || port(vec![x_face(0, 0)], 0.0, PortReachEnd::Right);
        let ports = solver.coupling_ports(&[spec()])?;
        let device = CpuDevice::new();
        let one = |value: f64| Tensor::<CpuRuntime>::from_slice(&[value], &[1], &device);
        for (depth, discharge) in [
            (-0.1, 0.0),
            (f64::NAN, 0.0),
            (1.0, f64::INFINITY),
            (0.0, 0.2),
        ] {
            let message = solver
                .coupling_port_ghosts(&ports, &one(depth)?, &one(discharge)?)
                .err()
                .map(|error| error.to_string())
                .unwrap_or_default();
            assert!(
                message.contains("zero discharge at zero depth"),
                "{message}"
            );
        }
        let two = Tensor::<CpuRuntime>::from_slice(&[1.0, 1.0], &[2], &device)?;
        let message = solver
            .coupling_port_ghosts(&ports, &two, &two)
            .err()
            .map(|error| error.to_string())
            .unwrap_or_default();
        assert!(message.contains("one F64 lane per port"), "{message}");
        let other = solver.coupling_ports(&[spec()])?;
        let ghosts = solver.coupling_port_ghosts(&other, &one(1.0)?, &one(0.0)?)?;
        let message = ghosts
            .check(&ports)
            .err()
            .map(|error| error.to_string())
            .unwrap_or_default();
        assert!(message.contains("another port set"), "{message}");
        Ok(())
    }
}
