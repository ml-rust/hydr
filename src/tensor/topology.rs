use crate::contract::{Axis, BoundaryFace, EngineError};
use crate::model::Domain;
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

#[derive(Clone, Copy, Debug)]
pub(crate) struct FaceSupport {
    pub face: BoundaryFace,
    pub left: Option<usize>,
    pub right: Option<usize>,
}

#[derive(Debug)]
pub(crate) struct TensorTopology<R: Runtime<DType = DType>> {
    pub faces: Vec<FaceSupport>,
    pub left_ids: Tensor<R>,
    pub right_ids: Tensor<R>,
    pub left_sign: Tensor<R>,
    pub right_sign: Tensor<R>,
    pub left_normal_sign: Tensor<R>,
    pub right_normal_sign: Tensor<R>,
    pub x_mask: Tensor<R>,
    pub widths: Tensor<R>,
}

impl<R: Runtime<DType = DType>> TensorTopology<R> {
    pub(crate) fn new(domain: &Domain, device: &R::Device) -> Result<Self, EngineError> {
        let (nx, ny) = domain.grid().dimensions();
        let mut faces = Vec::new();
        let mut add = |face, left: Option<usize>, right: Option<usize>| {
            let left = left.filter(|&cell| domain.active()[cell]);
            let right = right.filter(|&cell| domain.active()[cell]);
            if left.is_some() || right.is_some() {
                faces.push(FaceSupport { face, left, right });
            }
        };
        for row in 0..ny {
            for column in 0..=nx {
                add(
                    BoundaryFace {
                        axis: Axis::X,
                        column,
                        row,
                    },
                    column.checked_sub(1).map(|column| row * nx + column),
                    (column < nx).then(|| row * nx + column),
                );
            }
        }
        for row in 0..=ny {
            for column in 0..nx {
                add(
                    BoundaryFace {
                        axis: Axis::Y,
                        column,
                        row,
                    },
                    row.checked_sub(1).map(|row| row * nx + column),
                    (row < ny).then(|| row * nx + column),
                );
            }
        }
        let mut left_ids = Vec::with_capacity(faces.len());
        let mut right_ids = Vec::with_capacity(faces.len());
        for support in &faces {
            let index = |cell: Option<usize>| {
                cell.ok_or_else(|| {
                    EngineError::rejected(
                        "tensor topology",
                        "face lacks active support: check topology",
                    )
                })
                .and_then(|cell| {
                    i64::try_from(cell).map_err(|_| {
                        EngineError::rejected(
                            "tensor topology",
                            format!("cell index {cell} exceeds i64: reduce grid dimensions"),
                        )
                    })
                })
            };
            left_ids.push(index(support.left.or(support.right))?);
            right_ids.push(index(support.right.or(support.left))?);
        }
        let tensor = |values: Vec<f64>| {
            Tensor::<R>::from_slice(&values, &[faces.len()], device).map_err(EngineError::from)
        };
        let (dx, dy) = domain.grid().spacing();
        Ok(Self {
            left_ids: Tensor::<R>::from_slice(&left_ids, &[faces.len()], device)?,
            right_ids: Tensor::<R>::from_slice(&right_ids, &[faces.len()], device)?,
            left_sign: tensor(
                faces
                    .iter()
                    .map(|f| if f.left.is_some() { -1.0 } else { 0.0 })
                    .collect(),
            )?,
            right_sign: tensor(
                faces
                    .iter()
                    .map(|f| if f.right.is_some() { 1.0 } else { 0.0 })
                    .collect(),
            )?,
            left_normal_sign: tensor(
                faces
                    .iter()
                    .map(|f| if f.left.is_some() { 1.0 } else { -1.0 })
                    .collect(),
            )?,
            right_normal_sign: tensor(
                faces
                    .iter()
                    .map(|f| if f.right.is_some() { 1.0 } else { -1.0 })
                    .collect(),
            )?,
            x_mask: tensor(
                faces
                    .iter()
                    .map(|f| if f.face.axis == Axis::X { 1.0 } else { 0.0 })
                    .collect(),
            )?,
            widths: tensor(
                faces
                    .iter()
                    .map(|f| if f.face.axis == Axis::X { dy } else { dx })
                    .collect(),
            )?,
            faces,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::CartesianGrid;
    use crate::model::SpatialReference;
    use numr::runtime::cpu::{CpuDevice, CpuRuntime};

    #[test]
    fn inactive_cells_supply_reflected_walls() -> Result<(), Box<dyn std::error::Error>> {
        let domain = Domain::new(
            CartesianGrid::new(2, 1, 2.0, 3.0)?,
            vec![0.0; 2],
            vec![0.0; 2],
            vec![true, false],
            SpatialReference::new("projected".into(), "datum".into(), [0.0; 2])?,
        )?;
        let topology = TensorTopology::<CpuRuntime>::new(&domain, &CpuDevice::new())?;
        assert_eq!(topology.faces.len(), 4);
        assert_eq!(topology.left_ids.try_to_vec::<i64>()?, vec![0; 4]);
        assert_eq!(topology.right_ids.try_to_vec::<i64>()?, vec![0; 4]);
        assert_eq!(
            topology.left_sign.try_to_vec::<f64>()?,
            vec![0.0, -1.0, 0.0, -1.0]
        );
        assert_eq!(
            topology.right_sign.try_to_vec::<f64>()?,
            vec![1.0, 0.0, 1.0, 0.0]
        );
        assert_eq!(
            topology.left_normal_sign.try_to_vec::<f64>()?,
            vec![-1.0, 1.0, -1.0, 1.0]
        );
        assert_eq!(
            topology.right_normal_sign.try_to_vec::<f64>()?,
            vec![1.0, -1.0, 1.0, -1.0]
        );
        assert_eq!(
            topology.x_mask.try_to_vec::<f64>()?,
            vec![1.0, 1.0, 0.0, 0.0]
        );
        assert_eq!(
            topology.widths.try_to_vec::<f64>()?,
            vec![3.0, 3.0, 2.0, 2.0]
        );
        assert_eq!(topology.faces[1].face.column, 1);
        assert_eq!(topology.faces[2].face.axis, Axis::Y);
        Ok(())
    }
}
