use super::{WaterTransferError, finite, reject, upward_add};
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, IndexingOps, ReduceOps, ScalarOps, ShapeOps, TwoSumOps,
    UnaryOps,
};
use numr::runtime::{Device, Runtime, RuntimeClient};
use numr::tensor::Tensor;

#[derive(Debug)]
struct Level<R: Runtime<DType = DType>> {
    left: Tensor<R>,
    right: Tensor<R>,
}

#[derive(Clone, Copy, Debug)]
pub enum AggregationKind {
    Water,
    Momentum,
}

#[derive(Debug)]
pub struct TransferTree<R: Runtime<DType = DType>> {
    input_count: usize,
    order: Tensor<R>,
    levels: Vec<Level<R>>,
    pub degree_margin: Tensor<R>,
}

pub struct Aggregate<R: Runtime<DType = DType>> {
    pub value: Tensor<R>,
    pub bound: Tensor<R>,
}

impl<R: Runtime<DType = DType>> TransferTree<R>
where
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + IndexingOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + ShapeOps<R>
        + TwoSumOps<R>
        + UnaryOps<R>,
{
    pub fn new_grouped(
        client: &R::Client,
        group_count: usize,
        input_count: usize,
        targets: &[(usize, usize, &str)],
    ) -> Result<Self, WaterTransferError> {
        to_index(input_count)?;
        let mut groups = vec![Vec::new(); group_count];
        for &(pool, input, identity) in targets {
            if input >= input_count {
                return Err(WaterTransferError::input(
                    "transfer input exceeds declared input count",
                ));
            }
            let group = groups.get_mut(pool).ok_or_else(|| {
                WaterTransferError::input("transfer target exceeds declared group count")
            })?;
            group.push((identity, input));
        }
        let zero = input_count;
        let mut order = Vec::new();
        let mut widths = Vec::new();
        let mut margins = Vec::new();
        for group in &mut groups {
            group.sort_by(|a, b| a.0.cmp(b.0));
            let degree = group.len();
            if degree as u128 >= (1u128 << 50) {
                return Err(WaterTransferError::input(
                    "connection degree exceeds representable allocation support",
                ));
            }
            margins.push((degree as f64 + 4.0) * (2.0 * f64::EPSILON));
            if group.is_empty() {
                order.push(to_index(zero)?);
                widths.push(1);
            } else {
                for &(_, index) in group.iter() {
                    order.push(to_index(index)?);
                }
                widths.push(degree);
            }
        }
        let tensor_index =
            |values: &[i64]| Tensor::<R>::from_slice(values, &[values.len()], client.device());
        let order = tensor_index(&order)?;
        let mut levels = Vec::new();
        while widths.iter().any(|&n| n > 1) {
            let total = widths.iter().try_fold(0usize, |total, width| {
                total
                    .checked_add(*width)
                    .ok_or_else(|| WaterTransferError::input("transfer tree width overflows"))
            })?;
            let mut left = Vec::new();
            let mut right = Vec::new();
            let mut next = Vec::new();
            let mut offset = 0;
            for width in widths {
                next.push(width.div_ceil(2));
                for i in (0..width).step_by(2) {
                    left.push(to_index(offset + i)?);
                    right.push(to_index(if i + 1 < width {
                        offset + i + 1
                    } else {
                        total
                    })?);
                }
                offset += width;
            }
            levels.push(Level {
                left: tensor_index(&left)?,
                right: tensor_index(&right)?,
            });
            widths = next;
        }
        Ok(Self {
            input_count,
            order,
            levels,
            degree_margin: Tensor::from_slice(&margins, &[margins.len()], client.device())?,
        })
    }

    pub fn aggregate(
        &self,
        client: &R::Client,
        endpoint_values: &Tensor<R>,
        kind: AggregationKind,
    ) -> Result<Aggregate<R>, WaterTransferError> {
        if endpoint_values.shape() != [self.input_count]
            || endpoint_values.dtype() != DType::F64
            || !endpoint_values.device().is_same(client.device())
            || !self.order.device().is_same(client.device())
        {
            return Err(WaterTransferError::input(format!(
                "transfer input shape={:?}, dtype={:?}, expected_count={}: supply matching F64 values on the tree's configured device",
                endpoint_values.shape(),
                endpoint_values.dtype(),
                self.input_count
            )));
        }
        finite(client, endpoint_values, "transfer inputs")?;
        let zero = Tensor::<R>::zeros(&[1], DType::F64, client.device())?;
        if matches!(kind, AggregationKind::Water) && endpoint_values.numel() > 0 {
            reject(
                client,
                &client.lt(endpoint_values, &zero)?,
                "negative water contribution in transfer inputs. Supply nonnegative physical magnitudes",
            )?;
        }
        let packed = client.cat(&[endpoint_values, &zero], 0)?;
        let mut volume = client.index_select(&packed, 0, &self.order)?;
        let mut bound = Tensor::<R>::zeros(volume.shape(), DType::F64, client.device())?;
        for level in &self.levels {
            let packed = client.cat(&[&volume, &zero], 0)?;
            let errors = client.cat(&[&bound, &zero], 0)?;
            let a = client.index_select(&packed, 0, &level.left)?;
            let b = client.index_select(&packed, 0, &level.right)?;
            let (sum, low) = client.two_sum(&a, &b)?;
            finite(client, &sum, "transfer aggregation")?;
            if matches!(kind, AggregationKind::Water) {
                let positive = client.mul(&client.gt(&a, &zero)?, &client.gt(&b, &zero)?)?;
                let lost = client.mul(
                    &positive,
                    &client.maximum(&client.le(&sum, &a)?, &client.le(&sum, &b)?)?,
                )?;
                reject(
                    client,
                    &lost,
                    "positive connection volume is absorbed in aggregation. Rescale the storage or connection support",
                )?;
            }
            let a_bound = client.index_select(&errors, 0, &level.left)?;
            let b_bound = client.index_select(&errors, 0, &level.right)?;
            bound = upward_add(
                client,
                &upward_add(client, &a_bound, &b_bound)?,
                &client.abs(&low)?,
            )?;
            volume = sum;
        }
        Ok(Aggregate {
            value: volume,
            bound,
        })
    }
}

fn to_index(value: usize) -> Result<i64, WaterTransferError> {
    i64::try_from(value)
        .map_err(|_| WaterTransferError::input("connection topology exceeds runtime index support"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    fn client() -> CpuClient {
        CpuClient::new(CpuDevice::new()).with_parallelism(ParallelismConfig::new(Some(1), None))
    }

    #[test]
    fn water_absorption_rejects_while_momentum_retains_the_rounding_bound() -> TestResult {
        let client = client();
        let tree =
            TransferTree::<CpuRuntime>::new_grouped(&client, 1, 2, &[(0, 0, "a"), (0, 1, "b")])?;
        let quantum = f64::from_bits(1);
        let values = Tensor::from_slice(&[1.0, quantum], &[2], client.device())?;
        assert!(matches!(
            tree.aggregate(&client, &values, AggregationKind::Water),
            Err(WaterTransferError::Numerical(_))
        ));
        let momentum = tree.aggregate(&client, &values, AggregationKind::Momentum)?;
        assert_eq!(momentum.value.item::<f64>()?, 1.0);
        assert!(momentum.bound.item::<f64>()? >= quantum);
        Ok(())
    }

    #[test]
    fn signed_cancellation_preserves_earlier_absolute_rounding() -> TestResult {
        let client = client();
        let tree = TransferTree::<CpuRuntime>::new_grouped(
            &client,
            1,
            3,
            &[(0, 0, "a"), (0, 1, "b"), (0, 2, "c")],
        )?;
        let quantum = f64::from_bits(1);
        let values = Tensor::from_slice(&[1.0, quantum, -1.0], &[3], client.device())?;
        let momentum = tree.aggregate(&client, &values, AggregationKind::Momentum)?;
        assert_eq!(momentum.value.item::<f64>()?, 0.0);
        assert!(momentum.bound.item::<f64>()? >= quantum);
        Ok(())
    }

    #[test]
    fn identity_order_preserves_assembly_bits_and_empty_groups() -> TestResult {
        let client = client();
        let quantum = 3.0 * f64::EPSILON / 2.0;
        let original = TransferTree::<CpuRuntime>::new_grouped(
            &client,
            3,
            3,
            &[(1, 0, "b"), (1, 1, "a"), (1, 2, "c")],
        )?;
        let reordered = TransferTree::<CpuRuntime>::new_grouped(
            &client,
            3,
            3,
            &[(1, 2, "a"), (1, 0, "c"), (1, 1, "b")],
        )?;
        let values = Tensor::from_slice(&[quantum, 1.0, 0.0], &[3], client.device())?;
        let reordered_values = Tensor::from_slice(&[0.0, quantum, 1.0], &[3], client.device())?;
        let a = original.aggregate(&client, &values, AggregationKind::Water)?;
        let b = reordered.aggregate(&client, &reordered_values, AggregationKind::Water)?;
        let a_values = a.value.try_to_vec::<f64>()?;
        let b_values = b.value.try_to_vec::<f64>()?;
        assert_eq!(
            a_values.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            b_values.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
        );
        assert_eq!(a.bound.try_to_vec::<f64>()?, b.bound.try_to_vec::<f64>()?);
        assert_eq!(a_values[0], 0.0);
        assert_eq!(a_values[2], 0.0);
        assert_eq!(a.bound.try_to_vec::<f64>()?[0], 0.0);
        assert_eq!(
            original.degree_margin.try_to_vec::<f64>()?,
            vec![8.0 * f64::EPSILON, 14.0 * f64::EPSILON, 8.0 * f64::EPSILON]
        );
        Ok(())
    }

    #[test]
    fn transfer_inputs_require_declared_count_rank_and_dtype() -> TestResult {
        let client = client();
        let tree = TransferTree::<CpuRuntime>::new_grouped(&client, 1, 2, &[(0, 0, "a")])?;
        let wrong_count = Tensor::from_slice(&[1.0_f64], &[1], client.device())?;
        let wrong_rank = Tensor::from_slice(&[1.0_f64, 2.0], &[1, 2], client.device())?;
        let wrong_dtype = Tensor::from_slice(&[1.0_f32, 2.0], &[2], client.device())?;
        for values in [&wrong_count, &wrong_rank, &wrong_dtype] {
            assert!(matches!(
                tree.aggregate(&client, values, AggregationKind::Water),
                Err(WaterTransferError::Input(_))
            ));
        }
        Ok(())
    }

    #[test]
    fn strided_transfer_inputs_ignore_nonfinite_padding() -> TestResult {
        let client = client();
        let tree =
            TransferTree::<CpuRuntime>::new_grouped(&client, 1, 2, &[(0, 0, "a"), (0, 1, "b")])?;
        let values = Tensor::from_slice(
            &[f64::NAN, 1.0, f64::INFINITY, 2.0],
            &[2, 2],
            client.device(),
        )?
        .narrow(1, 1, 1)?
        .squeeze(Some(1));
        assert!(!values.is_contiguous());
        let result = tree.aggregate(&client, &values, AggregationKind::Water)?;
        assert_eq!(result.value.item::<f64>()?, 3.0);
        assert_eq!(result.bound.item::<f64>()?, 0.0);
        Ok(())
    }

    #[test]
    fn empty_input_layout_keeps_all_groups_zero() -> TestResult {
        let client = client();
        let tree = TransferTree::<CpuRuntime>::new_grouped(&client, 3, 0, &[])?;
        let values = Tensor::from_slice(&[] as &[f64], &[0], client.device())?;
        let result = tree.aggregate(&client, &values, AggregationKind::Water)?;
        assert_eq!(result.value.try_to_vec::<f64>()?, vec![0.0; 3]);
        assert_eq!(result.bound.try_to_vec::<f64>()?, vec![0.0; 3]);
        Ok(())
    }
}
