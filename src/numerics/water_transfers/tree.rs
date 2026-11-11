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
pub(crate) enum AggregationKind {
    Water,
    Momentum,
}

#[derive(Debug)]
pub(crate) struct TransferTree<R: Runtime<DType = DType>> {
    input_count: usize,
    order: Tensor<R>,
    levels: Vec<Level<R>>,
    pub degree_margin: Tensor<R>,
}

pub(crate) struct Aggregate<R: Runtime<DType = DType>> {
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
