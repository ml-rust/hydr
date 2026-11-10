use super::CoupledLevelPoolError;
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, IndexingOps, ReduceOps, ScalarOps, ShapeOps, TwoSumOps,
    UnaryOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

#[derive(Debug)]
struct Level<R: Runtime<DType = DType>> {
    left: Tensor<R>,
    right: Tensor<R>,
}

#[derive(Debug)]
pub(super) struct TransferTree<R: Runtime<DType = DType>> {
    order: Tensor<R>,
    levels: Vec<Level<R>>,
    pub degree_margin: Tensor<R>,
}

pub(super) struct Aggregate<R: Runtime<DType = DType>> {
    pub volume: Tensor<R>,
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
    pub fn new(
        client: &R::Client,
        pool_count: usize,
        endpoints: &[(usize, usize)],
        ids: &[String],
    ) -> Result<Self, CoupledLevelPoolError> {
        let mut groups = vec![Vec::new(); pool_count];
        for (edge, &(a, b)) in endpoints.iter().enumerate() {
            groups[a].push((ids[edge].as_str(), edge));
            groups[b].push((ids[edge].as_str(), endpoints.len() + edge));
        }
        let zero = endpoints
            .len()
            .checked_mul(2)
            .ok_or_else(|| CoupledLevelPoolError::input("endpoint count overflows"))?;
        let mut order = Vec::new();
        let mut widths = Vec::new();
        let mut margins = Vec::new();
        for group in &mut groups {
            group.sort_by(|a, b| a.0.cmp(b.0));
            let degree = group.len();
            if degree as u128 >= (1u128 << 50) {
                return Err(CoupledLevelPoolError::input(
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
            let total: usize = widths.iter().sum();
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
            order,
            levels,
            degree_margin: Tensor::from_slice(&margins, &[margins.len()], client.device())?,
        })
    }

    pub fn aggregate(
        &self,
        client: &R::Client,
        endpoint_values: &Tensor<R>,
    ) -> Result<Aggregate<R>, CoupledLevelPoolError> {
        let zero = Tensor::<R>::zeros(&[1], DType::F64, client.device())?;
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
            let a_bound = client.index_select(&errors, 0, &level.left)?;
            let b_bound = client.index_select(&errors, 0, &level.right)?;
            bound = upward_add(
                client,
                &upward_add(client, &a_bound, &b_bound)?,
                &client.abs(&low)?,
            )?;
            volume = sum;
        }
        Ok(Aggregate { volume, bound })
    }
}

pub(super) fn finite<R: Runtime<DType = DType>>(
    client: &R::Client,
    value: &Tensor<R>,
    field: &str,
) -> Result<(), CoupledLevelPoolError>
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
    let maximum = Tensor::<R>::full_scalar(&[], DType::F64, f64::MAX, client.device())?;
    let admitted = client.le(&client.abs(value)?, &maximum)?;
    if client.min(&admitted, &[0], false)?.item::<f64>()? != 1.0 {
        return Err(CoupledLevelPoolError::numerical(format!(
            "{field} contains nonfinite values. Inspect the physical scale and arithmetic envelope"
        )));
    }
    Ok(())
}

pub(super) fn reject<R: Runtime<DType = DType>>(
    client: &R::Client,
    mask: &Tensor<R>,
    message: &str,
) -> Result<(), CoupledLevelPoolError>
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
    if client.max(mask, &[0], false)?.item::<f64>()? != 0.0 {
        return Err(CoupledLevelPoolError::numerical(message));
    }
    Ok(())
}

pub(super) fn upward_add<R: Runtime<DType = DType>>(
    client: &R::Client,
    a: &Tensor<R>,
    b: &Tensor<R>,
) -> Result<Tensor<R>, CoupledLevelPoolError>
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
    let (sum, low) = client.two_sum(a, b)?;
    let zero = Tensor::<R>::zeros(&[], DType::F64, client.device())?;
    let rounding = client.add_scalar(
        &client.mul_scalar(&client.abs(&sum)?, f64::EPSILON)?,
        f64::from_bits(1),
    )?;
    let raised = client.add(&sum, &rounding)?;
    let result = client.where_cond(&client.gt(&low, &zero)?, &raised, &sum)?;
    finite(client, &result, "transfer assembly error bound")?;
    Ok(result)
}

fn to_index(value: usize) -> Result<i64, CoupledLevelPoolError> {
    i64::try_from(value).map_err(|_| {
        CoupledLevelPoolError::input("connection topology exceeds runtime index support")
    })
}
