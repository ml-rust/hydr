use super::super::{ChannelError, ChannelGeometry, ChannelReachDefinition, context::Context};
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, HypotOps, MidpointOps, ReduceOps, ScalarOps, ShapeOps,
    UnaryOps,
};
use numr::runtime::{Runtime, RuntimeClient, SquareRootContract};
use numr::tensor::Tensor;

pub(in crate::channel::flow) enum TensorGeometry<R: Runtime<DType = DType>> {
    Rectangular(Box<RectangularGeometry<R>>),
    Triangular(Box<TriangleGeometry<R>>),
}

pub(in crate::channel::flow) struct RectangularGeometry<R: Runtime<DType = DType>> {
    pub width: Tensor<R>,
    pub face_width: Tensor<R>,
}

pub(in crate::channel::flow) struct TriangleGeometry<R: Runtime<DType = DType>> {
    pub coefficient: Tensor<R>,
    pub root_coefficient: Tensor<R>,
    pub root_coefficient_lower: Tensor<R>,
    pub root_coefficient_upper: Tensor<R>,
    pub root_gravity: Tensor<R>,
    pub root_gravity_lower: Tensor<R>,
    pub root_gravity_upper: Tensor<R>,
    pub root_two: Tensor<R>,
    pub root_two_lower: Tensor<R>,
    pub root_two_upper: Tensor<R>,
    pub left_contact_factor: Tensor<R>,
    pub right_contact_factor: Tensor<R>,
}

impl<R: Runtime<DType = DType>> TensorGeometry<R>
where
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + HypotOps<R>
        + MidpointOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + ShapeOps<R>
        + UnaryOps<R>,
{
    pub fn new(
        client: &R::Client,
        definition: &ChannelReachDefinition,
        cap: &Tensor<R>,
        length: &Tensor<R>,
    ) -> Result<Self, ChannelError> {
        match &definition.geometry {
            ChannelGeometry::Rectangular { cells } => {
                let values: Vec<_> = cells.iter().map(|c| c.width_m).collect();
                let count = values.len();
                let width = Tensor::<R>::from_slice(&values, &[count], client.device())?;
                let first = width.narrow(0, 0, 1)?;
                let last = width.narrow(0, count - 1, 1)?;
                let face_width = if count == 1 {
                    client.cat(&[&first, &last], 0)?
                } else {
                    let middle = client.midpoint(
                        &width.narrow(0, 0, count - 1)?,
                        &width.narrow(0, 1, count - 1)?,
                    )?;
                    client.cat(&[&first, &middle, &last], 0)?
                };
                Ok(Self::Rectangular(Box::new(RectangularGeometry {
                    width,
                    face_width,
                })))
            }
            ChannelGeometry::PrismaticTriangular {
                left_bank_slope_h_per_v,
                right_bank_slope_h_per_v,
                ..
            } => {
                if R::square_root_contract(DType::F64) != SquareRootContract::Ieee754RoundToNearest
                {
                    return Err(ChannelError::input(
                        "triangular F64 square roots lack round-to-nearest guarantees: select an admitted runtime",
                    ));
                }
                let make = |value| Tensor::<R>::from_slice(&[value], &[1], client.device());
                let left = make(*left_bank_slope_h_per_v)?;
                let right = make(*right_bank_slope_h_per_v)?;
                let one = make(1.0)?;
                let coefficient = client.midpoint(&left, &right)?;
                require_positive(&coefficient, "triangular midpoint coefficient")?;
                let root_coefficient = client.sqrt(&coefficient)?;
                let lower = super::super::triangular::envelope::lower::<R>;
                let upper = super::super::triangular::envelope::upper::<R>;
                let coefficient_lower = lower(client, &coefficient)?;
                require_positive(&coefficient_lower, "triangular lower midpoint coefficient")?;
                let root_coefficient_lower = lower(client, &client.sqrt(&coefficient_lower)?)?;
                let root_coefficient_upper =
                    upper(client, &client.sqrt(&upper(client, &coefficient)?)?)?;
                let root_gravity = client.sqrt(&make(definition.config.gravity)?)?;
                let root_gravity_lower = lower(client, &root_gravity)?;
                let root_gravity_upper = upper(client, &root_gravity)?;
                let root_two = client.sqrt(&make(2.0)?)?;
                let root_two_lower = lower(client, &root_two)?;
                let root_two_upper = upper(client, &root_two)?;
                for (tensor, name) in [
                    (&root_coefficient, "triangular coefficient square root"),
                    (
                        &root_coefficient_lower,
                        "triangular lower coefficient square root",
                    ),
                    (
                        &root_coefficient_upper,
                        "triangular upper coefficient square root",
                    ),
                    (&root_gravity, "triangular gravity square root"),
                    (&root_gravity_lower, "triangular lower gravity square root"),
                    (&root_gravity_upper, "triangular upper gravity square root"),
                    (&root_two, "triangular two square root"),
                    (&root_two_lower, "triangular lower two square root"),
                    (&root_two_upper, "triangular upper two square root"),
                ] {
                    require_positive(tensor, name)?;
                }
                let left_contact_factor = client.hypot(&left, &one)?;
                let right_contact_factor = client.hypot(&right, &one)?;
                require_positive(&left_contact_factor, "left bank contact factor")?;
                require_positive(&right_contact_factor, "right bank contact factor")?;
                let area = client.mul(&client.mul(&coefficient, cap)?, cap)?;
                let volume = client.mul(&area, length)?;
                let finite = client.le(
                    &volume,
                    &Tensor::<R>::full_scalar(
                        volume.shape(),
                        DType::F64,
                        f64::MAX,
                        client.device(),
                    )?,
                )?;
                let positive = client.gt(
                    &volume,
                    &Tensor::<R>::full_scalar(volume.shape(), DType::F64, 0.0, client.device())?,
                )?;
                if client
                    .min(&client.mul(&finite, &positive)?, &[0], false)?
                    .item::<f64>()?
                    != 1.0
                {
                    return Err(ChannelError::input(
                        "triangular geometric capacity is nonfinite or unresolved: rescale slopes, depths, or lengths",
                    ));
                }
                Ok(Self::Triangular(Box::new(TriangleGeometry {
                    coefficient,
                    root_coefficient,
                    root_coefficient_lower,
                    root_coefficient_upper,
                    root_gravity,
                    root_gravity_lower,
                    root_gravity_upper,
                    root_two,
                    root_two_lower,
                    root_two_upper,
                    left_contact_factor,
                    right_contact_factor,
                })))
            }
        }
    }
}

fn require_positive<R: Runtime<DType = DType>>(
    tensor: &Tensor<R>,
    name: &str,
) -> Result<(), ChannelError> {
    let value = tensor.item::<f64>()?;
    if !value.is_finite() || value <= 0.0 {
        return Err(ChannelError::input(format!(
            "{name}={value}: rescale geometry to resolve finite positive denominators"
        )));
    }
    Ok(())
}

impl<R: Runtime<DType = DType>> Context<R>
where
    R::Client: BinaryOps<R> + UnaryOps<R>,
{
    pub fn section_depth(&self, area: &Tensor<R>) -> Result<Tensor<R>, ChannelError> {
        Ok(match &self.geometry {
            TensorGeometry::Rectangular(section) => self.client.div(area, &section.width)?,
            TensorGeometry::Triangular(section) => self
                .client
                .div(&self.client.sqrt(area)?, &section.root_coefficient)?,
        })
    }
}

impl<R: Runtime<DType = DType>> Context<R>
where
    R::Client: BinaryOps<R> + ScalarOps<R>,
{
    pub fn section_perimeter(&self, depth: &Tensor<R>) -> Result<Tensor<R>, ChannelError> {
        Ok(match &self.geometry {
            TensorGeometry::Rectangular(section) => self
                .client
                .add(&section.width, &self.client.mul_scalar(depth, 2.0)?)?,
            TensorGeometry::Triangular(section) => self.client.add(
                &self.client.mul(depth, &section.left_contact_factor)?,
                &self.client.mul(depth, &section.right_contact_factor)?,
            )?,
        })
    }
}
