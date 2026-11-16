use super::super::{ChannelError, ChannelReachDefinition};
use super::surveyed_config::ChannelSurveyedConfig;
use crate::channel::flow::surveyed::{
    config::PrismaticRiemannConfig,
    error::{HydraulicError, checked, into_public},
    work::PrismaticRiemannWork,
};
use crate::channel::section::HydraulicGeometry;
use crate::channel::section::{
    CrossSectionBatch, CrossSectionDefinition, CrossSectionError, PrismaticPerimeterBatch,
    PrismaticPerimeterRecovery, PrismaticSectionBatch,
};
use numr::{
    dtype::DType,
    ops::{
        CompareOps, HypotOps, MidpointOps, NextRepresentableOps, ReduceOps, ScalarOps, TensorOps,
        TwoSumOps,
    },
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};

pub(in crate::channel::flow) struct SurveyedGeometry<R: Runtime<DType = DType>> {
    pub section: PrismaticSectionBatch<R>,
    pub perimeter: Option<PrismaticPerimeterBatch<R>>,
    pub numerical: ChannelSurveyedConfig,
    pub riemann: PrismaticRiemannConfig,
    pub cell_owners: Tensor<R>,
    pub face_owners: Tensor<R>,
}

pub(in crate::channel::flow) fn section_error(error: CrossSectionError) -> ChannelError {
    let context = error.to_string();
    match error {
        CrossSectionError::InvalidInput { .. } => ChannelError::input(context),
        CrossSectionError::Runtime { .. } => ChannelError::Runtime {
            cell: None,
            context,
        },
        _ => ChannelError::numerical(context),
    }
}

fn hydraulic_error(error: HydraulicError) -> ChannelError {
    match into_public(error, PrismaticRiemannWork::default()) {
        crate::channel::flow::surveyed::error::PrismaticRiemannError::Channel(error) => error,
        error => ChannelError::numerical(error.to_string()),
    }
}

impl<R: Runtime<DType = DType>> SurveyedGeometry<R>
where
    R::Client: TensorOps<R>
        + CompareOps<R>
        + HypotOps<R>
        + MidpointOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>,
{
    pub fn new(
        client: &R::Client,
        definition: &ChannelReachDefinition,
        profile: &CrossSectionDefinition,
        numerical: &ChannelSurveyedConfig,
        length: &Tensor<R>,
    ) -> Result<Self, ChannelError> {
        numerical.check()?;
        let riemann = numerical.riemann_config(&definition.config);
        riemann.check().map_err(hydraulic_error)?;
        let section = CrossSectionBatch::new(client.clone(), vec![profile.clone()])
            .map_err(section_error)?
            .certify_prismatic()
            .map_err(section_error)?;
        let perimeter = if definition.geometry.cells().any(|c| c.manning_n > 0.0) {
            Some(section.certify_perimeter().map_err(section_error)?)
        } else {
            None
        };
        let count = definition.geometry.len();
        let cell_owners = Tensor::from_slice(&vec![0i64; count], &[count], client.device())?;
        let face_count = count.checked_add(1).ok_or_else(|| {
            ChannelError::input("surveyed face count overflows usize: reduce cell count")
        })?;
        let face_owners =
            Tensor::from_slice(&vec![0i64; face_count], &[face_count], client.device())?;
        let owner = Tensor::from_slice(&[0i64], &[1], client.device())?;
        let geometry = HydraulicGeometry::from_batch(&section);
        checked(client, "surveyed reach capacity", |a| {
            let bands = geometry.gather(a, &owner)?;
            let lengths = a.point(length)?;
            let capacity = a.mul(&bands.cap_area, &lengths)?;
            let zero = Tensor::from_slice(&[0.0], &[1], client.device())?;
            if client.min(&client.gt(capacity.lower(), &zero)?, &[0], false)?.item::<f64>()? != 1.0 {
                return Err(HydraulicError::Numerical { context: "surveyed cap volume has no finite positive certificate: rescale profile or lengths".into() });
            }
            Ok(())
        }).map_err(hydraulic_error)?;
        Ok(Self {
            section,
            perimeter,
            numerical: numerical.clone(),
            riemann,
            cell_owners,
            face_owners,
        })
    }
}

impl<R: Runtime<DType = DType>> SurveyedGeometry<R>
where
    R::Client: TensorOps<R>
        + CompareOps<R>
        + MidpointOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>,
{
    pub fn recover_depth(&self, area: &Tensor<R>) -> Result<Tensor<R>, ChannelError> {
        let query = self
            .section
            .recover_depths_tensor(&self.cell_owners, area, &self.numerical.recovery)
            .map_err(section_error)?;
        let client = HydraulicGeometry::from_batch(&self.section);
        Ok(client
            .client()
            .midpoint(query.depth_m().lower(), query.depth_m().upper())?)
    }

    pub fn exterior_area(&self, depth: &Tensor<R>) -> Result<Tensor<R>, ChannelError> {
        let geometry = HydraulicGeometry::from_batch(&self.section);
        let client = geometry.client();
        let owners =
            Tensor::from_slice(&vec![0i64; depth.numel()], depth.shape(), client.device())?;
        let query = self
            .section
            .evaluate_depths_tensor(&owners, depth)
            .map_err(section_error)?;
        let area = client.midpoint(query.area_m2().lower(), query.area_m2().upper())?;
        // Recovery admits the represented area against original-coordinate cap support.
        self.section
            .recover_depths_tensor(&owners, &area, &self.numerical.recovery)
            .map_err(section_error)?;
        Ok(area)
    }
}

impl<R: Runtime<DType = DType>> SurveyedGeometry<R>
where
    R::Client: TensorOps<R> + CompareOps<R> + NextRepresentableOps<R> + ScalarOps<R> + TwoSumOps<R>,
{
    pub fn recover_perimeter(
        &self,
        area: &Tensor<R>,
    ) -> Result<PrismaticPerimeterRecovery<R>, ChannelError> {
        let perimeter = self.perimeter.as_ref().ok_or_else(|| ChannelError::numerical("surveyed perimeter certificate is absent: request friction geometry only for positive roughness"))?;
        perimeter
            .recover_depths_tensor(&self.cell_owners, area, &self.numerical.recovery)
            .map_err(section_error)
    }
}
