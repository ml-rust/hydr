use super::{ChannelError, conserved_point::ConservedPoint, context::Context};
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, IndexingOps, MidpointOps, NextRepresentableOps,
    ReduceOps, ScalarOps, TensorOps, TwoSumOps, UnaryOps,
};
use numr::runtime::{Device, Runtime};
use numr::tensor::Tensor;

/// Recovered cell fields. The F64 values are the selected high parts.
pub(super) struct Fields<R: Runtime<DType = DType>> {
    pub area: Tensor<R>,
    pub depth: Tensor<R>,
    pub velocity: Tensor<R>,
    pub discharge: Tensor<R>,
    /// Two-part area in m². The low part is `+0.0` outside surveyed geometry.
    pub area_point: ConservedPoint<R>,
    /// Two-part discharge in m³/s. The low part is `+0.0` outside surveyed geometry.
    pub discharge_point: ConservedPoint<R>,
    /// Radius discarded when the area point is selected, in m². A diagnostic only.
    pub area_selection_bound: Tensor<R>,
    /// Radius discarded when the discharge point is selected, in m³/s. A diagnostic only.
    pub discharge_selection_bound: Tensor<R>,
}

impl<R: Runtime<DType = DType>> Context<R>
where
    R::Client: BinaryOps<R> + CompareOps<R> + ConditionalOps<R> + ReduceOps<R> + UnaryOps<R>,
{
    pub fn finite(&self, value: &Tensor<R>, name: &str) -> Result<(), ChannelError> {
        let mask = self
            .client
            .le(&self.client.abs(value)?, &self.constant(value, f64::MAX)?)?;
        if !self.all(&mask)? {
            let index = self.first_rejected(&mask)?;
            let input = value.narrow(0, index, 1)?.item::<f64>()?;
            return Err(ChannelError::numerical(format!(
                "{name}, lane={index}, value={input}: supply finite values and rescale channel inputs"
            )));
        }
        Ok(())
    }

    pub fn all(&self, mask: &Tensor<R>) -> Result<bool, ChannelError> {
        Ok(self.client.min(mask, &[0], false)?.item::<f64>()? == 1.0)
    }

    pub(super) fn first_rejected(&self, admitted: &Tensor<R>) -> Result<usize, ChannelError> {
        let rejected = self.client.eq(admitted, &self.constant(admitted, 0.0)?)?;
        let selected = self.client.argmax(&rejected, 0, false)?.item::<i64>()?;
        usize::try_from(selected)
            .ok()
            .filter(|index| *index < admitted.numel())
            .ok_or_else(|| ChannelError::numerical(format!(
                "rejected lane={selected}, count={}: inspect the selected runtime's reduction result",
                admitted.numel()
            )))
    }

    pub fn represented(
        &self,
        required: &Tensor<R>,
        value: &Tensor<R>,
        name: &str,
    ) -> Result<(), ChannelError> {
        self.finite(value, name)?;
        let zero = self.constant(value, 0.0)?;
        let admitted = self.client.maximum(
            &self.client.eq(required, &zero)?,
            &self.client.gt(&self.client.abs(value)?, &zero)?,
        )?;
        if !self.all(&admitted)? {
            let index = self.first_rejected(&admitted)?;
            return Err(ChannelError::numerical(format!(
                "{name}, lane={index}, value=0 underflows from a nonzero physical operand: rescale inputs or use a representable interval"
            )));
        }
        Ok(())
    }

    pub fn total(&self, value: &Tensor<R>) -> Result<f64, ChannelError> {
        let total = self.client.sum(value, &[0], false)?.item::<f64>()?;
        if !total.is_finite() {
            return Err(ChannelError::numerical(format!(
                "channel total={total}: rescale water or momentum"
            )));
        }
        Ok(total)
    }
}

impl<R: Runtime<DType = DType>> Context<R>
where
    R::Client: TensorOps<R>
        + CompareOps<R>
        + MidpointOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>,
{
    /// Recover fields from both parts of the conserved state.
    ///
    /// Surveyed geometry divides the two-part values. Other geometry reads the high parts only.
    pub fn fields(
        &self,
        volume_point: &ConservedPoint<R>,
        momentum_point: &ConservedPoint<R>,
    ) -> Result<Fields<R>, ChannelError> {
        let volume = &volume_point.high;
        let momentum = &momentum_point.high;
        for (point, name) in [
            (volume_point, "cell water volume"),
            (momentum_point, "cell longitudinal momentum"),
        ] {
            for value in [&point.high, &point.low] {
                if value.shape() != [self.count()]
                    || value.dtype() != DType::F64
                    || !value.device().is_same(self.length.device())
                {
                    return Err(ChannelError::input(
                        "channel state requires equal F64 cell arrays on the configured device: rebuild state with this client",
                    ));
                }
            }
            self.finite(&point.high, name)?;
        }
        let c = &self.client;
        let zero = self.constant(volume, 0.0)?;
        let one = self.constant(volume, 1.0)?;
        let wet = c.gt(volume, &zero)?;
        let mut area = c.div(volume, &self.length)?;
        let mut points = None;
        if let super::geometry::TensorGeometry::Surveyed(geometry) = &self.geometry {
            let nonnegative_volume = c.ge(volume, &zero)?;
            let nonnegative_area = c.ge(&area, &zero)?;
            let finite_area = c.le(&c.abs(&area)?, &self.constant(&area, f64::MAX)?)?;
            let admitted = c.mul(
                &nonnegative_volume,
                &c.mul(&nonnegative_area, &finite_area)?,
            )?;
            if !self.all(&admitted)? {
                let index = self.first_rejected(&admitted)?;
                let read = |value: &Tensor<R>| value.narrow(0, index, 1)?.item::<f64>();
                return Err(ChannelError::numerical(format!(
                    "cell={index}, volume_m3={}, length_m={}, area_m2={}: require nonnegative water and finite nonnegative represented area before surveyed recovery; reduce the interval or rescale inputs",
                    read(volume)?, read(&self.length)?, read(&area)?
                )).at(index));
            }
            let recovered =
                self.surveyed_points(volume_point, momentum_point, &geometry.cap_area_upper)?;
            area = recovered.area.high.clone();
            points = Some(recovered);
        }
        let depth = self.section_depth(&area)?;
        let (velocity, discharge, area_point, discharge_point, area_bound, discharge_bound) =
            match points {
                Some(points) => (
                    points.velocity,
                    points.discharge.high.clone(),
                    points.area,
                    points.discharge,
                    points.area_bound,
                    points.discharge_bound,
                ),
                None => {
                    let safe_volume = c.where_cond(&wet, volume, &one)?;
                    let discharge = c.div(momentum, &self.length)?;
                    (
                        c.div(momentum, &safe_volume)?,
                        discharge.clone(),
                        ConservedPoint {
                            high: area.clone(),
                            low: zero.clone(),
                        },
                        ConservedPoint {
                            high: discharge,
                            low: zero.clone(),
                        },
                        zero.clone(),
                        zero.clone(),
                    )
                }
            };
        let mut admitted = c.ge(volume, &zero)?;
        admitted = c.mul(&admitted, &c.maximum(&wet, &c.eq(momentum, &zero)?)?)?;
        for value in [&velocity, &discharge] {
            admitted = c.mul(
                &admitted,
                &c.maximum(&c.eq(momentum, &zero)?, &c.gt(&c.abs(value)?, &zero)?)?,
            )?;
        }
        for value in [&area, &depth] {
            admitted = c.mul(
                &admitted,
                &c.maximum(&c.eq(volume, &zero)?, &c.gt(value, &zero)?)?,
            )?;
        }
        if !matches!(&self.geometry, super::geometry::TensorGeometry::Surveyed(_)) {
            admitted = c.mul(&admitted, &c.le(&depth, &self.cap)?)?;
        }
        if !self.all(&admitted)? {
            let index = self.first_rejected(&admitted)?;
            let read = |value: &Tensor<R>| value.narrow(0, index, 1)?.item::<f64>();
            return Err(ChannelError::numerical(format!(
                "cell={index}, volume_m3={}, momentum_m4_per_s={}, area_m2={}, depth_m={}, velocity_m_per_s={}, discharge_m3_per_s={}, maximum_depth_m={}: require nonnegative represented water, zero dry momentum, resolved motion, and depth within the bank limit; reduce the interval or correct inputs",
                read(volume)?, read(momentum)?, read(&area)?, read(&depth)?, read(&velocity)?, read(&discharge)?, read(&self.cap)?
            )).at(index));
        }
        for (value, name) in [
            (&area, "channel area"),
            (&depth, "channel depth"),
            (&velocity, "channel velocity"),
            (&discharge, "channel discharge"),
            (&c.add(&self.bed, &depth)?, "channel stage"),
        ] {
            self.finite(value, name)?;
        }
        self.total(volume)?;
        Ok(Fields {
            area,
            depth,
            velocity,
            discharge,
            area_point,
            discharge_point,
            area_selection_bound: area_bound,
            discharge_selection_bound: discharge_bound,
        })
    }

    /// Fields of a state with `+0.0` low parts.
    #[cfg(test)]
    pub fn fields_high(
        &self,
        volume: &Tensor<R>,
        momentum: &Tensor<R>,
    ) -> Result<Fields<R>, ChannelError> {
        self.fields(
            &ConservedPoint::from_high(volume)?,
            &ConservedPoint::from_high(momentum)?,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::super::{
        ChannelConfig, ChannelError, ChannelGeometry, ChannelInitialState, ChannelReach,
        ChannelReachDefinition, ChannelSurveyedConfig, SurveyedChannelCell,
    };
    use crate::channel::{CrossSectionDefinition, CrossSectionPoint};
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};

    type TestResult<T = ()> = Result<T, Box<dyn std::error::Error>>;

    fn reach() -> TestResult<ChannelReach<CpuRuntime>> {
        Ok(ChannelReach::new(
            CpuClient::new(CpuDevice::new()),
            ChannelReachDefinition {
                vertical_datum: "Synthetic SI datum".into(),
                provenance: "Surveyed candidate-state admission".into(),
                geometry: ChannelGeometry::PrismaticSurveyed {
                    profile: CrossSectionDefinition {
                        id: "section".into(),
                        vertical_datum: "Synthetic SI datum".into(),
                        points: [(-1.0, 3.0), (-1.0, 0.0), (1.0, 0.0), (1.0, 3.0)]
                            .into_iter()
                            .map(|(station_m, elevation_m)| CrossSectionPoint {
                                station_m,
                                elevation_m,
                            })
                            .collect(),
                        maximum_stage_m: 3.0,
                        profile_provenance: "Two-metre rectangular survey".into(),
                    },
                    cells: vec![SurveyedChannelCell {
                        id: "cell".into(),
                        length_m: 0.5,
                        manning_n: 0.0,
                    }],
                    numerical: ChannelSurveyedConfig::default(),
                },
                lateral_bindings: Vec::new(),
                config: ChannelConfig::default(),
            },
            ChannelInitialState {
                time_s: 0.0,
                volume_m3: vec![0.25],
                longitudinal_momentum_m4_per_s: vec![0.0],
            },
        )?)
    }

    fn check_rejected_candidate(volume: f64) -> TestResult {
        let reach = reach()?;
        let before = reach.snapshot_data()?;
        let candidate = reach.context.tensor(&[volume])?;
        let result = reach
            .context
            .fields_high(&candidate, &reach.accepted.momentum.high);
        let error = result
            .err()
            .ok_or("Candidate requires numerical rejection")?;
        assert!(
            matches!(error, ChannelError::Numerical { cell: Some(0), .. }),
            "{error}"
        );
        assert!(
            error.to_string().contains("before surveyed recovery"),
            "{error}"
        );
        assert_eq!(candidate.item::<f64>()?.to_bits(), volume.to_bits());
        assert_eq!(reach.snapshot_data()?, before);
        Ok(())
    }

    #[test]
    fn surveyed_overflowing_candidate_area_rejects_numerically_without_state_change() -> TestResult
    {
        assert!((f64::MAX / 0.5).is_infinite());
        check_rejected_candidate(f64::MAX)
    }

    #[test]
    fn surveyed_negative_candidate_volume_rejects_numerically_without_state_change() -> TestResult {
        check_rejected_candidate(-0.25)
    }

    #[test]
    fn surveyed_candidate_shape_error_remains_invalid_input() -> TestResult {
        let reach = reach()?;
        let before = reach.snapshot_data()?;
        let candidate = reach.context.tensor(&[0.25, 0.25])?;
        let result = reach
            .context
            .fields_high(&candidate, &reach.accepted.momentum.high);
        let error = result
            .err()
            .ok_or("Candidate shape requires input rejection")?;
        assert!(
            matches!(error, ChannelError::InvalidInput { .. }),
            "{error}"
        );
        assert_eq!(reach.snapshot_data()?, before);
        Ok(())
    }
}
