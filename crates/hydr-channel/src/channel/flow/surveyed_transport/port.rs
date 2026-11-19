//! Flux port admission on surveyed reaches: the hydraulic-depth speed check and dry export.

use crate::channel::flow::{
    ChannelError, ChannelTensorBoundary,
    admission::Fields,
    boundary::check_port,
    conserved_point::ConservedPoint,
    conserved_update::{exact, expanded},
    context::Context,
    geometry::{SurveyedGeometry, section_error},
};
use numr::{
    dtype::DType,
    ops::{BinaryOps, CompareOps, NextRepresentableOps, ScalarOps, TensorOps, TwoSumOps, UnaryOps},
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};

impl<R: Runtime<DType = DType>> Context<R>
where
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    /// Rejects a port speed below the end cell's `|u| + c` and export from a dry end cell.
    ///
    /// `c = sqrt(g D_h)` takes the certified lower bound of the hydraulic depth `D_h = A / T`
    /// at the end cell's two-part area point. The required speed is the downward-rounded lower
    /// bound of `|u| + c`. A dry end cell holds `+0.0` or `-0.0` in all four area and discharge
    /// parts. It carries no interior wave, so it requires no speed.
    pub(super) fn check_surveyed_flux_ports(
        &self,
        geometry: &SurveyedGeometry<R>,
        fields: &Fields<R>,
        left: &ChannelTensorBoundary<R>,
        right: &ChannelTensorBoundary<R>,
    ) -> Result<(), ChannelError> {
        for (endpoint, boundary) in [left, right].into_iter().enumerate() {
            let ChannelTensorBoundary::FluxPort(port) = boundary else {
                continue;
            };
            let cell = if endpoint == 0 { 0 } else { self.count() - 1 };
            let part = |point: &ConservedPoint<R>| {
                Ok::<_, ChannelError>(ConservedPoint {
                    high: point.high.narrow(0, cell, 1)?,
                    low: point.low.narrow(0, cell, 1)?,
                })
            };
            let area = part(&fields.area_point)?;
            let discharge = part(&fields.discharge_point)?;
            let mut dry = true;
            for value in [&area.high, &area.low, &discharge.high, &discharge.low] {
                dry &= value.item::<f64>()? == 0.0;
            }
            let required = if dry {
                0.0
            } else {
                self.surveyed_characteristic_lower(geometry, &area, &discharge)
                    .map_err(|error| error.at(cell))?
            };
            check_port(endpoint, cell, port, required, dry)?;
        }
        Ok(())
    }

    /// Downward-rounded lower bound of `|u| + sqrt(g D_h)` at one wet two-part state.
    fn surveyed_characteristic_lower(
        &self,
        geometry: &SurveyedGeometry<R>,
        area: &ConservedPoint<R>,
        discharge: &ConservedPoint<R>,
    ) -> Result<f64, ChannelError> {
        let c = &self.client;
        let points = expanded(c, "surveyed flux port end cell point", |a| {
            exact(a, &area.high, &area.low)
        })?;
        let owners = Tensor::<R>::zeros(&[1], DType::I64, c.device())?;
        let recovery = geometry
            .section
            .recover_area_points_tensor(&owners, &points, &geometry.numerical.recovery)
            .map_err(section_error)?;
        let hydraulic = recovery.hydraulic_depth_m();
        let gravity = self.constant(&area.high, self.definition.config.gravity)?;
        let required = expanded(c, "surveyed flux port characteristic speed", |a| {
            let area = exact(a, &area.high, &area.low)?;
            let discharge = exact(a, &discharge.high, &discharge.low)?;
            let area = a.expansion_interval(&area)?;
            let discharge = a.expansion_interval(&discharge)?;
            let c = a.client();
            let zero = Tensor::zeros(&[1], DType::F64, c.device())?;
            // |Q| from the discharge endpoints. An interval across zero has lower bound zero.
            // The area divisor must have a strictly positive lower bound.
            let (lower, upper) = (discharge.lower(), discharge.upper());
            let magnitude = a.bounds(
                &c.maximum(&c.maximum(lower, &c.neg(upper)?)?, &zero)?,
                &c.maximum(&c.abs(lower)?, &c.abs(upper)?)?,
            )?;
            let speed = a.div_positive_interval(&magnitude, &area)?;
            let depth = a.point(hydraulic.lower())?;
            let gravity = a.point(&gravity)?;
            let wave = a.mul(&gravity, &depth)?;
            let sound = a.sqrt_nonnegative(&wave)?;
            Ok(a.add(&speed, &sound)?.lower().clone())
        })?;
        self.finite(
            &required,
            "surveyed flux port end cell characteristic speed",
        )?;
        Ok(required.item::<f64>()?)
    }
}

#[cfg(test)]
mod tests {
    use crate::channel::flow::{
        ChannelReach, ChannelTensorBoundary, ChannelTensorFluxPort,
        surveyed_update::tests::{Client, TestResult, cpu, moving_reach},
    };
    use numr::dtype::DType;
    use numr::runtime::Runtime;
    use numr::runtime::cpu::CpuRuntime;

    fn port<R>(
        reach: &ChannelReach<R>,
        water: f64,
        speed: f64,
    ) -> TestResult<ChannelTensorBoundary<R>>
    where
        R: Runtime<DType = DType>,
        R::Client: Client<R>,
    {
        let context = &reach.context;
        Ok(ChannelTensorBoundary::FluxPort(Box::new(
            ChannelTensorFluxPort::new(
                context.tensor(&[water])?,
                context.tensor(&[0.0])?,
                context.tensor(&[speed])?,
                Some(context.tensor(&[speed])?),
                "surveyed port admission fixture",
            )?,
        )))
    }

    /// Depths 2 m and 1 m beside a dry right end cell, on the width-2 rectangular survey.
    fn reach<R>(client: R::Client) -> TestResult<ChannelReach<R>>
    where
        R: Runtime<DType = DType>,
        R::Client: Client<R>,
    {
        moving_reach::<R>(client, &[20.0, 10.0, 0.0], &[4.0, -2.0, 0.0], &[0.0; 3])
    }

    fn rejection<R>(
        reach: &ChannelReach<R>,
        left: &ChannelTensorBoundary<R>,
        right: &ChannelTensorBoundary<R>,
    ) -> TestResult<String>
    where
        R: Runtime<DType = DType>,
        R::Client: Client<R>,
    {
        let context = &reach.context;
        let fields = context.fields(&reach.accepted.volume, &reach.accepted.momentum)?;
        match context.flux(&fields, left, right) {
            Ok(_) => Err("the port was admitted".into()),
            Err(error) => Ok(error.to_string()),
        }
    }

    fn speed_bounds_the_hydraulic_depth_characteristic<R>(client: R::Client) -> TestResult
    where
        R: Runtime<DType = DType>,
        R::Client: Client<R>,
    {
        let reach = reach::<R>(client)?;
        let context = &reach.context;
        let gravity = context.definition.config.gravity;
        // The rectangle has D_h = h = 2 m in the left end cell, which moves at 0.2 m/s.
        let characteristic = 0.2 + (gravity * 2.0).sqrt();
        let wall = ChannelTensorBoundary::Wall;
        let slow = port(&reach, 0.0, characteristic * (1.0 - 1e-12))?;
        let message = rejection(&reach, &slow, &wall)?;
        assert!(message.contains("flux port left speed="), "{message}");
        assert!(
            message.contains("is below the end cell |u|+c="),
            "{message}"
        );
        let fields = context.fields(&reach.accepted.volume, &reach.accepted.momentum)?;
        let fast = port(&reach, 0.0, characteristic * (1.0 + 1e-12))?;
        context.flux(&fields, &fast, &wall)?;
        Ok(())
    }

    fn dry_end_rejects_export_and_admits_import<R>(client: R::Client) -> TestResult
    where
        R: Runtime<DType = DType>,
        R::Client: Client<R>,
    {
        let reach = reach::<R>(client)?;
        let context = &reach.context;
        let wall = ChannelTensorBoundary::Wall;
        // A dry end cell requires no speed, so a slow port passes the speed check.
        let export = port(&reach, 0.125, 1e-3)?;
        let message = rejection(&reach, &wall, &export)?;
        assert!(
            message.contains("flux port right exports water=0.125 m³/s from a dry end cell"),
            "{message}"
        );
        let fields = context.fields(&reach.accepted.volume, &reach.accepted.momentum)?;
        let import = port(&reach, -0.125, 1e-3)?;
        let flux = context.flux(&fields, &wall, &import)?;
        assert_eq!(flux.water.try_to_vec::<f64>()?[3], -0.125);
        Ok(())
    }

    #[test]
    fn cpu_surveyed_port_speed_bounds_the_hydraulic_depth_characteristic() -> TestResult {
        speed_bounds_the_hydraulic_depth_characteristic::<CpuRuntime>(cpu())
    }

    #[test]
    fn cpu_surveyed_dry_end_rejects_export_and_admits_import() -> TestResult {
        dry_end_rejects_export_and_admits_import::<CpuRuntime>(cpu())
    }

    #[cfg(feature = "cuda")]
    mod cuda {
        use super::*;
        use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};

        #[test]
        fn cuda_surveyed_port_speed_bounds_the_hydraulic_depth_characteristic() -> TestResult {
            speed_bounds_the_hydraulic_depth_characteristic::<CudaRuntime>(CudaClient::new(
                CudaDevice::new(0),
            )?)
        }

        #[test]
        fn cuda_surveyed_dry_end_rejects_export_and_admits_import() -> TestResult {
            dry_end_rejects_export_and_admits_import::<CudaRuntime>(CudaClient::new(
                CudaDevice::new(0),
            )?)
        }
    }
}
