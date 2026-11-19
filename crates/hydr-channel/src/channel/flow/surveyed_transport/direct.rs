//! Direct-rate end faces of surveyed reaches.
//!
//! A direct end face takes its rates from the endpoint policy, not from a point Riemann
//! problem. Its rates enter the two-part face arrays with `+0.0` low parts and radii.

use crate::channel::flow::{
    ChannelError, ChannelTensorBoundary,
    admission::Fields,
    boundary::{DirectBoundaryFrame, assembly::is_direct},
    conserved_update::Parts,
    context::Context,
    geometry::SurveyedGeometry,
};
use numr::{
    dtype::DType,
    ops::{CompareOps, MidpointOps, NextRepresentableOps, ScalarOps, TensorOps, TwoSumOps},
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::{
    integrate::CertifiedSimpsonAlgorithms,
    optimize::roots::{BatchedBisectionAlgorithms, CertifiedBisectionAlgorithms},
};

impl<R: Runtime<DType = DType>> Context<R>
where
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + MidpointOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>
        + CertifiedSimpsonAlgorithms<R>
        + CertifiedBisectionAlgorithms<R>
        + BatchedBisectionAlgorithms<R>,
{
    /// Rates at the direct end faces, or `None` when every face solves a point Riemann problem.
    ///
    /// With `check`, each flux port first passes its admission against its end cell. Each
    /// scalar end takes its certified trace on the end cell's wave curve.
    pub(super) fn surveyed_direct_frame(
        &self,
        geometry: &SurveyedGeometry<R>,
        fields: &Fields<R>,
        left: &ChannelTensorBoundary<R>,
        right: &ChannelTensorBoundary<R>,
        check: bool,
    ) -> Result<Option<DirectBoundaryFrame<R>>, ChannelError> {
        if !is_direct(left) && !is_direct(right) {
            return Ok(None);
        }
        if check {
            self.check_surveyed_flux_ports(geometry, fields, left, right)?;
        }
        self.direct_frame(left, right, || {
            self.surveyed_scalar_frame(geometry, fields, left, right)
        })
    }

    /// Two-part face rates with each direct entry placed around the numerical entries.
    ///
    /// A direct entry holds the F64 rate as `high`, with `+0.0` as `low` and as `bound`.
    pub(super) fn surveyed_direct_parts(
        &self,
        frame: Option<&DirectBoundaryFrame<R>>,
        numerical: Option<&Parts<R>>,
        direct: Option<&Tensor<R>>,
    ) -> Result<Parts<R>, ChannelError> {
        let zero = direct.map(|rate| self.constant(rate, 0.0)).transpose()?;
        let place = |numerical: Option<&Tensor<R>>, direct: Option<&Tensor<R>>| {
            self.assemble_boundary_faces(frame, numerical, direct)
        };
        Ok(Parts {
            high: place(numerical.map(|p| &p.high), direct)?,
            low: place(numerical.map(|p| &p.low), zero.as_ref())?,
            bound: place(numerical.map(|p| &p.bound), zero.as_ref())?,
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::channel::flow::{
        ChannelTensorBoundary, ChannelTensorFluxPort,
        conserved_update::Parts,
        surveyed_update::tests::{Client, TestResult, cpu, moving_reach},
    };
    use numr::dtype::DType;
    use numr::runtime::Runtime;
    use numr::runtime::cpu::CpuRuntime;
    use numr::tensor::Tensor;

    fn bits<R: Runtime<DType = DType>>(value: &Tensor<R>) -> TestResult<Vec<u64>> {
        Ok(value
            .try_to_vec::<f64>()?
            .into_iter()
            .map(f64::to_bits)
            .collect())
    }

    fn port<R>(
        reach: &crate::channel::flow::ChannelReach<R>,
        values: [f64; 4],
    ) -> TestResult<ChannelTensorBoundary<R>>
    where
        R: Runtime<DType = DType>,
        R::Client: Client<R>,
    {
        let context = &reach.context;
        Ok(ChannelTensorBoundary::FluxPort(Box::new(
            ChannelTensorFluxPort::new(
                context.tensor(&[values[0]])?,
                context.tensor(&[values[1]])?,
                context.tensor(&[values[2]])?,
                Some(context.tensor(&[values[3]])?),
                "direct face fixture",
            )?,
        )))
    }

    /// Direct entries surround the numerical entries with `+0.0` low parts and radii.
    fn direct_parts_surround_numerical<R>(client: R::Client) -> TestResult
    where
        R: Runtime<DType = DType>,
        R::Client: Client<R>,
    {
        let reach = moving_reach::<R>(client, &[20.0, 10.0, 15.0], &[4.0, -2.0, 3.0], &[0.0; 3])?;
        let context = &reach.context;
        let wall = ChannelTensorBoundary::Wall;
        let left = port(&reach, [-0.75, 2.5, 20.0, 9.0])?;
        let right = port(&reach, [0.375, 1.25, 20.0, 9.0])?;
        for (l, r, endpoints) in [
            (&left, &right, vec![0, 1]),
            (&left, &wall, vec![0]),
            (&wall, &right, vec![1]),
        ] {
            let fields = context.fields(&reach.accepted.volume, &reach.accepted.momentum)?;
            let crate::channel::flow::geometry::TensorGeometry::Surveyed(geometry) =
                &context.geometry
            else {
                return Err("surveyed geometry required".into());
            };
            let frame = context
                .surveyed_direct_frame(geometry, &fields, l, r, true)?
                .ok_or("a flux port end must give a direct frame")?;
            assert_eq!(frame.endpoints, endpoints);
            let inner = endpoints.len();
            let numerical_len = 4 - inner;
            let numerical = Parts {
                high: context.tensor(&vec![3.0; numerical_len])?,
                low: context.tensor(&vec![2f64.powi(-60); numerical_len])?,
                bound: context.tensor(&vec![2f64.powi(-70); numerical_len])?,
            };
            let water = context.surveyed_direct_parts(
                Some(&frame),
                Some(&numerical),
                Some(&frame.longitudinal_water),
            )?;
            let high = water.high.try_to_vec::<f64>()?;
            let low = water.low.try_to_vec::<f64>()?;
            let bound = water.bound.try_to_vec::<f64>()?;
            assert_eq!((high.len(), low.len(), bound.len()), (4, 4, 4));
            let rates = frame.longitudinal_water.try_to_vec::<f64>()?;
            for (slot, &endpoint) in endpoints.iter().enumerate() {
                let face = if endpoint == 0 { 0 } else { 3 };
                // The left port's outward export of -0.75 enters along +x.
                let expected: f64 = [0.75, 0.375][endpoint];
                assert_eq!(high[face].to_bits(), expected.to_bits());
                assert_eq!(rates[slot].to_bits(), expected.to_bits());
                assert_eq!(low[face].to_bits(), 0.0_f64.to_bits());
                assert_eq!(bound[face].to_bits(), 0.0_f64.to_bits());
            }
            let first = usize::from(endpoints.first() == Some(&0));
            assert_eq!(
                bits(&water.low.narrow(0, first, numerical_len)?)?,
                bits(&numerical.low)?
            );
            assert_eq!(
                bits(&water.bound.narrow(0, first, numerical_len)?)?,
                bits(&numerical.bound)?
            );
        }
        let fields = context.fields(&reach.accepted.volume, &reach.accepted.momentum)?;
        let crate::channel::flow::geometry::TensorGeometry::Surveyed(geometry) = &context.geometry
        else {
            return Err("surveyed geometry required".into());
        };
        assert!(
            context
                .surveyed_direct_frame(geometry, &fields, &wall, &wall, true)?
                .is_none()
        );
        Ok(())
    }

    /// Both port faces of one interval enter the face arrays as single-part rates.
    fn port_faces_enter_the_flux<R>(client: R::Client) -> TestResult
    where
        R: Runtime<DType = DType>,
        R::Client: Client<R>,
    {
        let reach = moving_reach::<R>(client, &[20.0, 10.0, 15.0], &[4.0, -2.0, 3.0], &[0.0; 3])?;
        let context = &reach.context;
        let left = port(&reach, [-0.75, 2.5, 20.0, 9.0])?;
        let right = port(&reach, [0.375, 1.25, 30.0, 9.0])?;
        let fields = context.fields(&reach.accepted.volume, &reach.accepted.momentum)?;
        let flux = context.flux(&fields, &left, &right)?;
        let rates = flux.surveyed.as_ref().ok_or("surveyed rates required")?;
        assert_eq!(rates.direct, [true, true]);
        let water = flux.water.try_to_vec::<f64>()?;
        let momentum = flux.momentum.try_to_vec::<f64>()?;
        assert_eq!([water[0], water[3]], [0.75, 0.375]);
        assert_eq!([momentum[0], momentum[3]], [2.5, 1.25]);
        for part in [
            &rates.water.low,
            &rates.water.bound,
            &rates.momentum.low,
            &rates.momentum.bound,
        ] {
            let part = part.try_to_vec::<f64>()?;
            assert_eq!([part[0].to_bits(), part[3].to_bits()], [0, 0]);
        }
        // A port bound serves both directions of its face.
        let rightward = flux.speed.rightward.try_to_vec::<f64>()?;
        let leftward = flux.speed.leftward.try_to_vec::<f64>()?;
        assert_eq!([rightward[0], leftward[0]], [20.0, 20.0]);
        assert_eq!([rightward[3], leftward[3]], [30.0, 30.0]);
        let riemann = flux
            .riemann
            .as_ref()
            .ok_or("Riemann diagnostics required")?;
        let widths = riemann.water_flux_width.try_to_vec::<f64>()?;
        assert_eq!([widths[0], widths[3]], [0.0, 0.0]);
        assert!(flux.prescribed_water.is_some());
        assert_eq!(
            flux.prescribed_water
                .as_ref()
                .ok_or("prescribed mask required")?
                .try_to_vec::<f64>()?,
            vec![0.0; 4]
        );
        // The right port export joins the depletion limit of the last cell.
        let reservation = context.outgoing_rate(&flux.water)?.try_to_vec::<f64>()?;
        assert_eq!(reservation[2], 0.375 + (-water[2]).max(0.0));
        Ok(())
    }

    #[test]
    fn cpu_direct_parts_surround_the_numerical_faces() -> TestResult {
        direct_parts_surround_numerical::<CpuRuntime>(cpu())
    }

    #[test]
    fn cpu_port_faces_enter_the_surveyed_flux_as_single_part_rates() -> TestResult {
        port_faces_enter_the_flux::<CpuRuntime>(cpu())
    }

    #[cfg(feature = "cuda")]
    mod cuda {
        use super::*;
        use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};

        #[test]
        fn cuda_direct_parts_surround_the_numerical_faces() -> TestResult {
            direct_parts_surround_numerical::<CudaRuntime>(CudaClient::new(CudaDevice::new(0))?)
        }

        #[test]
        fn cuda_port_faces_enter_the_surveyed_flux_as_single_part_rates() -> TestResult {
            port_faces_enter_the_flux::<CudaRuntime>(CudaClient::new(CudaDevice::new(0))?)
        }
    }
}
