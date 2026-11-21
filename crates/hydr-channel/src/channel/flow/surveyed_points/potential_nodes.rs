//! Ordinary wet-potential Simpson callbacks that reuse value-node endpoint states.

use super::potential_callbacks::{
    PanelFields, fields_fourth_bound, fields_integrand, fourth_bound, hull, integrand,
};
use crate::channel::flow::surveyed::{
    equation_ops::scalar,
    error::{Arithmetic, HydraulicResult},
    lane_quadrature::SimpsonCallbacks,
};
use crate::channel::section::ConditionalBandSegments;
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, IndexingOps, NextRepresentableOps, ScalarOps, TensorOps, TwoSumOps,
    UtilityOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;
use solvr::common::arithmetic::Interval;

/// Endpoint area and hydraulic depth of every value node, with the owner-selected segments.
struct NodeStates<R: Runtime<DType = DType>> {
    selected: ConditionalBandSegments<R>,
    nodes: usize,
    lower: [Interval<R>; 2],
    upper: [Interval<R>; 2],
}

/// Ordinary wet-potential Simpson callbacks that keep value-node endpoint states for the panels.
///
/// Solvr panel `j` of lane `k` spans the lower endpoint of value node `2j` through the upper
/// endpoint of node `2j+2`. Its endpoint states are those value-node states, so the fourth bound
/// gathers them instead of evaluating the segments again. The nominal query evaluates its point
/// once: the hull of two equal endpoint states is that state.
pub(crate) struct NodeCallbacks<'s, R: Runtime<DType = DType>> {
    segments: &'s ConditionalBandSegments<R>,
    gravity: f64,
    nodes: Option<NodeStates<R>>,
}

impl<'s, R: Runtime<DType = DType>> NodeCallbacks<'s, R> {
    pub(crate) fn new(segments: &'s ConditionalBandSegments<R>, gravity: f64) -> Self {
        Self {
            segments,
            gravity,
            nodes: None,
        }
    }
}

impl<R> SimpsonCallbacks<R> for NodeCallbacks<'_, R>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    fn values(
        &mut self,
        a: &mut Arithmetic<'_, R>,
        owner: &Tensor<R>,
        x: &Interval<R>,
    ) -> HydraulicResult<Interval<R>> {
        self.nodes = None;
        let selected = self.segments.select(a, owner)?;
        let lower = selected.point_state(a, x.lower())?;
        let upper = selected.point_state(a, x.upper())?;
        let fields = PanelFields {
            active: selected.active_mask().clone(),
            area: hull(a, &lower[0], &upper[0])?,
            gap: a.bounds(
                selected.physical_gap().lower(),
                selected.physical_gap().upper(),
            )?,
            hydraulic_depth: hull(a, &lower[1], &upper[1])?,
        };
        let gravity = scalar(a, self.gravity)?;
        let value = fields_integrand(a, &fields, &gravity)?;
        self.nodes = Some(NodeStates {
            selected,
            nodes: owner.numel(),
            lower,
            upper,
        });
        Ok(value)
    }

    fn nominal(
        &mut self,
        a: &mut Arithmetic<'_, R>,
        owner: &Tensor<R>,
        point: &Tensor<R>,
    ) -> HydraulicResult<Interval<R>> {
        let gravity = scalar(a, self.gravity)?;
        let Some(nodes) = self
            .nodes
            .as_ref()
            .filter(|nodes| nodes.nodes == owner.numel())
        else {
            let parameter = a.point(point)?;
            return integrand(a, self.segments, owner, &parameter, &gravity);
        };
        let selected = &nodes.selected;
        let [area, hydraulic_depth] = selected.point_state(a, point)?;
        let fields = PanelFields {
            active: selected.active_mask().clone(),
            area,
            gap: a.bounds(
                selected.physical_gap().lower(),
                selected.physical_gap().upper(),
            )?,
            hydraulic_depth,
        };
        fields_integrand(a, &fields, &gravity)
    }

    fn fourth(
        &mut self,
        a: &mut Arithmetic<'_, R>,
        owner: &Tensor<R>,
        x: &Interval<R>,
    ) -> HydraulicResult<Tensor<R>> {
        let gravity = scalar(a, self.gravity)?;
        let nodes = self.nodes.take();
        let layout = match &nodes {
            Some(nodes) => panel_nodes(a, nodes.nodes, owner.numel())?,
            None => None,
        };
        let (Some(nodes), Some([starts, ends])) = (nodes, layout) else {
            return fourth_bound(a, self.segments, owner, x, &gravity);
        };
        let c = a.client();
        let gather = |a: &mut Arithmetic<'_, R>, value: &Interval<R>, rows: &Tensor<R>| {
            let lower = c.index_select(value.lower(), 0, rows)?;
            let upper = c.index_select(value.upper(), 0, rows)?;
            HydraulicResult::Ok(a.bounds(&lower, &upper)?)
        };
        let selected = &nodes.selected;
        let lower_area = gather(a, &nodes.lower[0], &starts)?;
        let upper_area = gather(a, &nodes.upper[0], &ends)?;
        let lower_depth = gather(a, &nodes.lower[1], &starts)?;
        let upper_depth = gather(a, &nodes.upper[1], &ends)?;
        let fields = PanelFields {
            active: c.index_select(selected.active_mask(), 0, &starts)?,
            area: hull(a, &lower_area, &upper_area)?,
            gap: gather(a, selected.physical_gap(), &starts)?,
            hydraulic_depth: hull(a, &lower_depth, &upper_depth)?,
        };
        fields_fourth_bound(a, &fields, &gravity)
    }
}

/// Value-node indices of each panel's lower and upper endpoint.
///
/// `nodes = K*(2N+1)` value nodes and `panels = K*N` panels fix `K` lanes of `N` panels.
/// Other counts return `None`.
fn panel_nodes<R>(
    a: &mut Arithmetic<'_, R>,
    nodes: usize,
    panels: usize,
) -> HydraulicResult<Option<[Tensor<R>; 2]>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    let lanes = panels
        .checked_mul(2)
        .and_then(|twice| nodes.checked_sub(twice));
    let Some(lanes) =
        lanes.filter(|&lanes| lanes != 0 && panels != 0 && panels.is_multiple_of(lanes))
    else {
        return Ok(None);
    };
    let per_lane = panels / lanes;
    let stride = 2 * per_lane + 1;
    let c = a.client();
    let first = c
        .arange(0.0, nodes as f64, stride as f64, DType::I64)?
        .unsqueeze(1)?;
    let offset = c
        .arange(0.0, (2 * per_lane) as f64, 2.0, DType::I64)?
        .unsqueeze(0)?;
    let starts = c.add(&first, &offset)?.reshape(&[panels])?;
    let two = Tensor::from_slice(&[2_i64], &[1], c.device())?;
    let ends = c.add(&starts, &two)?;
    Ok(Some([starts, ends]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::channel::flow::surveyed::{
        PrismaticRiemannConfig,
        error::{HydraulicError, checked},
        lane_quadrature::simpson_contributions,
        work::WorkLedger,
    };
    use crate::channel::section::node_fixture::{self, Client, Result, hydraulic};
    use solvr::integrate::{CertifiedSimpsonAlgorithms, CertifiedSimpsonOptions};

    const GRAVITY: f64 = 9.80665;

    fn bits<R: Runtime<DType = DType>>(value: &Tensor<R>) -> HydraulicResult<Vec<u64>> {
        Ok(value
            .try_to_vec::<f64>()?
            .into_iter()
            .map(f64::to_bits)
            .collect())
    }

    /// Evaluates every query through the segment callbacks, as the closure path does.
    struct Recomputed<'s, R: Runtime<DType = DType>> {
        segments: &'s ConditionalBandSegments<R>,
    }

    impl<R> SimpsonCallbacks<R> for Recomputed<'_, R>
    where
        R: Runtime<DType = DType>,
        R::Client: Client<R>,
    {
        fn values(
            &mut self,
            a: &mut Arithmetic<'_, R>,
            owner: &Tensor<R>,
            x: &Interval<R>,
        ) -> HydraulicResult<Interval<R>> {
            let gravity = scalar(a, GRAVITY)?;
            integrand(a, self.segments, owner, x, &gravity)
        }

        fn nominal(
            &mut self,
            a: &mut Arithmetic<'_, R>,
            owner: &Tensor<R>,
            point: &Tensor<R>,
        ) -> HydraulicResult<Interval<R>> {
            let point = a.point(point)?;
            let gravity = scalar(a, GRAVITY)?;
            integrand(a, self.segments, owner, &point, &gravity)
        }

        fn fourth(
            &mut self,
            a: &mut Arithmetic<'_, R>,
            owner: &Tensor<R>,
            x: &Interval<R>,
        ) -> HydraulicResult<Tensor<R>> {
            let gravity = scalar(a, GRAVITY)?;
            fourth_bound(a, self.segments, owner, x, &gravity)
        }
    }

    /// Output bits of every callback, in call order.
    struct Recorder<S> {
        inner: S,
        outputs: Vec<Vec<u64>>,
    }

    impl<R: Runtime<DType = DType>, S: SimpsonCallbacks<R>> SimpsonCallbacks<R> for Recorder<S> {
        fn values(
            &mut self,
            a: &mut Arithmetic<'_, R>,
            owner: &Tensor<R>,
            x: &Interval<R>,
        ) -> HydraulicResult<Interval<R>> {
            let value = self.inner.values(a, owner, x)?;
            self.outputs.push(bits(value.lower())?);
            self.outputs.push(bits(value.upper())?);
            Ok(value)
        }

        fn nominal(
            &mut self,
            a: &mut Arithmetic<'_, R>,
            owner: &Tensor<R>,
            point: &Tensor<R>,
        ) -> HydraulicResult<Interval<R>> {
            let value = self.inner.nominal(a, owner, point)?;
            self.outputs.push(bits(value.lower())?);
            self.outputs.push(bits(value.upper())?);
            Ok(value)
        }

        fn fourth(
            &mut self,
            a: &mut Arithmetic<'_, R>,
            owner: &Tensor<R>,
            x: &Interval<R>,
        ) -> HydraulicResult<Tensor<R>> {
            let value = self.inner.fourth(a, owner, x)?;
            self.outputs.push(bits(&value)?);
            Ok(value)
        }
    }

    /// Callback output bits, result bits, and work of one integration over `[0,1]` per lane.
    type Run = (
        Vec<Vec<u64>>,
        Vec<Vec<u64>>,
        crate::channel::flow::surveyed::work::PrismaticRiemannWork,
    );

    fn integrate<R, S>(client: &R::Client, callbacks: S, lanes: usize, panels: usize) -> Result<Run>
    where
        R: Runtime<DType = DType>,
        R::Client: Client<R> + CertifiedSimpsonAlgorithms<R>,
        S: SimpsonCallbacks<R>,
    {
        let config = PrismaticRiemannConfig::default();
        let work = WorkLedger::new(&config);
        let left = Tensor::zeros(&[lanes], DType::F64, client.device())?;
        let right = Tensor::ones(&[lanes], DType::F64, client.device())?;
        let options = CertifiedSimpsonOptions {
            panels,
            max_points: config.max_native_query_points,
        };
        let mut recorder = Recorder {
            inner: callbacks,
            outputs: Vec::new(),
        };
        let result = hydraulic(simpson_contributions(
            client,
            &left,
            &right,
            options,
            &work,
            &mut recorder,
        ))?;
        let mut values = Vec::new();
        for value in [
            result.value(),
            result.lower(),
            result.upper(),
            result.abs_error_bound(),
            result.quadrature_error_bound(),
        ] {
            values.push(hydraulic(bits(value))?);
        }
        Ok((recorder.outputs, values, work.snapshot()))
    }

    fn reused_nodes_match_recomputed<R>(client: &R::Client) -> Result
    where
        R: Runtime<DType = DType>,
        R::Client: Client<R> + CertifiedSimpsonAlgorithms<R>,
    {
        let segments = node_fixture::segments::<R>(client, 0.5)?;
        let rows = Tensor::from_slice(&[3_i64, 0, 1, 3], &[4], client.device())?;
        let selected = hydraulic(checked(client, "node reuse lanes", |a| {
            segments.select(a, &rows)
        }))?;
        for panels in [1, 4, 32] {
            let reused =
                integrate::<R, _>(client, NodeCallbacks::new(&selected, GRAVITY), 4, panels)?;
            let recomputed = integrate::<R, _>(
                client,
                Recomputed {
                    segments: &selected,
                },
                4,
                panels,
            )?;
            assert_eq!(reused.0.len(), 5, "panels={panels}");
            assert_eq!(reused.0, recomputed.0, "callback bits, panels={panels}");
            assert_eq!(reused.1, recomputed.1, "result bits, panels={panels}");
            assert_eq!(reused.2, recomputed.2, "work, panels={panels}");
            assert!(reused.0[4].iter().any(|&bound| f64::from_bits(bound) > 0.0));
            assert_ne!(reused.0[0], reused.0[1], "panels={panels}");
        }
        Ok(())
    }

    fn fourth_without_values_recomputes<R>(client: &R::Client) -> Result
    where
        R: Runtime<DType = DType>,
        R::Client: Client<R>,
    {
        let segments = node_fixture::segments::<R>(client, 0.5)?;
        let owner = Tensor::from_slice(&[0_i64, 3, 1], &[3], client.device())?;
        let lower = Tensor::from_slice(&[0.0, 0.25, 0.5], &[3], client.device())?;
        let upper = Tensor::from_slice(&[0.5, 1.0, 0.75], &[3], client.device())?;
        hydraulic(checked(client, "node fallback fourth", |a| {
            let x = a.bounds(&lower, &upper)?;
            let mut nodes = NodeCallbacks::new(&segments, GRAVITY);
            let cached = nodes.fourth(a, &owner, &x)?;
            let gravity = scalar(a, GRAVITY)?;
            let direct = fourth_bound(a, &segments, &owner, &x, &gravity)?;
            assert_eq!(bits(&cached)?, bits(&direct)?);
            Ok(())
        }))
    }

    /// The error for a layout that `panel_nodes` rejects although it is valid.
    fn layout_error(layout: &str) -> HydraulicError {
        HydraulicError::Numerical {
            context: format!("panel_nodes rejected {layout}"),
        }
    }

    fn panel_layout<R>(client: &R::Client) -> Result
    where
        R: Runtime<DType = DType>,
        R::Client: Client<R>,
    {
        hydraulic(checked(client, "node panel layout", |a| {
            let Some([starts, ends]) = panel_nodes::<R>(a, 18, 8)? else {
                return Err(layout_error("two lanes of four panels"));
            };
            assert_eq!(starts.try_to_vec::<i64>()?, [0, 2, 4, 6, 9, 11, 13, 15]);
            assert_eq!(ends.try_to_vec::<i64>()?, [2, 4, 6, 8, 11, 13, 15, 17]);
            let Some([starts, ends]) = panel_nodes::<R>(a, 3, 1)? else {
                return Err(layout_error("one lane of one panel"));
            };
            assert_eq!(starts.try_to_vec::<i64>()?, [0]);
            assert_eq!(ends.try_to_vec::<i64>()?, [2]);
            for (nodes, panels) in [(19, 8), (16, 8), (13, 4), (0, 0), (5, 0), (1, 1)] {
                assert!(
                    panel_nodes::<R>(a, nodes, panels)?.is_none(),
                    "{nodes}/{panels}"
                );
            }
            Ok(())
        }))
    }

    #[test]
    fn cpu_node_callbacks_reuse_value_states_bit_for_bit() -> Result {
        use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
        let client = CpuClient::new(CpuDevice::new());
        reused_nodes_match_recomputed::<CpuRuntime>(&client)?;
        fourth_without_values_recomputes::<CpuRuntime>(&client)?;
        panel_layout::<CpuRuntime>(&client)
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_node_callbacks_reuse_value_states_bit_for_bit() -> Result {
        use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
        let client = CudaClient::new(CudaDevice::new(0))?;
        reused_nodes_match_recomputed::<CudaRuntime>(&client)?;
        fourth_without_values_recomputes::<CudaRuntime>(&client)?;
        panel_layout::<CudaRuntime>(&client)
    }
}
