#[cfg(test)]
use super::source_sonic_types::SourceSonicCertificates;
use super::{
    point_root_session::PointRootSession,
    source_sonic_binding,
    source_sonic_types::{SonicSide, SourceSonicBinding, SourceSonicRequest, SourceSonicStatus},
    wet_depth_types::WetDepthStatus,
};
use crate::channel::flow::surveyed::error::{HydraulicResult, checked};
use numr::{
    dtype::DType,
    ops::{
        CompareOps, ConditionalOps, IndexingOps, MidpointOps, NextRepresentableOps, ScalarOps,
        SortingOps, TensorOps, TwoSumOps,
    },
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::{
    integrate::CertifiedSimpsonAlgorithms,
    optimize::{
        CertifiedMappedBisectionAlgorithms, CertifiedRootTolerance, CertifiedUpperBracketAlgorithms,
    },
};

impl<R> PointRootSession<'_, '_, R>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + MidpointOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>
        + CertifiedSimpsonAlgorithms<R>
        + CertifiedMappedBisectionAlgorithms<R>
        + CertifiedUpperBracketAlgorithms<R>,
{
    pub(super) fn start_source_sonic(
        &mut self,
        invocation_owner: &Tensor<R>,
        side: SonicSide,
    ) -> HydraulicResult<()> {
        self.sources
            .check_root_witness(self.invocation.source_witness())?;
        self.check_domains()?;
        let c = self.sources.geometry.client();
        let (owners, rows) = checked::<R, _>(c, "source sonic request owner admission", |a| {
            source_sonic_binding::owners(
                a,
                self,
                invocation_owner,
                &self.sonic.status[side.index()],
                SourceSonicStatus::NotRequested as i64,
            )
        })?;
        if owners.numel() == 0 {
            return Ok(());
        }
        let tolerance = CertifiedRootTolerance {
            atol: self.invocation.source_witness().config().root.atol,
            rtol: self.invocation.source_witness().config().root.rtol,
        };
        let request = checked::<R, _>(c, "source sonic request registration", |a| {
            let status = Tensor::full_scalar(
                owners.shape(),
                DType::I64,
                SourceSonicStatus::Unresolved as i64 as f64,
                c.device(),
            )?;
            Ok(SourceSonicRequest {
                binding: SourceSonicBinding::new(&self.invocation, side, owners, rows),
                status,
                dry_residual: super::wave_curve_types::unavailable(a, invocation_owner.numel())?,
                source_residual: super::wave_curve_types::unavailable(a, invocation_owner.numel())?,
                domains: Vec::new(),
                tolerance,
                terminal: false,
                endpoint_points: 0,
                bracket_points: 0,
                bracket_expansions: 0,
            })
        })?;
        self.sonic.status[side.index()] = c.index_put(
            &self.sonic.status[side.index()],
            0,
            &request.binding.owners,
            &request.status,
        )?;
        let index = self.sonic.requests.len();
        self.sonic.requests.push(request);
        let checkpoint = self.checkpoint();
        let result = (|| -> HydraulicResult<()> {
            super::source_sonic_start::locate(
                self.sources,
                &mut self.invocation,
                &mut self.sonic.requests[index],
            )?;
            refine_request(
                self.sources,
                &mut self.invocation,
                &mut self.sonic.requests[index],
                tolerance,
            )
        })();
        if result.is_err() {
            self.sonic.requests[index].terminal = true;
        }
        self.reconcile(checkpoint, result.is_ok())?;
        let request = &self.sonic.requests[index];
        self.sonic.status[side.index()] = c.index_put(
            &self.sonic.status[side.index()],
            0,
            &request.binding.owners,
            &request.status,
        )?;
        result
    }

    #[cfg(test)]
    pub(super) fn refine_source_sonic(
        &mut self,
        tolerance: CertifiedRootTolerance,
    ) -> HydraulicResult<()> {
        self.sources
            .check_root_witness(self.invocation.source_witness())?;
        self.check_domains()?;
        super::wet_depth::check_tolerance(
            &tolerance,
            &self.invocation.source_witness().config().root,
        )?;
        if self.sonic.requests.iter().any(|request| request.terminal) {
            return Err(super::point_root_session::invalid(
                "source sonic refinement remains sealed after an earlier numerical error",
            ));
        }
        let checkpoint = self.checkpoint();
        let result = (|| -> HydraulicResult<()> {
            for request in &mut self.sonic.requests {
                if let Err(error) =
                    refine_request(self.sources, &mut self.invocation, request, tolerance)
                {
                    request.terminal = true;
                    return Err(error);
                }
            }
            Ok(())
        })();
        self.reconcile(checkpoint, result.is_ok())?;
        result
    }

    #[cfg(test)]
    pub(super) fn certify_source_sonic(&mut self) -> HydraulicResult<SourceSonicCertificates<R>> {
        self.sources
            .check_root_witness(self.invocation.source_witness())?;
        self.check_domains()?;
        if self.sonic.requests.iter().any(|request| request.terminal) {
            return Err(super::point_root_session::invalid(
                "source sonic certificates remain sealed after an earlier numerical error",
            ));
        }
        super::source_sonic_certificates::evaluate(self)
    }

    pub(super) fn refine_source_sonic_selected(
        &mut self,
        owners: &Tensor<R>,
        side: SonicSide,
        tolerance: CertifiedRootTolerance,
    ) -> HydraulicResult<()> {
        self.sources
            .check_root_witness(self.invocation.source_witness())?;
        self.check_domains()?;
        super::wet_depth::check_tolerance(
            &tolerance,
            &self.invocation.source_witness().config().root,
        )?;
        if self.sonic.requests.iter().any(|request| request.terminal) {
            return Err(super::point_root_session::invalid(
                "selected source sonic refinement remains sealed after an earlier numerical error",
            ));
        }
        let c = self.sources.geometry.client();
        let demand = checked::<R, _>(c, "source sonic selected complete request admission", |a| {
            let (owners, _) = super::dry_vacuum_evidence::owners(a, self, owners)?;
            let status = c.index_select(&self.sonic.status[side.index()], 0, &owners)?;
            let unrequested = Tensor::full_scalar(
                &[],
                DType::I64,
                SourceSonicStatus::NotRequested as i64 as f64,
                c.device(),
            )?;
            if c.nonzero(&c.eq(&status, &unrequested)?)?.numel() != 0 {
                return Err(super::point_root_session::invalid(
                    "selected sonic refinement includes an unstarted owner and side",
                ));
            }
            let demand = c.index_put(
                &Tensor::zeros(self.invocation_rows.shape(), a.dtype(), c.device())?,
                0,
                &owners,
                &Tensor::ones(owners.shape(), a.dtype(), c.device())?,
            )?;
            for request in &self.sonic.requests {
                if request.binding.side == side
                    && c.nonzero(&c.index_select(&demand, 0, &request.binding.owners)?)?
                        .numel()
                        != 0
                    && (tolerance.atol > request.tolerance.atol
                        || tolerance.rtol > request.tolerance.rtol)
                {
                    return Err(super::point_root_session::invalid(
                        "selected sonic tolerance exceeds the current request allowance",
                    ));
                }
            }
            Ok(demand)
        })?;
        let checkpoint = self.checkpoint();
        let result = (|| -> HydraulicResult<()> {
            for request in &mut self.sonic.requests {
                if request.binding.side != side {
                    continue;
                }
                if c.nonzero(&c.index_select(&demand, 0, &request.binding.owners)?)?
                    .numel()
                    == 0
                {
                    continue;
                }
                if let Err(error) = super::source_sonic_selection::refine(
                    self.sources,
                    &mut self.invocation,
                    request,
                    &demand,
                    &tolerance,
                ) {
                    request.terminal = true;
                    return Err(error);
                }
            }
            Ok(())
        })();
        self.reconcile(checkpoint, result.is_ok())?;
        result
    }
}

fn refine_request<R>(
    sources: &mut super::paired_sources::PairedSources<'_, R>,
    invocation: &mut super::root_invocation::RootInvocation,
    request: &mut SourceSonicRequest<R>,
    tolerance: CertifiedRootTolerance,
) -> HydraulicResult<()>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + MidpointOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>
        + CertifiedSimpsonAlgorithms<R>
        + CertifiedMappedBisectionAlgorithms<R>,
{
    let c = sources.geometry.client();
    let mut output = Vec::with_capacity(request.domains.len());
    for mut domain in std::mem::take(&mut request.domains) {
        request
            .binding
            .check(invocation, domain.side, &domain.binding)?;
        let accepted = checked::<R, _>(c, "source sonic physical depth stopping criterion", |a| {
            super::point_root_depth::satisfied(a, &domain.normalized_depth, &tolerance)
        })?;
        if c.nonzero(&accepted)?.numel() == domain.parent_rows.numel() {
            domain.status = Tensor::full_scalar(
                domain.parent_rows.shape(),
                DType::I64,
                WetDepthStatus::DepthConverged as i64 as f64,
                c.device(),
            )?;
            output.push(domain);
            continue;
        }
        if domain.classification != SourceSonicStatus::PositiveRoot {
            let passed = Tensor::full_scalar(
                domain.parent_rows.shape(),
                DType::I64,
                WetDepthStatus::DepthConverged as i64 as f64,
                c.device(),
            )?;
            let floor = Tensor::full_scalar(
                domain.parent_rows.shape(),
                DType::I64,
                WetDepthStatus::AccuracyFloor as i64 as f64,
                c.device(),
            )?;
            domain.status = c.where_cond(&accepted, &passed, &floor)?;
            output.push(domain);
        } else {
            output.push(super::source_sonic_refinement::advance(
                sources,
                invocation,
                &request.binding,
                domain,
                &tolerance,
            )?);
        }
    }
    request.domains = output;
    request.tolerance = tolerance;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::corpus_coverage::Coverage;
    use super::super::{
        chart_test_support,
        point_root_session::PointRootSession,
        source_sonic_types::{SonicFanMembership, SonicSide, SourceSonicStatus},
        source_tests::support as frozen,
        wet_depth::tests::support,
        wet_root_types::WetRootStatus,
    };
    use frozen::{Result, hydraulic};
    use numr::{dtype::DType, runtime::Runtime};

    /// Checks the exact rectangle sonic root, then the frozen references at the given coverage.
    fn matrix<R>(client: &R::Client, coverage: Coverage) -> Result
    where
        R: Runtime<DType = DType>,
        R::Client: support::DepthClient<R>,
    {
        let config = chart_test_support::config();
        support::with_rectangle::<R, _>(
            client,
            1.0,
            4.0,
            &[[1.0, 0.0, 0.0, 0.0]],
            &config,
            |pair, work| {
                let rows = support::rows::<R>(client, &[0, 0])?;
                let mut session =
                    hydraulic(PointRootSession::new(pair, &rows, config.root.max_points))?;
                let owners = support::rows::<R>(client, &[1, 0])?;
                hydraulic(session.start_source_sonic(&owners, SonicSide::Left))?;
                let points = session.callback_points();
                let before = work.snapshot();
                let result = hydraulic(session.certify_source_sonic())?;
                assert_eq!(session.callback_points(), points);
                assert_eq!(work.snapshot().residual_points, before.residual_points);
                assert_eq!(
                    session.locator_status().try_to_vec::<i64>()?,
                    [WetRootStatus::NotRequested as i64; 2]
                );
                assert_eq!(
                    result.status[0].try_to_vec::<i64>()?,
                    [SourceSonicStatus::PositiveRoot as i64; 2]
                );
                assert_eq!(
                    result.status[1].try_to_vec::<i64>()?,
                    [SourceSonicStatus::NotRequested as i64; 2]
                );
                assert_eq!(result.domains.len(), 1);
                let domain = &result.domains[0];
                assert_eq!(domain.invocation_owner.try_to_vec::<i64>()?, [1, 0]);
                assert_eq!(domain.parent_rows.try_to_vec::<i64>()?, [0, 0]);
                assert_eq!(domain.fan_membership, SonicFanMembership::NotChecked);
                assert_eq!(domain.status.try_to_vec::<i64>()?, [0, 0]);
                // Independent rectangle sonic root: A=4/9, u=2/3, Q=8/27.
                for (actual, expected) in [
                    (&domain.query.area, [0x3fdc71c71c71c71c, 0x3fdc71c71c71c71d]),
                    (&domain.velocity, [0x3fe5555555555555, 0x3fe5555555555556]),
                    (&domain.discharge, [0x3fd2f684bda12f68, 0x3fd2f684bda12f69]),
                ] {
                    let expected = expected.map(f64::from_bits);
                    for actual in frozen::endpoints(actual)? {
                        assert!(
                            actual[0] <= expected[0] && expected[1] <= actual[1],
                            "{actual:?} excludes {expected:?}"
                        );
                    }
                }
                Ok(())
            },
        )?;
        references::run::<R>(client, coverage)?;
        pointed::run::<R>(client)
    }

    #[test]
    fn configured_cpu_source_sonic_roots_preserve_selected_side_and_owner() -> Result {
        use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};
        let original = CpuClient::new(CpuDevice::new());
        let before = original.parallelism();
        let config = ParallelismConfig::new(Some(2), Some(257));
        let client = original.with_parallelism(config);
        matrix::<CpuRuntime>(&client, Coverage::Full)?;
        contacts::run::<CpuRuntime>(&client)?;
        work::run::<CpuRuntime>(&client)?;
        ownership::run::<CpuRuntime>(&client)?;
        assert_eq!(client.parallelism(), config);
        assert_eq!(original.parallelism(), before);
        Ok(())
    }
    #[cfg(feature = "cuda")]
    #[test]
    fn actual_cuda_source_sonic_roots_preserve_selected_side_and_owner() -> Result {
        use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
        matrix::<CudaRuntime>(&CudaClient::new(CudaDevice::new(0))?, Coverage::Sample)
    }

    mod contacts {
        use super::{
            PointRootSession, Result, SonicSide, SourceSonicStatus, chart_test_support, frozen,
            hydraulic, support,
        };
        use numr::{dtype::DType, runtime::Runtime};

        pub(super) fn run<R>(client: &R::Client) -> Result
        where
            R: Runtime<DType = DType>,
            R::Client: support::DepthClient<R>,
        {
            exact_contacts::<R>(client)?;
            uncertain_contact::<R>(client)?;
            vacuum_limits::<R>(client)
        }

        fn exact_contacts<R>(client: &R::Client) -> Result
        where
            R: Runtime<DType = DType>,
            R::Client: support::DepthClient<R>,
        {
            let config = chart_test_support::config();
            let states = [
                [1.0, -2.0, 1.0, 2.0],
                [1.0, 1.0, 1.0, -1.0],
                [1.0, 2.0, 1.0, -2.0],
                [1.0, -3.0, 1.0, 3.0],
                [0.0, 0.0, 0.0, 0.0],
            ];
            support::with_rectangle::<R, _>(client, 1.0, 4.0, &states, &config, |pair, work| {
                let rows = support::rows::<R>(client, &[0, 1, 2, 3, 4])?;
                let mut session =
                    hydraulic(PointRootSession::new(pair, &rows, config.root.max_points))?;
                let owners = support::rows::<R>(client, &[0, 1, 2, 3, 4])?;
                for side in [SonicSide::Left, SonicSide::Right] {
                    hydraulic(session.start_source_sonic(&owners, side))?;
                }
                let before = work.snapshot();
                let result = hydraulic(session.certify_source_sonic())?;
                assert_eq!(work.snapshot().residual_points, before.residual_points);
                let expected = [
                    SourceSonicStatus::DryContact,
                    SourceSonicStatus::SourceContact,
                    SourceSonicStatus::NoRootSource,
                    SourceSonicStatus::NoRootDry,
                    SourceSonicStatus::DrySourceUnavailable,
                ]
                .map(|s| s as i64);
                assert_eq!(result.status[0].try_to_vec::<i64>()?, expected);
                assert_eq!(result.status[1].try_to_vec::<i64>()?, expected);
                assert_eq!(result.domains.len(), 4);
                for domain in &result.domains {
                    super::diagnostics::check(domain)?;
                    assert_eq!(domain.status.try_to_vec::<i64>()?, [0]);
                    match domain.classification {
                        SourceSonicStatus::DryContact => {
                            for value in [
                                &domain.query.area,
                                &domain.depth,
                                &domain.normalized_depth,
                                &domain.sound,
                                &domain.velocity,
                                &domain.discharge,
                            ] {
                                assert_eq!(frozen::endpoints(value)?, [[0.0, 0.0]]);
                            }
                        }
                        SourceSonicStatus::SourceContact => {
                            let sign = if domain.side == SonicSide::Left {
                                1.0
                            } else {
                                -1.0
                            };
                            assert_eq!(frozen::endpoints(&domain.query.area)?, [[1.0, 1.0]]);
                            assert_eq!(frozen::endpoints(&domain.discharge)?, [[sign, sign]]);
                            assert_eq!(
                                frozen::components(&domain.anchored_discharge)?,
                                frozen::components(&domain.source_discharge)?
                            );
                        }
                        _ => return Err("Noncontact outcome manufactured a contact query".into()),
                    }
                }
                for evidence in &result.evidence {
                    assert_eq!(evidence.status.try_to_vec::<i64>()?, expected);
                    assert_eq!(evidence.dry_residual.available.try_to_vec::<f64>()?[4], 0.0);
                    assert_eq!(
                        evidence.source_residual.available.try_to_vec::<f64>()?[4],
                        0.0
                    );
                }
                Ok(())
            })
        }

        fn uncertain_contact<R>(client: &R::Client) -> Result
        where
            R: Runtime<DType = DType>,
            R::Client: support::DepthClient<R>,
        {
            let config = chart_test_support::config();
            let section = frozen::batch::<R>(client)?;
            let profile = frozen::cases()
                .iter()
                .find(|case| case.id == "uniform_cap_retained_deficit_dyadic_velocity")
                .ok_or("Missing independent cap profile")?
                .owner;
            let owners = support::rows::<R>(client, &[i64::try_from(profile)?])?;
            // Exact rational Q=14*binary64(0x3ff52a7fa9d2f8ea), frozen as two exact components.
            // The represented velocity is near sqrt(14/8), while the complete sound certificate overlaps it.
            let input = [
                [14.0_f64.to_bits(), 0, 0],
                [0x4032852fb49899cd, 0xbcd0000000000000, 0],
                [14.0_f64.to_bits(), 0, 0],
                [0xc032852fb49899cd, 0x3cd0000000000000, 0],
            ]
            .iter()
            .map(|point| frozen::expansion::<R>(client, &[*point], true))
            .collect::<Result<Vec<_>>>()?;
            chart_test_support::with_inputs(client, section, owners, input, &config, |pair, _| {
                let rows = support::rows::<R>(client, &[0])?;
                let mut session =
                    hydraulic(PointRootSession::new(pair, &rows, config.root.max_points))?;
                for side in [SonicSide::Left, SonicSide::Right] {
                    hydraulic(session.start_source_sonic(&rows, side))?;
                }
                let result = hydraulic(session.certify_source_sonic())?;
                assert!(result.domains.is_empty());
                for status in &result.status {
                    assert_eq!(
                        status.try_to_vec::<i64>()?,
                        [SourceSonicStatus::Unresolved as i64]
                    );
                }
                for evidence in &result.evidence {
                    let residual = frozen::endpoints(&evidence.source_residual.physical)?[0];
                    assert!(residual[0] <= 0.0 && residual[1] >= 0.0);
                    assert!(
                        frozen::components(&evidence.source_residual.retained)?[0]
                            .map(f64::from_bits)
                            .iter()
                            .any(|&x| x != 0.0)
                    );
                }
                Ok(())
            })
        }

        fn vacuum_limits<R>(client: &R::Client) -> Result
        where
            R: Runtime<DType = DType>,
            R::Client: support::DepthClient<R>,
        {
            let config = chart_test_support::config();
            for (id, expected) in [
                (
                    "scalar_rectangle_vacuum_threshold",
                    SourceSonicStatus::DryContact,
                ),
                ("scalar_rectangle_vacuum", SourceSonicStatus::NoRootDry),
            ] {
                chart_test_support::with_pair::<R, _>(client, &[id], &config, |pair, _| {
                    let rows = support::rows::<R>(client, &[0])?;
                    let mut session =
                        hydraulic(PointRootSession::new(pair, &rows, config.root.max_points))?;
                    for side in [SonicSide::Left, SonicSide::Right] {
                        hydraulic(session.start_source_sonic(&rows, side))?;
                    }
                    let result = hydraulic(session.certify_source_sonic())?;
                    for status in &result.status {
                        assert_eq!(status.try_to_vec::<i64>()?, [expected as i64]);
                    }
                    if expected == SourceSonicStatus::DryContact {
                        assert_eq!(result.domains.len(), 2);
                        for domain in &result.domains {
                            // Frozen sonic-dry-limit label and zero limiting front speed establish these values without Q/A.
                            for value in [
                                &domain.query.area,
                                &domain.depth,
                                &domain.velocity,
                                &domain.discharge,
                            ] {
                                assert_eq!(frozen::endpoints(value)?, [[0.0, 0.0]]);
                            }
                        }
                    } else {
                        assert!(result.domains.is_empty());
                        for evidence in &result.evidence {
                            assert!(
                                frozen::endpoints(&evidence.dry_residual.physical)?[0][0] > 0.0
                            );
                        }
                    }
                    Ok(())
                })?;
            }
            Ok(())
        }
    }

    mod diagnostics {
        use super::{Result, frozen, support};
        use crate::channel::flow::surveyed_points::source_sonic_types::{
            SonicFanMembership, SourceSonicCertificate,
        };
        use numr::{dtype::DType, runtime::Runtime};

        pub(super) fn check<R>(domain: &SourceSonicCertificate<R>) -> Result
        where
            R: Runtime<DType = DType>,
            R::Client: support::DepthClient<R>,
        {
            assert_eq!(domain.fan_membership, SonicFanMembership::NotChecked);
            let n = domain.invocation_owner.numel();
            assert_eq!(domain.parent_rows.numel(), n);
            assert_eq!(domain.query.query.original_rows.numel(), n);
            assert_eq!(domain.depth_status.numel(), n);
            for value in [
                &domain.normalized_depth,
                &domain.sound,
                &domain.velocity,
                &domain.discharge,
                &domain.residual,
            ] {
                for [lower, upper] in frozen::endpoints(value)? {
                    assert!(lower.is_finite() && upper.is_finite() && lower <= upper);
                }
            }
            for [lower, upper] in frozen::endpoints(&domain.residual)? {
                assert!(lower <= 0.0 && upper >= 0.0);
            }
            for value in [
                &domain.source_discharge,
                &domain.area_difference,
                &domain.velocity_change,
                &domain.discharge_change,
                &domain.anchored_discharge,
            ] {
                assert!(
                    value
                        .abs_error()
                        .try_to_vec::<f64>()?
                        .iter()
                        .all(|&x| x.is_finite() && x >= 0.0)
                );
            }
            assert!(domain.callback_points >= domain.iterations);
            assert!(domain.mapped_points >= domain.iterations);
            Ok(())
        }

        pub(super) fn evidence<R: Runtime<DType = DType>>(
            result: &crate::channel::flow::surveyed_points::source_sonic_types::SourceSonicCertificates<R>,
        ) -> Result {
            assert!(result.callback_points <= result.point_limit);
            let mut reported = result
                .domains
                .iter()
                .map(|d| d.callback_points)
                .sum::<usize>();
            for item in &result.evidence {
                let owners = item.invocation_owner.try_to_vec::<i64>()?;
                assert_eq!(item.parent_rows.numel(), owners.len());
                assert_eq!(item.status.numel(), owners.len());
                assert_eq!(item.dry_residual.available.numel(), owners.len());
                assert_eq!(item.source_residual.available.numel(), owners.len());
                let global = result.status[item.side.index()].try_to_vec::<i64>()?;
                let status = item.status.try_to_vec::<i64>()?;
                for (index, owner) in owners.into_iter().enumerate() {
                    assert_eq!(global[usize::try_from(owner)?], status[index]);
                }
                assert!(item.bracket_expansions <= item.bracket_points);
                reported += item.endpoint_points + item.bracket_points;
            }
            assert!(reported <= result.callback_points);
            Ok(())
        }
    }

    mod ownership {
        use super::{
            PointRootSession, Result, SonicSide, SourceSonicStatus, chart_test_support, frozen,
            hydraulic, support,
        };
        use crate::channel::flow::surveyed_points::{
            root_invocation::RootInvocation, wet_root_types::WetRootStatus,
        };
        use numr::{
            dtype::DType,
            ops::BinaryOps,
            runtime::{Runtime, RuntimeClient},
            tensor::Tensor,
        };
        use solvr::optimize::CertifiedRootTolerance;

        pub(super) fn run<R>(client: &R::Client) -> Result
        where
            R: Runtime<DType = DType>,
            R::Client: support::DepthClient<R>,
        {
            mixed_requests::<R>(client)?;
            mutations::<R>(client)?;
            nested_continuation::<R>(client)?;
            foreign_invocation::<R>(client, false)?;
            foreign_invocation::<R>(client, true)?;
            stale_policy::<R>(client)?;
            owned_snapshots::<R>(client)
        }

        fn mixed_requests<R>(client: &R::Client) -> Result
        where
            R: Runtime<DType = DType>,
            R::Client: support::DepthClient<R>,
        {
            let config = chart_test_support::config();
            support::with_rectangle::<R, _>(
                client,
                1.0,
                4.0,
                &[
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [1.0, 0.0, 1.0, 0.0],
                ],
                &config,
                |pair, work| {
                    let parents = support::rows::<R>(client, &[2, 0, 2, 1])?;
                    let mut session = hydraulic(PointRootSession::new(
                        pair,
                        &parents,
                        config.root.max_points,
                    ))?;
                    hydraulic(session.start_wet(&support::rows::<R>(client, &[2, 0])?))?;
                    assert_eq!(
                        session.domains()[0].invocation_owner.try_to_vec::<i64>()?,
                        [2, 0]
                    );
                    assert_eq!(
                        session.domains()[0].parent_rows.try_to_vec::<i64>()?,
                        [2, 2]
                    );
                    let located_points = session.callback_points();
                    hydraulic(
                        session.start_source_sonic(
                            &support::rows::<R>(client, &[1])?,
                            SonicSide::Left,
                        ),
                    )?;
                    hydraulic(
                        session.start_source_sonic(
                            &support::rows::<R>(client, &[3])?,
                            SonicSide::Right,
                        ),
                    )?;
                    assert_eq!(
                        session.locator_status().try_to_vec::<i64>()?,
                        [
                            WetRootStatus::Located as i64,
                            WetRootStatus::NotRequested as i64,
                            WetRootStatus::Located as i64,
                            WetRootStatus::NotRequested as i64
                        ]
                    );
                    let before = work.snapshot();
                    let cache = chart_test_support::cache(session.sources())?;
                    for bad in [vec![1, 1], vec![-1], vec![4]] {
                        assert!(
                            session
                                .start_source_sonic(
                                    &support::rows::<R>(client, &bad)?,
                                    SonicSide::Left
                                )
                                .is_err()
                        );
                    }
                    assert!(
                        session
                            .start_source_sonic(&support::rows::<R>(client, &[1])?, SonicSide::Left)
                            .is_err()
                    );
                    assert!(
                        session
                            .start_wet(&support::rows::<R>(client, &[0])?)
                            .is_err()
                    );
                    assert!(
                        session
                            .start_wet(&support::rows::<R>(client, &[1])?)
                            .is_err()
                    );
                    assert_eq!(work.snapshot(), before);
                    assert_eq!(chart_test_support::cache(session.sources())?, cache);
                    // Duplicate source rows remain independent invocation owners and can start separate histories.
                    hydraulic(
                        session.start_source_sonic(
                            &support::rows::<R>(client, &[0])?,
                            SonicSide::Left,
                        ),
                    )?;
                    hydraulic(
                        session.start_source_sonic(
                            &support::rows::<R>(client, &[2])?,
                            SonicSide::Left,
                        ),
                    )?;
                    hydraulic(session.refine_wet_depth_in_place(CertifiedRootTolerance {
                        atol: 0.0,
                        rtol: 0.0,
                    }))?;
                    hydraulic(session.refine_source_sonic(CertifiedRootTolerance {
                        atol: 1e-17,
                        rtol: 0.0,
                    }))?;
                    let result = hydraulic(session.certify_source_sonic())?;
                    super::diagnostics::evidence(&result)?;
                    assert_eq!(result.invocation_rows.try_to_vec::<i64>()?, [2, 0, 2, 1]);
                    for domain in &result.domains {
                        let owners = domain.invocation_owner.try_to_vec::<i64>()?;
                        let rows = domain.parent_rows.try_to_vec::<i64>()?;
                        let originals = domain.query.query.original_rows.try_to_vec::<i64>()?;
                        for (index, owner) in owners.into_iter().enumerate() {
                            assert_eq!(rows[index], [2, 0, 2, 1][usize::try_from(owner)?]);
                            assert_eq!(originals[index], 101 + rows[index]);
                        }
                    }
                    let sonic_points = session
                        .sonic
                        .requests
                        .iter()
                        .map(|request| {
                            request.endpoint_points
                                + request.bracket_points
                                + request
                                    .domains
                                    .iter()
                                    .map(|domain| domain.callback_points)
                                    .sum::<usize>()
                        })
                        .sum::<usize>();
                    let wet_points = session
                        .domains()
                        .iter()
                        .map(|domain| domain.callback_points)
                        .sum::<usize>();
                    assert_eq!(
                        session.callback_points(),
                        located_points + sonic_points + wet_points
                    );
                    assert_eq!(session.callback_points(), work.snapshot().residual_points);
                    assert_eq!(session.point_limit(), config.root.max_points);
                    Ok(())
                },
            )
        }

        fn mutations<R>(client: &R::Client) -> Result
        where
            R: Runtime<DType = DType>,
            R::Client: support::DepthClient<R>,
        {
            let config = chart_test_support::config();
            for mutation in 0..6 {
                let state = if mutation == 5 {
                    [1.0, 2.0, 1.0, 0.0]
                } else {
                    [1.0, 0.0, 1.0, 0.0]
                };
                support::with_rectangle::<R, _>(
                    client,
                    1.0,
                    4.0,
                    &[state],
                    &config,
                    |pair, work| {
                        let rows = support::rows::<R>(client, &[0, 0])?;
                        let mut session =
                            hydraulic(PointRootSession::new(pair, &rows, config.root.max_points))?;
                        let owner = support::rows::<R>(client, &[0])?;
                        hydraulic(session.start_source_sonic(&owner, SonicSide::Left))?;
                        match mutation {
                            0 => session.sonic.requests[0].domains[0].side = SonicSide::Right,
                            1 => {
                                session.sonic.requests[0].domains[0].parent_rows =
                                    support::rows::<R>(client, &[1])?
                            }
                            2 => {
                                session.sonic.requests[0].domains[0].invocation_owner =
                                    support::rows::<R>(client, &[1])?
                            }
                            3 => {
                                session.sonic.requests[0].domains[0].iterations =
                                    config.root.max_iterations + 1
                            }
                            4 => {
                                session.invocation = hydraulic(RootInvocation::new(
                                    hydraulic(session.sources.root_witness())?,
                                    config.root.max_points,
                                ))?
                            }
                            5 => {
                                assert!(session.sonic.requests[0].domains.is_empty());
                                assert_eq!(
                                    session.sonic.requests[0].status.try_to_vec::<i64>()?,
                                    [SourceSonicStatus::NoRootSource as i64]
                                );
                                session.sonic.requests[0].binding.side = SonicSide::Right;
                            }
                            _ => return Err("Unexpected binding mutation".into()),
                        }
                        let before = work.snapshot();
                        let cache = chart_test_support::cache(session.sources())?;
                        assert!(session.certify_source_sonic().is_err());
                        assert!(
                            session
                                .refine_source_sonic(CertifiedRootTolerance {
                                    atol: config.root.atol,
                                    rtol: config.root.rtol
                                })
                                .is_err()
                        );
                        assert!(
                            session
                                .start_source_sonic(
                                    &support::rows::<R>(client, &[1])?,
                                    SonicSide::Right
                                )
                                .is_err()
                        );
                        assert_eq!(work.snapshot(), before);
                        assert_eq!(chart_test_support::cache(session.sources())?, cache);
                        Ok(())
                    },
                )?;
            }
            Ok(())
        }

        fn nested_continuation<R>(client: &R::Client) -> Result
        where
            R: Runtime<DType = DType>,
            R::Client: support::DepthClient<R>,
        {
            let config = chart_test_support::config();
            support::with_rectangle::<R, _>(
                client,
                1.0,
                4.0,
                &[[1.0, 0.0, 1.0, 0.0]],
                &config,
                |pair, work| {
                    let rows = support::rows::<R>(client, &[0, 0])?;
                    let mut session =
                        hydraulic(PointRootSession::new(pair, &rows, config.root.max_points))?;
                    for owner in [0, 1] {
                        hydraulic(session.start_source_sonic(
                            &support::rows::<R>(client, &[owner])?,
                            SonicSide::Left,
                        ))?;
                    }
                    let donor = session.sonic.requests[0]
                        .domains
                        .pop()
                        .ok_or("Missing donor root")?;
                    let state = donor
                        .refinement
                        .ok_or("Donor lacks initialized continuation")?;
                    assert!(session.sonic.requests[1].domains[0].refinement.is_some());
                    session.sonic.requests[1].domains[0].refinement = Some(state);
                    let before = work.snapshot();
                    let cache = chart_test_support::cache(session.sources())?;
                    assert!(session.certify_source_sonic().is_err());
                    assert!(
                        session
                            .refine_source_sonic(CertifiedRootTolerance {
                                atol: config.root.atol,
                                rtol: config.root.rtol
                            })
                            .is_err()
                    );
                    assert_eq!(work.snapshot(), before);
                    assert_eq!(chart_test_support::cache(session.sources())?, cache);
                    Ok(())
                },
            )
        }

        fn foreign_invocation<R>(client: &R::Client, initialized: bool) -> Result
        where
            R: Runtime<DType = DType>,
            R::Client: support::DepthClient<R>,
        {
            let config = chart_test_support::config();
            let id = if initialized {
                "scalar_rectangle_right_dry_sonic"
            } else {
                "retained_vacuum_inside_film"
            };
            let side = if initialized {
                SonicSide::Left
            } else {
                SonicSide::Right
            };
            chart_test_support::with_pair::<R, _>(client, &[id], &config, |pair, work| {
                let rows = support::rows::<R>(client, &[0])?;
                let donor = {
                    let mut first =
                        hydraulic(PointRootSession::new(pair, &rows, config.root.max_points))?;
                    hydraulic(first.start_source_sonic(&rows, side))?;
                    assert_eq!(
                        first.sonic.requests[0].domains[0].refinement.is_some(),
                        initialized
                    );
                    first.sonic.requests[0]
                        .domains
                        .pop()
                        .ok_or("Missing foreign source-sonic domain")?
                };
                let mut second =
                    hydraulic(PointRootSession::new(pair, &rows, config.root.max_points))?;
                hydraulic(second.start_source_sonic(&rows, side))?;
                second.sonic.requests[0].domains[0] = donor;
                let before = work.snapshot();
                let cache = chart_test_support::cache(second.sources())?;
                assert!(second.certify_source_sonic().is_err());
                assert!(
                    second
                        .refine_source_sonic(CertifiedRootTolerance {
                            atol: config.root.atol,
                            rtol: config.root.rtol
                        })
                        .is_err()
                );
                assert_eq!(work.snapshot(), before);
                assert_eq!(chart_test_support::cache(second.sources())?, cache);
                Ok(())
            })
        }

        fn stale_policy<R>(client: &R::Client) -> Result
        where
            R: Runtime<DType = DType>,
            R::Client: support::DepthClient<R>,
        {
            let config = chart_test_support::config();
            support::with_rectangle::<R, _>(
                client,
                1.0,
                4.0,
                &[[1.0, 0.0, 1.0, 0.0]],
                &config,
                |pair, work| {
                    let rows = support::rows::<R>(client, &[0])?;
                    let mut session =
                        hydraulic(PointRootSession::new(pair, &rows, config.root.max_points))?;
                    hydraulic(session.start_source_sonic(&rows, SonicSide::Left))?;
                    session.sources.evaluation.panels *= 2;
                    let before = work.snapshot();
                    let cache = chart_test_support::cache(session.sources())?;
                    assert!(session.start_source_sonic(&rows, SonicSide::Right).is_err());
                    assert!(
                        session
                            .refine_source_sonic(CertifiedRootTolerance {
                                atol: 0.0,
                                rtol: 0.0
                            })
                            .is_err()
                    );
                    assert!(session.certify_source_sonic().is_err());
                    assert_eq!(work.snapshot(), before);
                    assert_eq!(chart_test_support::cache(session.sources())?, cache);
                    Ok(())
                },
            )
        }

        fn owned_snapshots<R>(client: &R::Client) -> Result
        where
            R: Runtime<DType = DType>,
            R::Client: support::DepthClient<R>,
        {
            let config = chart_test_support::config();
            support::with_rectangle::<R, _>(
                client,
                1.0,
                4.0,
                &[[1.0, 0.0, 1.0, 0.0]],
                &config,
                |pair, _| {
                    let rows = support::rows::<R>(client, &[0])?;
                    let mut session =
                        hydraulic(PointRootSession::new(pair, &rows, config.root.max_points))?;
                    hydraulic(session.start_source_sonic(&rows, SonicSide::Left))?;
                    let parameter = frozen::endpoints(
                        &session.sonic.requests[0].domains[0].query.query.parameter,
                    )?;
                    let depth = frozen::endpoints(&session.sonic.requests[0].domains[0].depth)?;
                    let evidence =
                        frozen::endpoints(&session.sonic.requests[0].source_residual.physical)?;
                    let result = hydraulic(session.certify_source_sonic())?;
                    let zero = Tensor::zeros(&[1], DType::F64, client.device())?;
                    let changed = Tensor::full_scalar(&[1], DType::I64, 99.0, client.device())?;
                    client.copy_into(result.domains[0].query.query.parameter.lower(), &zero)?;
                    client.copy_into(result.domains[0].depth.lower(), &zero)?;
                    client.copy_into(result.evidence[0].source_residual.physical.lower(), &zero)?;
                    client.copy_into(&result.invocation_rows, &changed)?;
                    assert_eq!(
                        frozen::endpoints(
                            &session.sonic.requests[0].domains[0].query.query.parameter
                        )?,
                        parameter
                    );
                    assert_eq!(
                        frozen::endpoints(&session.sonic.requests[0].domains[0].depth)?,
                        depth
                    );
                    assert_eq!(
                        frozen::endpoints(&session.sonic.requests[0].source_residual.physical)?,
                        evidence
                    );
                    assert_eq!(session.invocation_rows().try_to_vec::<i64>()?, [0]);
                    hydraulic(session.certify_source_sonic())?;
                    Ok(())
                },
            )
        }
    }

    mod pointed {
        use super::{
            PointRootSession, Result, SonicSide, SourceSonicStatus, chart_test_support, frozen,
            hydraulic, support,
        };
        use crate::channel::{
            flow::surveyed_points::source_sonic_charts,
            section::{CrossSectionBatch, CrossSectionDefinition, CrossSectionPoint},
        };
        use numr::{
            dtype::DType,
            runtime::{Runtime, RuntimeClient},
            tensor::Tensor,
        };
        use solvr::optimize::CertifiedRootTolerance;

        pub(super) fn run<R>(client: &R::Client) -> Result
        where
            R: Runtime<DType = DType>,
            R::Client: support::DepthClient<R>,
        {
            // Independent control: A=h²,g=2,c=A^(1/4),Phi=4*c. Epsilon=5*2^-83 gives Asonic=2^-332.
            for upper_exponent in [-331, -330] {
                let mut config = chart_test_support::config();
                config.gravity_m_per_s2 = 2.0;
                let section = CrossSectionBatch::<R>::new(
                    client.clone(),
                    vec![CrossSectionDefinition {
                        id: "pointed_source_sonic_control".into(),
                        vertical_datum: "Exact pointed datum".into(),
                        points: [(-4.0, 4.0), (0.0, 0.0), (4.0, 4.0)]
                            .into_iter()
                            .map(|(station_m, elevation_m)| CrossSectionPoint {
                                station_m,
                                elevation_m,
                            })
                            .collect(),
                        maximum_stage_m: 4.0,
                        profile_provenance: "Independent A=h² identity".into(),
                    }],
                )?
                .certify_prismatic()?;
                let owners = support::rows::<R>(client, &[0, 0])?;
                let epsilon = 5.0 * 2.0_f64.powi(-83);
                let input = [
                    [[0, 0, 0], [1.0_f64.to_bits(), 0, 0]],
                    [[0, 0, 0], [(-4.0_f64).to_bits(), epsilon.to_bits(), 0]],
                    [[1.0_f64.to_bits(), 0, 0], [0, 0, 0]],
                    [[4.0_f64.to_bits(), (-epsilon).to_bits(), 0], [0, 0, 0]],
                ]
                .iter()
                .map(|values| frozen::expansion::<R>(client, values, true))
                .collect::<Result<Vec<_>>>()?;
                chart_test_support::with_inputs(
                    client,
                    section,
                    owners,
                    input,
                    &config,
                    |pair, work| {
                        let rows = support::rows::<R>(client, &[0, 1])?;
                        let mut session =
                            hydraulic(PointRootSession::new(pair, &rows, config.root.max_points))?;
                        hydraulic(session.start_source_sonic(
                            &support::rows::<R>(client, &[1])?,
                            SonicSide::Left,
                        ))?;
                        hydraulic(session.start_source_sonic(
                            &support::rows::<R>(client, &[0])?,
                            SonicSide::Right,
                        ))?;
                        for request in &mut session.sonic.requests {
                            assert_eq!(request.domains.len(), 1);
                            assert!(request.domains[0].refinement.is_none());
                            assert_eq!(
                                request.domains[0].classification,
                                SourceSonicStatus::PositiveRoot
                            );
                            let domain = &mut request.domains[0];
                            let zero = Tensor::zeros(&[1], DType::F64, client.device())?;
                            let upper = Tensor::full_scalar(
                                &[1],
                                DType::F64,
                                2.0_f64.powi(upper_exponent),
                                client.device(),
                            )?;
                            // Enlarge only a deferred supported bracket around the independently known root.
                            domain.query = hydraulic(source_sonic_charts::chart(
                                session.sources,
                                &domain.parent_rows,
                                domain.side,
                                &zero,
                                &upper,
                            ))?;
                            let (depth, normalized_depth) = hydraulic(
                    crate::channel::flow::surveyed_points::point_root_depth::certificates(
                        session.sources,
                        &domain.parent_rows,
                        &domain.query,
                    ),
                )?;
                            domain.depth = depth;
                            domain.normalized_depth = normalized_depth;
                        }
                        let points_before = session.callback_points();
                        hydraulic(session.refine_source_sonic(CertifiedRootTolerance {
                            atol: 2.0_f64.powi(-210),
                            rtol: 0.0,
                        }))?;
                        let result = hydraulic(session.certify_source_sonic())?;
                        assert_eq!(result.domains.len(), 2);
                        for domain in &result.domains {
                            super::diagnostics::check(domain)?;
                            assert_eq!(domain.status.try_to_vec::<i64>()?, [0]);
                            assert!(
                                domain.iterations > 0
                                    && domain.iterations <= config.root.max_iterations
                            );
                            let sign = if domain.side == SonicSide::Left {
                                1.0
                            } else {
                                -1.0
                            };
                            for (actual, expected) in [
                                (&domain.query.area, 2.0_f64.powi(-332)),
                                (&domain.depth, 2.0_f64.powi(-166)),
                                (&domain.velocity, sign * 2.0_f64.powi(-83)),
                                (&domain.discharge, sign * 2.0_f64.powi(-415)),
                            ] {
                                let [lower, upper] = frozen::endpoints(actual)?[0];
                                assert!(
                                    lower <= expected && expected <= upper,
                                    "{lower:e}..{upper:e} excludes {expected:e}"
                                );
                                assert!((upper - lower) / expected.abs() < 1e-10);
                            }
                        }
                        assert!(session.callback_points() > points_before);
                        let before = work.snapshot();
                        let points = session.callback_points();
                        let iterations = result
                            .domains
                            .iter()
                            .map(|d| d.iterations)
                            .collect::<Vec<_>>();
                        // Stricter finite stopping resumes the retained exact-root state without another endpoint admission.
                        hydraulic(session.refine_source_sonic(CertifiedRootTolerance {
                            atol: 2.0_f64.powi(-230),
                            rtol: 0.0,
                        }))?;
                        let after = hydraulic(session.certify_source_sonic())?;
                        assert_eq!(
                            after
                                .domains
                                .iter()
                                .map(|d| d.iterations)
                                .collect::<Vec<_>>(),
                            iterations
                        );
                        assert_eq!(session.callback_points(), points);
                        assert_eq!(work.snapshot().residual_points, before.residual_points);
                        Ok(())
                    },
                )?;
            }
            Ok(())
        }
    }

    mod references {
        use super::{
            PointRootSession, Result, SonicSide, SourceSonicStatus, chart_test_support, frozen,
            hydraulic, support,
        };
        use crate::channel::flow::surveyed_points::corpus_coverage::Coverage;
        use numr::{dtype::DType, runtime::Runtime};
        use solvr::optimize::CertifiedRootTolerance;

        struct Reference {
            id: &'static str,
            side: SonicSide,
            area: [u64; 2],
            depth: [u64; 2],
            velocity: [u64; 2],
            discharge: [u64; 2],
        }
        const REFERENCES: &[Reference] = &[
            Reference {
                id: "retained_vacuum_inside_film",
                side: SonicSide::Right,
                area: [0x35bc71c71c71c71c, 0x35bc71c71c71c71d],
                depth: [0x35bc71c71c71c71c, 0x35bc71c71c71c71d],
                velocity: [0xbad5555555555556, 0xbad5555555555555],
                discharge: [0xb0a2f684bda12f69, 0xb0a2f684bda12f68],
            },
            Reference {
                id: "tiny_rectangle_right_dry_fan",
                side: SonicSide::Left,
                area: [0x1a5c71c71c71c71c, 0x1a5c71c71c71c71d],
                depth: [0x1a5c71c71c71c71c, 0x1a5c71c71c71c71d],
                velocity: [0x2d25555555555555, 0x2d25555555555556],
                discharge: [0x0792f684bda12f68, 0x0792f684bda12f69],
            },
            Reference {
                id: "scalar_rectangle_right_dry_sonic",
                side: SonicSide::Left,
                area: [0x3fdc71c71c71c71c, 0x3fdc71c71c71c71d],
                depth: [0x3fdc71c71c71c71c, 0x3fdc71c71c71c71d],
                velocity: [0x3fe5555555555555, 0x3fe5555555555556],
                discharge: [0x3fd2f684bda12f68, 0x3fd2f684bda12f69],
            },
            Reference {
                id: "scalar_rectangle_left_dry_sonic",
                side: SonicSide::Right,
                area: [0x3fdc71c71c71c71c, 0x3fdc71c71c71c71d],
                depth: [0x3fdc71c71c71c71c, 0x3fdc71c71c71c71d],
                velocity: [0xbfe5555555555556, 0xbfe5555555555555],
                discharge: [0xbfd2f684bda12f69, 0xbfd2f684bda12f68],
            },
            Reference {
                id: "scalar_triangle_right_dry_sonic",
                side: SonicSide::Left,
                area: [0x3ffa36e2eb1c432c, 0x3ffa36e2eb1c432d],
                depth: [0x3ff47ae147ae147a, 0x3ff47ae147ae147b],
                velocity: [0x3fe9999999999999, 0x3fe999999999999a],
                discharge: [0x3ff4f8b588e368f0, 0x3ff4f8b588e368f1],
            },
            Reference {
                id: "scalar_triangle_left_dry_sonic",
                side: SonicSide::Right,
                area: [0x3ffa36e2eb1c432c, 0x3ffa36e2eb1c432d],
                depth: [0x3ff47ae147ae147a, 0x3ff47ae147ae147b],
                velocity: [0xbfe999999999999a, 0xbfe9999999999999],
                discharge: [0xbff4f8b588e368f1, 0xbff4f8b588e368f0],
            },
            Reference {
                id: "scalar_three_bands_both_knots_dry_fan",
                side: SonicSide::Left,
                area: [0x4013f08ac385e532, 0x4013f08ac385e533],
                depth: [0x3ffaefa8a3a38101, 0x3ffaefa8a3a38102],
                velocity: [0x3fedd625462a7e0a, 0x3fedd625462a7e0b],
                discharge: [0x4012976d9b060300, 0x4012976d9b060301],
            },
            Reference {
                id: "scalar_three_bands_both_knots_dry_fan_reverse",
                side: SonicSide::Right,
                area: [0x4013f08ac385e532, 0x4013f08ac385e533],
                depth: [0x3ffaefa8a3a38101, 0x3ffaefa8a3a38102],
                velocity: [0xbfedd625462a7e0b, 0xbfedd625462a7e0a],
                discharge: [0xc012976d9b060301, 0xc012976d9b060300],
            },
        ];

        pub(super) fn run<R>(client: &R::Client, coverage: Coverage) -> Result
        where
            R: Runtime<DType = DType>,
            R::Client: support::DepthClient<R>,
        {
            let replayed = coverage.select(REFERENCES, |reference| {
                (
                    reference.side == SonicSide::Left,
                    reference.id.starts_with("scalar_"),
                )
            });
            // Only explicitly labelled sonic candidates supply these truths. Legacy velocities use independent rational Q/A bounds.
            for reference in replayed {
                let config = chart_test_support::config();
                chart_test_support::with_pair::<R, _>(
                    client,
                    &[reference.id],
                    &config,
                    |pair, work| {
                        let cap = pair.bands.cap_depth.upper().try_to_vec::<f64>()?[0];
                        let parents = support::rows::<R>(client, &[0])?;
                        let mut session = hydraulic(PointRootSession::new(
                            pair,
                            &parents,
                            config.root.max_points,
                        ))?;
                        let owners = support::rows::<R>(client, &[0])?;
                        hydraulic(session.start_source_sonic(&owners, reference.side))?;
                        if f64::from_bits(reference.depth[0]) < 1e-30 {
                            let atol = (f64::from_bits(reference.depth[0]) / cap) * 1e-12;
                            assert!(atol > 0.0 && atol < config.root.atol);
                            hydraulic(
                                session.refine_source_sonic(CertifiedRootTolerance {
                                    atol,
                                    rtol: 0.0,
                                }),
                            )?;
                        }
                        let cache = chart_test_support::cache(session.sources())?;
                        assert!(
                            cache[1 - reference.side.index()]
                                .ready
                                .iter()
                                .all(|&x| x == 0.0)
                        );
                        let before = work.snapshot();
                        let result = hydraulic(session.certify_source_sonic())?;
                        assert_eq!(work.snapshot().residual_points, before.residual_points);
                        assert_eq!(work.snapshot().root_iterations, before.root_iterations);
                        assert_eq!(
                            result.status[reference.side.index()].try_to_vec::<i64>()?,
                            [SourceSonicStatus::PositiveRoot as i64]
                        );
                        super::diagnostics::evidence(&result)?;
                        assert_eq!(result.domains.len(), 1);
                        let domain = &result.domains[0];
                        super::diagnostics::check(domain)?;
                        assert_eq!(domain.status.try_to_vec::<i64>()?, [0], "{}", reference.id);
                        for (actual, expected) in [
                            (&domain.query.area, reference.area),
                            (&domain.depth, reference.depth),
                            (&domain.velocity, reference.velocity),
                            (&domain.discharge, reference.discharge),
                        ] {
                            let actual = frozen::endpoints(actual)?[0];
                            let expected = expected.map(f64::from_bits);
                            assert!(
                                actual[0] <= expected[0] && expected[1] <= actual[1],
                                "{}: {actual:?} excludes {expected:?}",
                                reference.id
                            );
                            if f64::from_bits(reference.depth[0]) < 1e-30 {
                                assert!(
                                    (actual[1] - actual[0]) / expected[0].abs() < 1e-10,
                                    "{} precision: {actual:?}",
                                    reference.id
                                );
                            }
                        }
                        // The retained film's selected sonic root is distinct from its wet-star root 2^-164.
                        if reference.id == "retained_vacuum_inside_film" {
                            assert!(
                                frozen::endpoints(&domain.query.area)?[0][0] > 2.0_f64.powi(-164)
                            );
                        }
                        Ok(())
                    },
                )?;
            }
            Ok(())
        }
    }

    mod work {
        use super::{
            PointRootSession, Result, SonicSide, SourceSonicStatus, chart_test_support, frozen,
            hydraulic, support,
        };
        use crate::channel::flow::{
            surveyed::{
                error::{HydraulicError, PrismaticRiemannRejection},
                work::QueryWork,
            },
            surveyed_points::wet_root_types::WetRootStatus,
        };
        use numr::{dtype::DType, runtime::Runtime};
        use solvr::optimize::CertifiedRootTolerance;

        pub(super) fn run<R>(client: &R::Client) -> Result
        where
            R: Runtime<DType = DType>,
            R::Client: support::DepthClient<R>,
        {
            narrow_capture::<R>(client)?;
            independent_rounds::<R>(client)?;
            post_callback_error::<R>(client)?;
            accounting_mismatch::<R>(client)?;
            wet_error_seals_certificates::<R>(client)
        }

        fn narrow_capture<R>(client: &R::Client) -> Result
        where
            R: Runtime<DType = DType>,
            R::Client: support::DepthClient<R>,
        {
            let config = chart_test_support::config();
            support::with_rectangle::<R, _>(
                client,
                1.0,
                4.0,
                &[[1.0, 0.0, 0.0, 0.0]],
                &config,
                |pair, work| {
                    let rows = support::rows::<R>(client, &[0; 100])?;
                    let mut session = hydraulic(PointRootSession::new(pair, &rows, 2))?;
                    let requested = support::rows::<R>(client, &[73])?;
                    hydraulic(session.start_source_sonic(&requested, SonicSide::Left))?;
                    assert_eq!(session.callback_points(), 2);
                    assert_eq!(session.point_limit(), 2);
                    let before = work.snapshot();
                    for _ in 0..2 {
                        hydraulic(session.refine_source_sonic(CertifiedRootTolerance {
                            atol: 0.0,
                            rtol: 0.0,
                        }))?;
                    }
                    assert_eq!(work.snapshot().residual_points, before.residual_points);
                    assert_eq!(session.callback_points(), 2);
                    assert!(
                        session
                            .start_source_sonic(&requested, SonicSide::Left)
                            .is_err()
                    );
                    let result = hydraulic(session.certify_source_sonic())?;
                    let statuses = result.status[0].try_to_vec::<i64>()?;
                    for (owner, &status) in statuses.iter().enumerate() {
                        assert_eq!(
                            status,
                            if owner == 73 {
                                SourceSonicStatus::PositiveRoot as i64
                            } else {
                                SourceSonicStatus::NotRequested as i64
                            }
                        );
                    }
                    assert_eq!(
                        result.domains[0].invocation_owner.try_to_vec::<i64>()?,
                        [73]
                    );
                    assert_eq!(result.domains[0].parent_rows.try_to_vec::<i64>()?, [0]);
                    assert_eq!(result.domains[0].iterations, 0);
                    let another = support::rows::<R>(client, &[74])?;
                    hydraulic(session.start_source_sonic(&another, SonicSide::Left))?;
                    assert_eq!(
                        session.sonic.status[0].try_to_vec::<i64>()?[74],
                        SourceSonicStatus::WorkLimit as i64
                    );
                    assert_eq!(session.callback_points(), 2);
                    Ok(())
                },
            )?;
            support::with_rectangle::<R, _>(
                client,
                1.0,
                4.0,
                &[[1.0, 0.0, 1.0, 0.0]],
                &config,
                |pair, _| {
                    let rows = support::rows::<R>(client, &[0; 100])?;
                    let mut session = hydraulic(PointRootSession::new(pair, &rows, 1))?;
                    hydraulic(session.start_wet(&support::rows::<R>(client, &[2])?))?;
                    assert_eq!(session.callback_points(), 1);
                    assert_eq!(session.point_limit(), 1);
                    assert_eq!(
                        session.locator_status().try_to_vec::<i64>()?[2],
                        WetRootStatus::Located as i64
                    );
                    assert_eq!(
                        session.domains()[0].invocation_owner.try_to_vec::<i64>()?,
                        [2]
                    );
                    Ok(())
                },
            )
        }

        fn independent_rounds<R>(client: &R::Client) -> Result
        where
            R: Runtime<DType = DType>,
            R::Client: support::DepthClient<R>,
        {
            let mut config = chart_test_support::config();
            config.root.max_iterations = 1;
            config.root.atol = 0.0;
            config.root.rtol = 0.0;
            support::with_rectangle::<R, _>(
                client,
                1.0,
                4.0,
                &[[1.0, 0.0, 1.0, 0.0]],
                &config,
                |pair, work| {
                    let rows = support::rows::<R>(client, &[0])?;
                    let mut session = hydraulic(PointRootSession::new(pair, &rows, 14))?;
                    for side in [SonicSide::Left, SonicSide::Right] {
                        hydraulic(session.start_source_sonic(&rows, side))?;
                    }
                    assert_eq!(session.callback_points(), 14);
                    assert_eq!(work.snapshot().root_iterations, 2);
                    assert_eq!(work.snapshot().maximum_root_iterations, 1);
                    let before = work.snapshot();
                    hydraulic(session.refine_source_sonic(CertifiedRootTolerance {
                        atol: 0.0,
                        rtol: 0.0,
                    }))?;
                    assert_eq!(session.callback_points(), 14);
                    assert_eq!(work.snapshot().residual_points, before.residual_points);
                    for request in &session.sonic.requests {
                        assert_eq!(request.domains[0].iterations, 1);
                    }
                    Ok(())
                },
            )
        }

        fn post_callback_error<R>(client: &R::Client) -> Result
        where
            R: Runtime<DType = DType>,
            R::Client: support::DepthClient<R>,
        {
            let mut config = chart_test_support::config();
            config.max_native_query_points = 6;
            support::with_rectangle::<R, _>(
                client,
                1.0,
                4.0,
                &[[1.0, 0.0, 0.0, 0.0]],
                &config,
                |pair, work| {
                    let rows = support::rows::<R>(client, &[0])?;
                    let mut session =
                        hydraulic(PointRootSession::new(pair, &rows, config.root.max_points))?;
                    let error = session
                        .start_source_sonic(&rows, SonicSide::Left)
                        .err()
                        .ok_or("Native work error did not surface")?;
                    assert!(matches!(
                        error,
                        HydraulicError::Rejected {
                            reason: PrismaticRiemannRejection::WorkLimit,
                            ..
                        }
                    ));
                    assert_eq!(work.snapshot().residual_points, 6);
                    assert_eq!(session.callback_points(), 6);
                    assert!(session.sonic.requests[0].terminal);
                    let before = work.snapshot();
                    assert!(session.start_source_sonic(&rows, SonicSide::Left).is_err());
                    assert!(
                        session
                            .refine_source_sonic(CertifiedRootTolerance {
                                atol: 0.0,
                                rtol: 0.0
                            })
                            .is_err()
                    );
                    assert!(session.certify_source_sonic().is_err());
                    assert_eq!(session.callback_points(), 6);
                    assert_eq!(work.snapshot(), before);
                    Ok(())
                },
            )
        }

        fn accounting_mismatch<R>(client: &R::Client) -> Result
        where
            R: Runtime<DType = DType>,
            R::Client: support::DepthClient<R>,
        {
            let config = chart_test_support::config();
            support::with_rectangle::<R, _>(
                client,
                1.0,
                4.0,
                &[[1.0, 0.0, 1.0, 0.0]],
                &config,
                |pair, work| {
                    let rows = support::rows::<R>(client, &[0])?;
                    let mut session = hydraulic(PointRootSession::new(pair, &rows, 10))?;
                    let before = session.checkpoint();
                    hydraulic(work.charge(QueryWork::Residual, 1))?;
                    assert!(session.reconcile(before, true).is_err());
                    assert_eq!(session.callback_points(), 1);
                    assert!(session.terminal);
                    let snapshot = work.snapshot();
                    assert!(session.start_wet(&rows).is_err());
                    assert!(session.start_source_sonic(&rows, SonicSide::Left).is_err());
                    assert!(
                        session
                            .refine_source_sonic(CertifiedRootTolerance {
                                atol: 0.0,
                                rtol: 0.0
                            })
                            .is_err()
                    );
                    assert_eq!(session.callback_points(), 1);
                    assert_eq!(work.snapshot(), snapshot);
                    Ok(())
                },
            )
        }

        fn wet_error_seals_certificates<R>(client: &R::Client) -> Result
        where
            R: Runtime<DType = DType>,
            R::Client: support::DepthClient<R>,
        {
            let mut config = chart_test_support::config();
            config.max_native_query_points = 3;
            support::with_rectangle::<R, _>(
                client,
                1.0,
                3.0,
                &[[1.0, 1.0, 1.0, 1.0]],
                &config,
                |pair, work| {
                    let rows = support::rows::<R>(client, &[0])?;
                    let mut session =
                        hydraulic(PointRootSession::new(pair, &rows, config.root.max_points))?;
                    hydraulic(session.start_source_sonic(&rows, SonicSide::Left))?;
                    assert_eq!(
                        session.sonic.status[0].try_to_vec::<i64>()?,
                        [SourceSonicStatus::SourceContact as i64]
                    );
                    hydraulic(session.start_wet(&rows))?;
                    assert_eq!(session.callback_points(), 3);
                    assert!(
                        session
                            .refine_wet_depth_in_place(CertifiedRootTolerance {
                                atol: 0.0,
                                rtol: 0.0
                            })
                            .is_err()
                    );
                    assert!(session.wet.terminal);
                    let before = work.snapshot();
                    assert!(session.certify_wet_star().is_err());
                    assert!(session.certify_wet_waves().is_err());
                    // Independent source-contact state still certifies without another residual or quadrature point.
                    let sonic = hydraulic(session.certify_source_sonic())?;
                    assert_eq!(sonic.domains.len(), 1);
                    assert_eq!(
                        frozen::endpoints(&sonic.domains[0].discharge)?,
                        [[1.0, 1.0]]
                    );
                    assert_eq!(session.callback_points(), 3);
                    assert_eq!(work.snapshot(), before);
                    Ok(())
                },
            )
        }
    }
}

#[cfg(test)]
mod selected_tests {
    use super::super::{
        chart_test_support, point_root_session::PointRootSession, source_sonic_types::SonicSide,
        source_tests::support as frozen, wet_depth::tests::support,
    };
    use crate::channel::flow::surveyed::error::checked;
    use frozen::{Result, hydraulic};
    use numr::{dtype::DType, runtime::Runtime};
    use solvr::optimize::CertifiedRootTolerance;

    fn snapshot<R>(
        session: &PointRootSession<'_, '_, R>,
        side: SonicSide,
        owner: i64,
    ) -> Result<Vec<Vec<u64>>>
    where
        R: Runtime<DType = DType>,
        R::Client: support::DepthClient<R>,
    {
        for request in &session.sonic.requests {
            if request.binding.side != side {
                continue;
            }
            for domain in &request.domains {
                let owners = domain.invocation_owner.try_to_vec::<i64>()?;
                if let Some(position) = owners.iter().position(|value| *value == owner) {
                    let rows = support::rows::<R>(
                        session.sources.geometry.client(),
                        &[i64::try_from(position)?],
                    )?;
                    let query = hydraulic(checked::<R, _>(
                        session.sources.geometry.client(),
                        "selected sonic cached-query snapshot",
                        |a| domain.query.select(a, &rows),
                    ))?;
                    let mut fields = Vec::new();
                    for tensor in [
                        query.area.lower(),
                        query.area.upper(),
                        query.query.parameter.lower(),
                        query.query.parameter.upper(),
                        query.query.scale.high(),
                        query.query.scale.low(),
                        query.query.scale.abs_error(),
                        query.above_start.lower(),
                        query.above_start.upper(),
                        query.below_end.lower(),
                        query.below_end.upper(),
                        &query.candidate,
                    ] {
                        fields.push(
                            tensor
                                .try_to_vec::<f64>()?
                                .into_iter()
                                .map(f64::to_bits)
                                .collect(),
                        );
                    }
                    for tensor in [
                        domain.depth.lower(),
                        domain.depth.upper(),
                        domain.normalized_depth.lower(),
                        domain.normalized_depth.upper(),
                        domain.numerical.epsilon().high(),
                        domain.numerical.epsilon().low(),
                        domain.numerical.epsilon().abs_error(),
                    ] {
                        fields.push(
                            tensor
                                .narrow(0, position, 1)?
                                .try_to_vec::<f64>()?
                                .into_iter()
                                .map(f64::to_bits)
                                .collect(),
                        );
                    }
                    fields.push(
                        domain
                            .status
                            .narrow(0, position, 1)?
                            .try_to_vec::<i64>()?
                            .into_iter()
                            .map(|value| value as u64)
                            .collect(),
                    );
                    return Ok(fields);
                }
            }
        }
        Err("Selected sonic snapshot owner is absent".into())
    }

    fn matrix<R>(client: &R::Client) -> Result
    where
        R: Runtime<DType = DType>,
        R::Client: support::DepthClient<R>,
    {
        let config = chart_test_support::config();
        let epsilon = 2.0_f64.powi(-50);
        support::with_rectangle::<R, _>(
            client,
            1.,
            4.,
            &[[1., 0., 1., 2. - epsilon], [1., 0., 1., 2. - 2. * epsilon]],
            &config,
            |pair, work| {
                let owners = support::rows::<R>(client, &[0, 1])?;
                let mut session =
                    hydraulic(PointRootSession::new(pair, &owners, config.root.max_points))?;
                hydraulic(session.start_source_sonic(&owners, SonicSide::Right))?;
                hydraulic(session.start_source_sonic(&owners, SonicSide::Left))?;
                assert!(
                    session.sonic.requests[0]
                        .domains
                        .iter()
                        .all(|d| d.refinement.is_none())
                );
                assert!(
                    session.sonic.requests[1]
                        .domains
                        .iter()
                        .all(|d| d.refinement.is_some())
                );
                let untouched_right = snapshot(&session, SonicSide::Right, 0)?;
                let untouched_left = [
                    snapshot(&session, SonicSide::Left, 0)?,
                    snapshot(&session, SonicSide::Left, 1)?,
                ];
                let locating = session
                    .sonic
                    .requests
                    .iter()
                    .map(|r| {
                        (
                            r.endpoint_points,
                            r.bracket_points,
                            r.bracket_expansions,
                            r.tolerance.atol.to_bits(),
                            r.tolerance.rtol.to_bits(),
                        )
                    })
                    .collect::<Vec<_>>();
                let target_owner = support::rows::<R>(client, &[1])?;
                let upper = session.sonic.requests[0].domains[0]
                    .normalized_depth
                    .upper()
                    .narrow(0, 1, 1)?
                    .item::<f64>()?;
                let target = upper * 0.25;
                assert!(target > 0. && target < upper * 0.5);
                let right_depth_before = [
                    session.sonic.requests[0].domains[0]
                        .depth
                        .lower()
                        .narrow(0, 1, 1)?
                        .item::<f64>()?,
                    session.sonic.requests[0].domains[0]
                        .depth
                        .upper()
                        .narrow(0, 1, 1)?
                        .item::<f64>()?,
                ];
                let right_callbacks_before = session.sonic.requests[0]
                    .domains
                    .iter()
                    .map(|domain| domain.callback_points)
                    .sum::<usize>();
                let right_session_before = session.callback_points();
                let right_work_before = work.snapshot().residual_points;
                hydraulic(session.refine_source_sonic_selected(
                    &target_owner,
                    SonicSide::Right,
                    CertifiedRootTolerance {
                        atol: target,
                        rtol: 0.,
                    },
                ))?;
                let right_work = work
                    .snapshot()
                    .residual_points
                    .checked_sub(right_work_before);
                assert!(right_work.is_some_and(|points| points > 0));
                assert_eq!(
                    session.callback_points().checked_sub(right_session_before),
                    right_work
                );
                assert_eq!(
                    session.sonic.requests[0]
                        .domains
                        .iter()
                        .map(|domain| domain.callback_points)
                        .sum::<usize>()
                        .checked_sub(right_callbacks_before),
                    right_work
                );
                assert_eq!(untouched_right, snapshot(&session, SonicSide::Right, 0)?);
                for (owner, before) in untouched_left.iter().enumerate() {
                    assert_eq!(
                        *before,
                        snapshot(&session, SonicSide::Left, i64::try_from(owner)?)?
                    );
                }
                assert_eq!(session.sonic.requests[0].domains.len(), 2);
                let mut preserved_deferred = false;
                let mut refined_right = false;
                for domain in &session.sonic.requests[0].domains {
                    if domain.invocation_owner.try_to_vec::<i64>()? == [1] {
                        assert!(domain.refinement.is_some());
                        assert!(domain.callback_points > 0 && domain.mapped_points > 0);
                        let depth = frozen::endpoints(&domain.depth)?[0];
                        let normalized = frozen::endpoints(&domain.normalized_depth)?[0];
                        assert!(depth[0] > 0. && normalized[0] > 0.);
                        assert!(frozen::endpoints(&domain.query.area)?[0][0] > 0.);
                        assert!(
                            depth[1] - depth[0] < right_depth_before[1] - right_depth_before[0]
                                || normalized[1] - normalized[0] <= target,
                            "Demanded Right owner retains its previous depth interval: {depth:?}, normalized={normalized:?}, target={target:e}"
                        );
                        refined_right = true;
                    }
                    if domain.refinement.is_none()
                        && domain.invocation_owner.try_to_vec::<i64>()? == [0]
                    {
                        assert_eq!(
                            [
                                domain.iterations,
                                domain.callback_points,
                                domain.mapped_points
                            ],
                            [0; 3]
                        );
                        preserved_deferred = true;
                    }
                }
                assert!(preserved_deferred);
                assert!(refined_right);
                let right_after = [
                    snapshot(&session, SonicSide::Right, 0)?,
                    snapshot(&session, SonicSide::Right, 1)?,
                ];
                let left_zero = snapshot(&session, SonicSide::Left, 0)?;
                let domain = &session.sonic.requests[1].domains[0];
                let lower = domain
                    .normalized_depth
                    .lower()
                    .narrow(0, 1, 1)?
                    .item::<f64>()?;
                let upper = domain
                    .normalized_depth
                    .upper()
                    .narrow(0, 1, 1)?
                    .item::<f64>()?;
                let target = (upper - lower) * 0.25;
                assert!(target > 0.);
                let left_depth_before = [
                    domain.depth.lower().narrow(0, 1, 1)?.item::<f64>()?,
                    domain.depth.upper().narrow(0, 1, 1)?.item::<f64>()?,
                ];
                let left_callbacks_before = domain.callback_points;
                let left_mapped_before = domain.mapped_points;
                let left_session_before = session.callback_points();
                let left_work_before = work.snapshot().residual_points;
                hydraulic(session.refine_source_sonic_selected(
                    &target_owner,
                    SonicSide::Left,
                    CertifiedRootTolerance {
                        atol: target,
                        rtol: 0.,
                    },
                ))?;
                let left_work = work
                    .snapshot()
                    .residual_points
                    .checked_sub(left_work_before);
                assert!(left_work.is_some_and(|points| points > 0));
                assert_eq!(
                    session.callback_points().checked_sub(left_session_before),
                    left_work
                );
                let mut refined_left = false;
                for domain in &session.sonic.requests[1].domains {
                    let owners = domain.invocation_owner.try_to_vec::<i64>()?;
                    if let Some(position) = owners.iter().position(|owner| *owner == 1) {
                        assert!(domain.refinement.is_some());
                        assert!(domain.mapped_points > left_mapped_before);
                        assert_eq!(
                            domain.callback_points.checked_sub(left_callbacks_before),
                            left_work
                        );
                        let depth = frozen::endpoints(&domain.depth)?[position];
                        let normalized = frozen::endpoints(&domain.normalized_depth)?[position];
                        assert!(depth[0] > 0. && normalized[0] > 0.);
                        assert!(frozen::endpoints(&domain.query.area)?[position][0] > 0.);
                        assert!(
                            depth[1] - depth[0] < left_depth_before[1] - left_depth_before[0]
                                || normalized[1] - normalized[0] <= target,
                            "Demanded Left owner retains its previous depth interval: {depth:?}, normalized={normalized:?}, target={target:e}"
                        );
                        refined_left = true;
                    }
                }
                assert!(refined_left);
                assert_eq!(left_zero, snapshot(&session, SonicSide::Left, 0)?);
                for (owner, before) in right_after.iter().enumerate() {
                    assert_eq!(
                        *before,
                        snapshot(&session, SonicSide::Right, i64::try_from(owner)?)?
                    );
                }
                assert_eq!(
                    locating,
                    session
                        .sonic
                        .requests
                        .iter()
                        .map(|r| (
                            r.endpoint_points,
                            r.bracket_points,
                            r.bracket_expansions,
                            r.tolerance.atol.to_bits(),
                            r.tolerance.rtol.to_bits()
                        ))
                        .collect::<Vec<_>>()
                );
                assert_eq!(session.callback_points(), work.snapshot().residual_points);
                assert_eq!(session.point_limit(), config.root.max_points);
                Ok(())
            },
        )
    }

    #[test]
    fn configured_cpu_selected_source_sonic_preserves_unrequested_states() -> Result {
        use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};
        let original = CpuClient::new(CpuDevice::new());
        let before = original.parallelism();
        let config = ParallelismConfig::new(Some(2), Some(257));
        let client = original.with_parallelism(config);
        matrix::<CpuRuntime>(&client)?;
        assert_eq!(client.parallelism(), config);
        assert_eq!(original.parallelism(), before);
        Ok(())
    }
    #[cfg(feature = "cuda")]
    #[test]
    fn actual_cuda_selected_source_sonic_preserves_unrequested_states() -> Result {
        use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
        matrix::<CudaRuntime>(&CudaClient::new(CudaDevice::new(0))?)
    }
}
