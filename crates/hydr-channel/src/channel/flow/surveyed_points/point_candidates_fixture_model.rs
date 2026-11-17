use crate::channel::flow::surveyed::PrismaticRiemannConfig;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::channel::flow::surveyed_points) struct Config {
    pub(in crate::channel::flow::surveyed_points) floats: [u64; 8],
    pub(in crate::channel::flow::surveyed_points) counts: [usize; 7],
}
impl Config {
    pub(in crate::channel::flow::surveyed_points) fn build(&self) -> PrismaticRiemannConfig {
        let f = self.floats.map(f64::from_bits);
        let n = self.counts;
        PrismaticRiemannConfig {
            gravity_m_per_s2: f[0],
            recovery: crate::channel::section::PrismaticRecoveryConfig {
                depth_atol_m: f[1],
                depth_rtol: f[2],
            },
            water_flux_atol_m3_per_s: f[3],
            momentum_flux_atol_m4_per_s2: f[4],
            flux_rtol: f[5],
            root: solvr::optimize::CertifiedBisectionOptions {
                atol: f[6],
                rtol: f[7],
                max_iterations: n[0],
                max_points: n[1],
            },
            max_bracket_expansions: n[2],
            initial_simpson_panels: n[3],
            max_simpson_panels: n[4],
            max_refinements: n[5],
            max_native_query_points: n[6],
        }
    }
}
#[derive(Clone, Copy)]
pub(in crate::channel::flow::surveyed_points) struct State {
    pub(in crate::channel::flow::surveyed_points) label: &'static str,
    pub(in crate::channel::flow::surveyed_points) area: [u64; 2],
    pub(in crate::channel::flow::surveyed_points) depth: Option<[u64; 2]>,
    pub(in crate::channel::flow::surveyed_points) velocity: Option<[u64; 2]>,
    pub(in crate::channel::flow::surveyed_points) discharge: [u64; 2],
    pub(in crate::channel::flow::surveyed_points) velocity_provenance: &'static str,
}
#[derive(Clone, Copy)]
pub(in crate::channel::flow::surveyed_points) struct Edge {
    pub(in crate::channel::flow::surveyed_points) label: &'static str,
    pub(in crate::channel::flow::surveyed_points) bounds: [u64; 2],
}
#[derive(Clone, Copy)]
pub(in crate::channel::flow::surveyed_points) struct Execution {
    pub(in crate::channel::flow::surveyed_points) config_identity: &'static str,
    pub(in crate::channel::flow::surveyed_points) expected_identity: &'static str,
    pub(in crate::channel::flow::surveyed_points) id: &'static str,
    pub(in crate::channel::flow::surveyed_points) config: usize,
    pub(in crate::channel::flow::surveyed_points) source_group: &'static str,
    pub(in crate::channel::flow::surveyed_points) group: &'static str,
    pub(in crate::channel::flow::surveyed_points) declared: &'static str,
    pub(in crate::channel::flow::surveyed_points) outcome: &'static str,
    pub(in crate::channel::flow::surveyed_points) physical: &'static str,
    pub(in crate::channel::flow::surveyed_points) traces: &'static [State],
    pub(in crate::channel::flow::surveyed_points) star: Option<State>,
    pub(in crate::channel::flow::surveyed_points) edges: &'static [Edge],
}
