use crate::channel::flow::surveyed::PrismaticRiemannConfig;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Config {
    pub(super) floats: [u64; 8],
    pub(super) counts: [usize; 7],
}
impl Config {
    pub(super) fn build(&self) -> PrismaticRiemannConfig {
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
pub(super) struct State {
    pub(super) label: &'static str,
    pub(super) area: [u64; 2],
    pub(super) depth: Option<[u64; 2]>,
    pub(super) velocity: Option<[u64; 2]>,
    pub(super) discharge: [u64; 2],
    pub(super) velocity_provenance: &'static str,
}
#[derive(Clone, Copy)]
pub(super) struct Edge {
    pub(super) label: &'static str,
    pub(super) bounds: [u64; 2],
}
#[derive(Clone, Copy)]
pub(super) struct Execution {
    pub(super) config_identity: &'static str,
    pub(super) expected_identity: &'static str,
    pub(super) id: &'static str,
    pub(super) config: usize,
    pub(super) source_group: &'static str,
    pub(super) group: &'static str,
    pub(super) declared: &'static str,
    pub(super) outcome: &'static str,
    pub(super) physical: &'static str,
    pub(super) traces: &'static [State],
    pub(super) star: Option<State>,
    pub(super) edges: &'static [Edge],
}
