use hydr::grid::CartesianGrid;
use hydr::model::{CellState, Domain, FlowState, SpatialReference};
use hydr::soil::{LandCell, LandModel, SoilState};
use hydr::solver::{
    BoundaryConditions, ForcingInterval, ForcingSchedule, InitialState, Solver, SolverConfig,
    SolverSetup,
};

pub type Result<T = ()> = std::result::Result<T, Box<dyn std::error::Error>>;

pub fn domain(nx: usize, ny: usize, active: Vec<bool>) -> Result<Domain> {
    Ok(Domain::new(
        CartesianGrid::new(nx, ny, 1.0, 1.0)?,
        vec![0.0; nx * ny],
        vec![0.0; nx * ny],
        active,
        SpatialReference::new(
            "synthetic metric plane".into(),
            "synthetic land-source datum".into(),
            [0.0; 2],
        )?,
    )?)
}

pub struct SourceCase {
    pub domain: Domain,
    pub land: LandModel,
    pub flow: FlowState,
    pub soil: SoilState,
    pub rain: Vec<f64>,
    pub evaporation: Vec<f64>,
    pub withdrawal: Vec<f64>,
    pub config: SolverConfig,
    pub end: f64,
}

impl SourceCase {
    pub fn new(domain: Domain, land: LandModel, flow: FlowState) -> Result<Self> {
        let count = domain.grid().cell_count();
        let soil = SoilState::initialized(&land)?;
        Ok(Self {
            domain,
            land,
            flow,
            soil,
            rain: vec![0.0; count],
            evaporation: vec![0.0; count],
            withdrawal: vec![0.0; count],
            config: SolverConfig::default(),
            end: 2.0,
        })
    }

    pub fn uniform(nx: usize, ny: usize, land: LandCell, cell: CellState) -> Result<Self> {
        let domain = domain(nx, ny, vec![true; nx * ny])?;
        let land = LandModel::new(&domain, vec![land; nx * ny])?;
        let flow = FlowState::new(&domain, vec![cell; nx * ny])?;
        Self::new(domain, land, flow)
    }

    pub fn build(self) -> Result<Solver<crate::runtime::Runtime>> {
        let boundaries = BoundaryConditions::impermeable_walls();
        let forcing = ForcingSchedule::new(
            &self.domain,
            &boundaries,
            vec![ForcingInterval {
                start: 0.0,
                end: self.end,
                rain: self.rain,
                potential_evaporation: self.evaporation,
                withdrawal: self.withdrawal,
                boundaries: vec![],
            }],
        )?;
        Ok(Solver::<crate::runtime::Runtime>::new(
            crate::runtime::client(),
            SolverSetup {
                domain: self.domain,
                land: self.land,
                boundaries,
                forcing,
                config: self.config,
            },
            InitialState {
                time: 0.0,
                flow: self.flow,
                soil: self.soil,
            },
        )?)
    }
}
