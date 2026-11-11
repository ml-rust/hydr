use super::boundary_common::Shape;
use hydr::channel::ChannelBoundary;

#[derive(Clone, Copy)]
pub struct Fixture {
    pub name: &'static str,
    pub shape: Shape,
    pub elevation: f64,
    pub loss: f64,
    pub depth: f64,
    pub velocity: f64,
    pub discharge: f64,
    pub momentum: f64,
    pub dissipation: f64,
    pub root_floor_bits: u64,
    pub root_ceil_bits: u64,
}

impl Fixture {
    pub fn boundary(self) -> ChannelBoundary {
        reservoir(self.elevation, self.loss)
    }
}

pub fn reservoir(elevation_m: f64, entrance_loss_coefficient: f64) -> ChannelBoundary {
    ChannelBoundary::QuiescentReservoir {
        elevation_m,
        entrance_loss_coefficient,
    }
}

// Frozen independent 150-digit exact-binary64-input oracle values.
#[allow(clippy::excessive_precision)]
pub const FIXTURES: [Fixture; 6] = [
    Fixture {
        name: "rectangular-outflow",
        shape: Shape::Rectangle {
            gravity: 1.0,
            width: 3.0,
        },
        elevation: 0.81,
        loss: 0.0,
        depth: 0.8100000000000000532907051820075139403343,
        velocity: 0.1999999999999999407881053533249854846419,
        discharge: 0.4859999999999998880895191177842136255497,
        momentum: 1.0813500000000000783373366175510554319388,
        dissipation: 0.0097199999999999920063942226988750388742,
        root_floor_bits: 4605471050941616620,
        root_ceil_bits: 4605471050941616620,
    },
    Fixture {
        name: "triangular-outflow",
        shape: Shape::Triangle {
            gravity: 2.0,
            coefficient: 1.0,
        },
        elevation: 0.81,
        loss: 0.0,
        depth: 0.8100000000000000532907051820075139403343,
        velocity: 0.3999999999999998815762107066499709692839,
        discharge: 0.2624399999999999568345288025739058986062,
        momentum: 0.4592700000000000215827355987130502455835,
        dissipation: 0.0209951999999999841151065993472037093422,
        root_floor_bits: 4605471050941616620,
        root_ceil_bits: 4605471050941616620,
    },
    Fixture {
        name: "rectangular-inflow-ideal",
        shape: Shape::Rectangle {
            gravity: 1.0,
            width: 1.0,
        },
        elevation: 2.375,
        loss: 0.0,
        depth: 2.0,
        velocity: -0.866025403784438646763723170752936183471,
        discharge: -1.732050807568877293527446341505872366942,
        momentum: 3.5,
        dissipation: 0.0,
        root_floor_bits: 4611686018427387904,
        root_ceil_bits: 4611686018427387904,
    },
    Fixture {
        name: "rectangular-inflow-loss",
        shape: Shape::Rectangle {
            gravity: 1.0,
            width: 1.0,
        },
        elevation: 2.75,
        loss: 1.0,
        depth: 2.0,
        velocity: -0.866025403784438646763723170752936183471,
        discharge: -1.732050807568877293527446341505872366942,
        momentum: 3.5,
        dissipation: 0.6495190528383289850727923780647021376035,
        root_floor_bits: 4611686018427387904,
        root_ceil_bits: 4611686018427387904,
    },
    Fixture {
        name: "triangular-inflow-ideal",
        shape: Shape::Triangle {
            gravity: 8.0,
            coefficient: 1.0,
        },
        elevation: 1.7199074074074074,
        loss: 0.0,
        depth: 1.5000000000000000176898606413302949410967,
        velocity: -1.875771446237125862974798689101935244077,
        discharge: -4.220485754033533291239703487243369413402,
        momentum: 16.916666666666667719213374825819242991245,
        dissipation: 0.0,
        root_floor_bits: 4609434218613702656,
        root_ceil_bits: 4609434218613702657,
    },
    Fixture {
        name: "triangular-inflow-loss",
        shape: Shape::Triangle {
            gravity: 8.0,
            coefficient: 1.0,
        },
        elevation: 1.9398148148148149,
        loss: 1.0,
        depth: 1.5000000000000000241955528635675428537458,
        velocity: -1.875771446237125886824545291288348995794,
        discharge: -4.220485754033533381511208467601553860924,
        momentum: 16.916666666666668106302062048935517591642,
        dissipation: 7.4249286413552909104260309165855648333167,
        root_floor_bits: 4609434218613702656,
        root_ceil_bits: 4609434218613702657,
    },
];
