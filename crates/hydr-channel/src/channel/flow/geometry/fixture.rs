use super::{ChannelGeometry, TriangularChannelCell};

pub(in crate::channel::flow) fn geometry(left: f64, right: f64, bed: f64) -> ChannelGeometry {
    ChannelGeometry::PrismaticTriangular {
        left_bank_slope_h_per_v: left,
        right_bank_slope_h_per_v: right,
        bed_elevation_m: bed,
        cells: vec![TriangularChannelCell {
            id: "cell".into(),
            length_m: 2.0,
            maximum_depth_m: 3.0,
            manning_n: 0.03,
        }],
    }
}
