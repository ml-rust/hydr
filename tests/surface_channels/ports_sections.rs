//! Cell sections of rectangular and surveyed-rectangle port reaches.

use hydr::channel::ChannelGeometry;

/// Width and bed of the first cell of a rectangular or surveyed-rectangle reach.
pub fn first_cell_section(geometry: &ChannelGeometry) -> Option<(f64, f64)> {
    match geometry {
        ChannelGeometry::Rectangular { cells } => cells
            .first()
            .map(|cell| (cell.width_m, cell.bed_elevation_m)),
        ChannelGeometry::PrismaticSurveyed { profile, .. } => {
            let stations = profile.points.iter().map(|p| p.station_m);
            let bed = profile.points.iter().map(|p| p.elevation_m);
            Some((
                stations.clone().fold(f64::MIN, f64::max) - stations.fold(f64::MAX, f64::min),
                bed.fold(f64::MAX, f64::min),
            ))
        }
        _ => None,
    }
}
