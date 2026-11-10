use super::CrossSectionError;
use std::collections::BTreeSet;

/// One surveyed point, ordered by nondecreasing horizontal station.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CrossSectionPoint {
    /// Horizontal station, in metres.
    pub station_m: f64,
    /// Bed or bank elevation in the declared vertical datum, in metres.
    pub elevation_m: f64,
}

/// Explicit open profile and its supported water-stage range.
#[derive(Clone, Debug, PartialEq)]
pub struct CrossSectionDefinition {
    /// Unique stable section identity.
    pub id: String,
    /// Vertical reference shared by this profile and its query stages.
    pub vertical_datum: String,
    /// Ordered piecewise-linear bed and bank points, including vertical walls.
    pub points: Vec<CrossSectionPoint>,
    /// Highest supported stage without extrapolated banks, in metres.
    pub maximum_stage_m: f64,
    /// Survey or analytical-profile evidence.
    pub profile_provenance: String,
}

pub(super) struct GeometryInputs {
    pub bed: Vec<f64>,
    pub owners: Vec<i64>,
    pub segment_sections: Vec<usize>,
    pub left_station: Vec<f64>,
    pub right_station: Vec<f64>,
    pub left_elevation: Vec<f64>,
    pub right_elevation: Vec<f64>,
}

pub(super) fn check(
    definitions: &[CrossSectionDefinition],
) -> Result<GeometryInputs, CrossSectionError> {
    if definitions.is_empty() {
        return Err(CrossSectionError::input(
            "section count=0: supply at least one explicit surveyed profile",
        ));
    }
    let mut ids = BTreeSet::new();
    let mut result = GeometryInputs {
        bed: Vec::with_capacity(definitions.len()),
        owners: vec![],
        segment_sections: vec![],
        left_station: vec![],
        right_station: vec![],
        left_elevation: vec![],
        right_elevation: vec![],
    };
    for (index, definition) in definitions.iter().enumerate() {
        for (field, value) in [
            ("id", &definition.id),
            ("vertical_datum", &definition.vertical_datum),
            ("profile_provenance", &definition.profile_provenance),
        ] {
            if value.trim().is_empty() || value.chars().any(char::is_control) {
                return Err(CrossSectionError::input(format!(
                    "{field}={value:?}: supply explicit nonempty profile identity and evidence"
                ))
                .at(index));
            }
        }
        if !ids.insert(&definition.id) {
            return Err(CrossSectionError::input(format!(
                "id={:?}: supply a unique section identity",
                definition.id
            ))
            .at(index));
        }
        if definition.points.len() < 3
            || definition
                .points
                .iter()
                .any(|p| !p.station_m.is_finite() || !p.elevation_m.is_finite())
        {
            return Err(CrossSectionError::input(format!(
                "id={:?}, points={}: supply at least three finite surveyed points",
                definition.id,
                definition.points.len()
            ))
            .at(index));
        }
        let first = definition.points[0];
        let last = definition.points[definition.points.len() - 1];
        let bed = definition
            .points
            .iter()
            .map(|p| p.elevation_m)
            .fold(f64::INFINITY, f64::min);
        let highest = definition
            .points
            .iter()
            .map(|p| p.elevation_m)
            .fold(f64::NEG_INFINITY, f64::max);
        let width = last.station_m - first.station_m;
        let span = highest - bed;
        if !width.is_finite()
            || width <= 0.0
            || !span.is_finite()
            || !definition.maximum_stage_m.is_finite()
            || definition.maximum_stage_m <= bed
            || definition.maximum_stage_m > first.elevation_m.min(last.elevation_m)
        {
            return Err(CrossSectionError::input(format!("id={:?}, extent={width}, elevation_span={span}, bed={bed}, maximum_stage={}: supply finite positive extent and a bank-supported cap above the bed", definition.id, definition.maximum_stage_m)).at(index));
        }
        let owner = i64::try_from(index).map_err(|_| {
            CrossSectionError::input("section count exceeds I64 indexing").at(index)
        })?;
        let mut vertical_direction = 0i8;
        for pair in definition.points.windows(2) {
            let a = pair[0];
            let b = pair[1];
            if b.station_m < a.station_m
                || (a.station_m == b.station_m && a.elevation_m == b.elevation_m)
            {
                return Err(CrossSectionError::input(format!("id={:?}, points={a:?}/{b:?}: supply nondecreasing stations without duplicate points", definition.id)).at(index));
            }
            if a.station_m == b.station_m {
                let direction = if b.elevation_m > a.elevation_m { 1 } else { -1 };
                if vertical_direction != 0 && vertical_direction != direction {
                    return Err(CrossSectionError::input(format!(
                        "id={:?}, station={}: remove retracing within the vertical profile run",
                        definition.id, a.station_m
                    ))
                    .at(index));
                }
                vertical_direction = direction;
            } else {
                vertical_direction = 0;
            }
            result.owners.push(owner);
            result.segment_sections.push(index);
            result.left_station.push(a.station_m);
            result.right_station.push(b.station_m);
            result.left_elevation.push(a.elevation_m);
            result.right_elevation.push(b.elevation_m);
        }
        result.bed.push(bed);
    }
    i64::try_from(result.owners.len())
        .map_err(|_| CrossSectionError::input("segment count exceeds I64 indexing"))?;
    Ok(result)
}
