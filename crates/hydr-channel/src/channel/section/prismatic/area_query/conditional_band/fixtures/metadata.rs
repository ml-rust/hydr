use super::loader::{FixtureError, Row};
use super::model::{
    ClipAuthority, FixtureSet, GapAuthority, GapRef, Panel, PanelKind, Path, PhysicalEndpoint,
};

fn clip(row: &Row<'_>, index: usize) -> Result<ClipAuthority, FixtureError> {
    match row.text(index)? {
        "S" => Ok(ClipAuthority::BandStart),
        "E" => Ok(ClipAuthority::BandEnd),
        "I" => Ok(ClipAuthority::Interior),
        _ => Err(row.error("Invalid clip identity")),
    }
}

pub(super) fn parse(text: &str) -> Result<FixtureSet, FixtureError> {
    let mut paths: Vec<Path> = Vec::new();
    let mut rows = 0;
    let mut padding = 0;
    for (line, text) in text.lines().enumerate().skip(1) {
        let row = Row::new(text, "panels.tsv", line + 1, 20)?;
        let index = row.usize(0)?;
        row.require(index < 44, "Path index exceeds frozen count")?;
        let physical_endpoint = match row.text(7)? {
            "L" => PhysicalEndpoint::Lower,
            "U" => PhysicalEndpoint::Upper,
            _ => return Err(row.error("Invalid physical endpoint")),
        };
        if index == paths.len() {
            paths.push(Path {
                id: row.text(1)?.to_owned(),
                binding_id: row.text(2)?.to_owned(),
                binding_index: row.usize(3)?,
                query_index: row.usize(4)?,
                original_parameter_endpoint: row.usize(5)?,
                physical_endpoint,
                panels: Vec::new(),
                padded_bands: Vec::new(),
            });
        }
        row.require(index + 1 == paths.len(), "Nonsequential path index")?;
        let path = paths
            .get_mut(index)
            .ok_or_else(|| row.error("Missing path"))?;
        row.require(
            path.id == row.text(1)?
                && path.binding_id == row.text(2)?
                && path.binding_index == row.usize(3)?
                && path.query_index == row.usize(4)?
                && path.original_parameter_endpoint == row.usize(5)?
                && path.physical_endpoint == physical_endpoint,
            "Inconsistent path identity",
        )?;
        row.require(row.text(6)? == "1", "Missing exact telescoping identity")?;
        row.require(
            path.binding_index == index / 2
                && path.query_index < 17
                && path.original_parameter_endpoint < 2,
            "Invalid original input mapping",
        )?;
        let band_index = row.usize(8)?;
        row.require(
            band_index == path.panels.len() + path.padded_bands.len() && band_index < 3,
            "Nonsequential band index",
        )?;
        let kind = match row.text(9)? {
            "O" => PanelKind::OrdinaryWet,
            "D" => PanelKind::DryFirstBand,
            "Z" => PanelKind::Zero,
            "P" => {
                for column in 10..20 {
                    row.require(row.text(column)? == "-", "Padded physical data is present")?;
                }
                path.padded_bands.push(band_index);
                padding += 1;
                rows += 1;
                continue;
            }
            _ => return Err(row.error("Invalid panel kind")),
        };
        row.require(
            path.padded_bands.is_empty(),
            "Physical panel follows padding",
        )?;
        let authority = match row.text(13)? {
            "C" => GapAuthority::ConditionedLocal,
            "G" => GapAuthority::OriginalGeometry,
            _ => return Err(row.error("Invalid gap authority")),
        };
        let gap = GapRef {
            bounds: [row.bits(10)?, row.bits(11)?],
            native_allowance: row.bits(12)?,
            exact_zero: row.boolean(19)?,
            authority,
        };
        row.require(
            gap.exact_zero == (kind == PanelKind::Zero),
            "Inconsistent zero gap",
        )?;
        path.panels.push(Panel {
            band_index,
            kind,
            gap,
            source_clip: clip(&row, 14)?,
            target_clip: clip(&row, 15)?,
            shared_boundary_identity: row.boolean(16)?,
            original_source_equals_target: row.boolean(17)?,
            ordinary_active: row.boolean(18)?,
            points: Vec::new(),
            intervals: Vec::new(),
            endpoint_geometry: Vec::new(),
        });
        rows += 1;
    }
    let error = || FixtureError {
        table: "panels.tsv",
        line: 0,
        detail: "Frozen panel counts differ",
    };
    if rows != 132
        || padding != 46
        || paths.len() != 44
        || paths
            .iter()
            .any(|p| p.panels.len() + p.padded_bands.len() != 3)
    {
        return Err(error());
    }
    let counts = paths
        .iter()
        .flat_map(|p| &p.panels)
        .fold([0; 3], |mut counts, p| {
            counts[match p.kind {
                PanelKind::OrdinaryWet => 0,
                PanelKind::DryFirstBand => 1,
                PanelKind::Zero => 2,
            }] += 1;
            counts
        });
    if counts != [35, 13, 38] {
        return Err(error());
    }
    Ok(FixtureSet { paths })
}
