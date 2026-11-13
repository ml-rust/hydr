use super::loader::{FixtureError, Row};
use super::model::{FixtureSet, PanelKind, RangePolicy, RangeRef, Sample};

pub(super) fn parse(text: &str, set: &mut FixtureSet) -> Result<(), FixtureError> {
    let mut rows = text.lines().enumerate().skip(1);
    let mut count = 0;
    while let Some((line, text)) = rows.next() {
        let first = Row::new(text, "samples.tsv", line + 1, 13)?;
        let key = (
            first.usize(0)?,
            first.usize(1)?,
            first.text(2)?,
            first.usize(3)?,
        );
        let mut fields = Vec::with_capacity(10);
        for field_index in 0..10 {
            let row = if field_index == 0 {
                Row::new(text, "samples.tsv", line + 1, 13)?
            } else {
                let (line, text) = rows
                    .next()
                    .ok_or_else(|| first.error("Incomplete sample"))?;
                Row::new(text, "samples.tsv", line + 1, 13)?
            };
            row.require(
                (row.usize(0)?, row.usize(1)?, row.text(2)?, row.usize(3)?) == key
                    && row.usize(4)? == field_index,
                "Inconsistent sample or field identity",
            )?;
            let policy = match row.text(11)? {
                "Z" => RangePolicy::ExactZero,
                "C" => RangePolicy::ConditionedLocal,
                "G" => RangePolicy::OriginalGeometry,
                _ => return Err(row.error("Invalid range policy")),
            };
            fields.push(RangeRef {
                minimum: [row.bits(5)?, row.bits(6)?],
                maximum: [row.bits(7)?, row.bits(8)?],
                native_allowances: [row.bits(9)?, row.bits(10)?],
                policy,
                positive: row.boolean(12)?,
            });
            count += 1;
        }
        let fields = fields
            .try_into()
            .map_err(|_| first.error("Wrong field count"))?;
        let panel = set
            .paths
            .get_mut(key.0)
            .and_then(|p| p.panels.get_mut(key.1))
            .ok_or_else(|| first.error("Sample names an absent physical panel"))?;
        first.require(panel.band_index == key.1, "Sample band identity differs")?;
        let samples = match key.2 {
            "P" if panel.kind == PanelKind::OrdinaryWet => &mut panel.points,
            "I" if panel.kind == PanelKind::OrdinaryWet => &mut panel.intervals,
            "D" if panel.kind == PanelKind::DryFirstBand => &mut panel.endpoint_geometry,
            _ => return Err(first.error("Sample kind differs from physical panel")),
        };
        first.require(key.3 == samples.len(), "Nonsequential sample index")?;
        samples.push(Sample { fields });
    }
    let complete = set.paths.iter().flat_map(|p| &p.panels).all(|p| {
        let lengths = (p.points.len(), p.intervals.len(), p.endpoint_geometry.len());
        lengths
            == match p.kind {
                PanelKind::OrdinaryWet => (5, 6, 0),
                PanelKind::DryFirstBand => (0, 0, 2),
                PanelKind::Zero => (0, 0, 0),
            }
    });
    if count != 4110 || !complete {
        return Err(FixtureError {
            table: "samples.tsv",
            line: 0,
            detail: "Frozen sample counts differ",
        });
    }
    Ok(())
}
