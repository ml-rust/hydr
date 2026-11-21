//! Batch layouts: every lane returns the bits of its single-row query.

use super::lanes::{Lane, Outcome, QUERIES, Query, WIDTHS, full_outputs, lanes, layout, outputs};
use super::runtime::{Runtime, client};
use super::support::TestResult;

/// Requires every lane of every width to equal its single-row text.
///
/// `run` evaluates one batch of lanes. `what` names the query in errors.
fn same_at_every_width(
    what: &str,
    lanes: &[Lane],
    run: impl Fn(&[Lane]) -> Outcome<Vec<String>>,
) -> TestResult {
    let mut singles = Vec::with_capacity(lanes.len());
    for lane in lanes {
        let [single] = <[String; 1]>::try_from(run(std::slice::from_ref(lane))?)
            .map_err(|rows| format!("{what} single query returned {} rows", rows.len()))?;
        singles.push(single);
    }
    for width in WIDTHS {
        let picked = layout(width, lanes.len());
        let batch: Vec<Lane> = picked.iter().map(|&i| lanes[i]).collect();
        let rows = run(&batch)?;
        if rows.len() != width {
            return Err(format!("{what} width {width} returned {} rows", rows.len()).into());
        }
        for (position, (row, &index)) in rows.iter().zip(&picked).enumerate() {
            let single = &singles[index];
            if row != single {
                return Err(format!(
                    "{what} width {width} lane {position}: {row} differs from the single-row {single}"
                )
                .into());
            }
        }
    }
    Ok(())
}

#[test]
fn culvert_barrel_queries_return_single_row_bits_at_every_layout_width() -> TestResult {
    let client = client();
    for query in QUERIES {
        same_at_every_width(&format!("{query:?}"), &lanes(query)?, |batch| {
            outputs::<Runtime>(&client, query, batch)
        })?;
    }
    same_at_every_width("full geometry", &lanes(Query::Geometry)?, |batch| {
        full_outputs::<Runtime>(&client, batch)
    })
}
