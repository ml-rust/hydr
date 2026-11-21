//! Batch layouts: every lane returns the bits of its single-row query.

use super::lanes::{Lane, WIDTHS, layout, layout_lanes, outputs};
use super::runtime::{Runtime, client};
use super::support::TestResult;

#[test]
fn culvert_outlet_rows_return_single_row_bits_at_every_layout_width() -> TestResult {
    let client = client();
    let lanes = layout_lanes()?;
    let mut singles = Vec::with_capacity(lanes.len());
    for lane in &lanes {
        let [single] =
            <[String; 1]>::try_from(outputs::<Runtime>(&client, std::slice::from_ref(lane))?)
                .map_err(|rows| format!("single query returned {} rows", rows.len()))?;
        singles.push(single);
    }
    for width in WIDTHS {
        let picked = layout(width, lanes.len());
        let batch: Vec<Lane> = picked
            .iter()
            .filter_map(|&i| lanes.get(i).copied())
            .collect();
        let rows = outputs::<Runtime>(&client, &batch)?;
        if rows.len() != width {
            return Err(format!("width {width} returned {} rows", rows.len()).into());
        }
        for (position, (row, &index)) in rows.iter().zip(&picked).enumerate() {
            let single = singles.get(index).ok_or("no single-row result")?;
            if row != single {
                return Err(format!(
                    "width {width} lane {position}: {row} differs from the single-row {single}"
                )
                .into());
            }
        }
    }
    Ok(())
}
