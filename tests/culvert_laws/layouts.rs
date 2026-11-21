//! Batch layouts: every culvert lane among weirs returns the bits of its single-lane proposal.

use super::lanes::{Lane, WIDTHS, layout, layout_lanes, outputs};
use super::support::{TestResult, client};
use numr::runtime::cpu::CpuRuntime;

#[test]
fn culvert_lanes_return_single_lane_bits_at_every_layout_width() -> TestResult {
    let client = client();
    let lanes = layout_lanes()?;
    let mut singles = Vec::with_capacity(lanes.len());
    for lane in &lanes {
        let [single] =
            <[String; 1]>::try_from(outputs::<CpuRuntime>(&client, std::slice::from_ref(lane))?)
                .map_err(|rows| format!("single proposal returned {} lanes", rows.len()))?;
        singles.push(single);
    }
    for width in WIDTHS {
        let picked = layout(width, lanes.len());
        let batch: Vec<Lane> = picked
            .iter()
            .filter_map(|&i| lanes.get(i).cloned())
            .collect();
        let rows = outputs::<CpuRuntime>(&client, &batch)?;
        if rows.len() != width {
            return Err(format!("width {width} returned {} lanes", rows.len()).into());
        }
        for (position, (row, &index)) in rows.iter().zip(&picked).enumerate() {
            let single = singles.get(index).ok_or("no single-lane result")?;
            if row != single {
                return Err(format!(
                    "width {width} lane {position}: {row} differs from the single-lane {single}"
                )
                .into());
            }
        }
    }
    Ok(())
}
