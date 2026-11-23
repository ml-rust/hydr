//! Profiles the certified culvert outlet march and the culvert law proposal.
//!
//! Usage: `culvert-profile <headwater|jump|law|circle|arc|show> <calls>`.
//!
//! `circle` and `arc` run `CulvertOutletBatch::headwater` on one mild partly full lane: a
//! circle, or a pipe-arch of a similar rise.
//!
//! `show` prints the jump rows of the outlet lanes and the law outputs once, for comparing
//! enclosures across changes.
//!
//! `headwater` runs `CulvertOutletBatch::headwater` over the outlet layout lanes: full, mild
//! partly full, rising, adverse, steep and jump rows on circles and boxes. `jump` runs
//! `CulvertOutletBatch::jump` over the same lanes. `law` runs `StructureBatch::propose` on
//! culvert lanes whose outlet headwater governs. Setup runs once, then the query runs `calls`
//! times. Run it under `perf stat -e instructions:u` with `0` calls and with `N` calls: the
//! difference over `N` is the instruction count per call.

mod alloc;

#[allow(dead_code)]
#[path = "../../../tests/culvert_outlets/lanes.rs"]
mod lanes;
#[allow(dead_code)]
#[path = "../../../tests/culverts/outlet_rows.rs"]
mod outlet_rows;
#[allow(dead_code)]
#[path = "../../../tests/culvert_laws/support.rs"]
mod support;

use alloc::{Counting, Totals};
use hydr::structures::{
    CulvertBarrel, CulvertInlet, CulvertInletBatch, CulvertOutletBatch, StructureBatch,
    StructureHeads, StructureLaw,
};
use numr::runtime::cpu::{CpuDevice, CpuRuntime};
use numr::tensor::Tensor;
use support::{Culvert, client, elements, heads, wet};

#[global_allocator]
static GLOBAL: Counting = Counting;

type Result<T = ()> = std::result::Result<T, Box<dyn std::error::Error>>;

/// Outlet batch and query tensors of the layout lanes.
type Query = (
    CulvertOutletBatch<CpuRuntime>,
    Tensor<CpuRuntime>,
    Tensor<CpuRuntime>,
);

fn outlet_query() -> Result<Query> {
    let lanes = lanes::layout_lanes()?;
    lanes::query::<CpuRuntime>(&client(), &lanes)
}

/// Mild, adverse and steep culverts with the discharges and tailwaters they run at.
fn law_lanes() -> Result<(Vec<Culvert>, Vec<f64>, Vec<f64>)> {
    let circle = CulvertBarrel::circular(1.2)?;
    let rectangle = CulvertBarrel::rectangular_box(2.0, 1.5)?;
    let mild = |barrel, code| -> Result<Culvert> {
        Ok(Culvert::both(
            barrel,
            [0.03, 0.0],
            CulvertInlet::hds5(code)?,
            0.5,
        ))
    };
    let mut steep = Culvert::both(rectangle, [0.8, 0.0], CulvertInlet::hds5(9)?, 0.5);
    steep.length_m = 40.0;
    steep.end_b = None;
    let mut culverts = Vec::new();
    let mut q = Vec::new();
    let mut tw = Vec::new();
    let sets = [
        (mild(circle, 1)?, [(3.0, 1.0), (5.0, 2.0), (2.0, 0.6)]),
        (mild(rectangle, 26)?, [(3.0, 1.0), (5.0, 2.0), (8.0, 1.2)]),
        (
            Culvert::both(circle, [0.0, 0.3], CulvertInlet::hds5(3)?, 0.5),
            [(3.0, 1.0), (5.0, 2.0), (2.0, 0.5)],
        ),
        (steep, [(15.0, 1.8), (18.0, 1.8), (21.0, 1.8)]),
    ];
    for (culvert, points) in sets {
        for (flow, tail) in points {
            culverts.push(culvert);
            q.push(flow);
            tw.push(tail);
        }
    }
    Ok((culverts, q, tw))
}

/// Law batch and head packets at the larger control headwater, outlet-governed lanes only.
fn law_query() -> Result<(StructureBatch<CpuRuntime>, Vec<StructureHeads>)> {
    let (culverts, q, tw) = law_lanes()?;
    let mut inlet_rows = Vec::new();
    let mut outlet_rows = Vec::new();
    for culvert in &culverts {
        let StructureLaw::Culvert(parameters) = culvert.law()? else {
            return Err("fixture law is not a culvert".into());
        };
        let (inlet, outlet) = parameters.forward_rows().ok_or("no A end")?;
        inlet_rows.push(inlet);
        outlet_rows.push(outlet);
    }
    let device = CpuDevice::new();
    let column =
        |values: &[f64]| Tensor::<CpuRuntime>::from_slice(values, &[values.len()], &device);
    let inlet = CulvertInletBatch::<CpuRuntime>::new(client(), inlet_rows)?
        .headwater(&column(&q)?)?
        .project()?;
    let outlet = CulvertOutletBatch::<CpuRuntime>::new(client(), outlet_rows)?
        .headwater(&column(&q)?, &column(&tw)?)?
        .project()?;
    let mut kept = Vec::new();
    let mut packets = Vec::new();
    for (index, culvert) in culverts.iter().enumerate() {
        let (Some(i), Some(o), Some(&t)) = (inlet.get(index), outlet.get(index), tw.get(index))
        else {
            return Err("lane count mismatch".into());
        };
        let hi = 0.5 * (i.headwater_m.0 + i.headwater_m.1);
        let ho = 0.5 * (o.headwater_m.0 + o.headwater_m.1);
        if !(ho.is_finite() && ho > hi * (1.0 + 1e-6)) {
            continue;
        }
        kept.push(*culvert);
        packets.push((wet(culvert.invert_a_m + ho), wet(culvert.invert_b_m + t)));
    }
    let elements = elements(&kept)?;
    let packets = elements
        .iter()
        .zip(packets)
        .map(|(element, (a, b))| heads(element, a, b))
        .collect();
    let batch = StructureBatch::<CpuRuntime>::with_culverts(client(), elements)?;
    Ok((batch, packets))
}

/// One mild partly full lane: a pipe-arch when `arc`, else a circle of a similar rise.
fn single_lane(arc: bool) -> Result<Query> {
    let barrel = if arc {
        CulvertBarrel::pipe_arch(1.8542, 1.397, 0.4572, 1.938_528)?
    } else {
        CulvertBarrel::circular(1.4)?
    };
    let row = hydr::structures::CulvertOutletRow {
        barrel,
        length_m: 40.0,
        manning_n: 0.013,
        entrance_loss: 0.5,
        slope: 0.001,
        gravity_m_per_s2: 9.81,
        face_blockage: hydr::structures::BlockageFraction::NONE,
    };
    let lane = lanes::Lane {
        row,
        discharge: 1.2,
        tailwater: 1.1,
    };
    lanes::query::<CpuRuntime>(&client(), &[lane])
}

/// Prints every jump row of the layout lanes and every law output, one line each.
fn show() -> Result {
    let (batch, discharge, tailwater) = outlet_query()?;
    let found = batch.jump(&discharge, &tailwater)?;
    for (row, jump) in found.headwater().project()?.iter().zip(found.project()?) {
        println!("{row:?} {jump:?}");
    }
    let (batch, packets) = law_query()?;
    for output in batch.propose(&packets)?.outputs()? {
        println!(
            "{:?} {:?} {:?}",
            output.regime, output.discharge_m3_per_s, output.hydraulic_operands
        );
    }
    Ok(())
}

fn main() -> Result {
    let args: Vec<String> = std::env::args().collect();
    let usage = "usage: culvert-profile <headwater|jump|law|circle|arc|show> <calls>";
    let mode = args.get(1).ok_or(usage)?;
    let calls: usize = args.get(2).ok_or(usage)?.parse()?;
    let before;
    let lanes;
    match mode.as_str() {
        "headwater" | "jump" => {
            let (batch, discharge, tailwater) = outlet_query()?;
            lanes = discharge.numel();
            before = Totals::now();
            for _ in 0..calls {
                if mode == "jump" {
                    batch.jump(&discharge, &tailwater)?;
                } else {
                    batch.headwater(&discharge, &tailwater)?;
                }
            }
        }
        "circle" | "arc" => {
            let (batch, discharge, tailwater) = single_lane(mode == "arc")?;
            lanes = discharge.numel();
            before = Totals::now();
            for _ in 0..calls {
                batch.headwater(&discharge, &tailwater)?;
            }
        }
        "law" => {
            let (batch, packets) = law_query()?;
            lanes = packets.len();
            before = Totals::now();
            for _ in 0..calls {
                batch.propose(&packets)?.outputs()?;
            }
        }
        "show" => return show(),
        other => return Err(format!("mode={other:?}: {usage}").into()),
    }
    let spent = Totals::now().since(before);
    let per = calls.max(1) as u64;
    println!(
        "mode={mode} lanes={lanes} calls={calls} allocations/call={} bytes/call={}",
        spent.calls / per,
        spent.bytes / per
    );
    Ok(())
}
