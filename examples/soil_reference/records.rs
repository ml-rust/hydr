//! Strict schema and full-state native evidence parsing.

use hydr::soil::NativeSoilState;
use std::collections::BTreeSet;
use std::path::Path;

pub(super) const HEADER: &str = "case,index,operation,model,initial_mode,S_ft,Ks_ft_per_s,IMDmax,initial_IMD,initial_F_ft,initial_Fu_ft,initial_Sat,initial_T_s,before_IMD,before_F_ft,before_Fu_ft,before_Sat,before_T_s,dt_s,irate_ft_per_s,ponded_depth_ft,f1_ft,c1_ft,root_ks_ft_per_s,root_ts_s,rate_ft_per_s,interval_infiltration_ft,accumulated_interval_infiltration_ft,F2_ft,Lu_ft,IMD,F_ft,Fu_ft,Sat,T_s";

pub struct Record {
    pub case: String,
    pub index: usize,
    pub operation: String,
    pub model: usize,
    pub initial_mode: String,
    fields: Vec<Option<f64>>,
}

impl Record {
    pub fn value(&self, index: usize) -> Result<f64, Box<dyn std::error::Error>> {
        self.fields.get(index).and_then(|x| *x).ok_or_else(|| {
            format!(
                "case={} index={} missing field={index}",
                self.case, self.index
            )
            .into()
        })
    }
    pub fn state(&self, start: usize) -> Result<NativeSoilState, Box<dyn std::error::Error>> {
        let sat = self.value(start + 3)?;
        if sat != 0.0 && sat != 1.0 {
            return Err(format!("case={} invalid Sat={sat}", self.case).into());
        }
        Ok(NativeSoilState {
            imd: self.value(start)?,
            f_ft: self.value(start + 1)?,
            fu_ft: self.value(start + 2)?,
            saturated: sat == 1.0,
            t_s: self.value(start + 4)?,
        })
    }
}

pub fn read(path: &Path) -> Result<Vec<Record>, Box<dyn std::error::Error>> {
    parse(&std::fs::read_to_string(path)?)
}

pub fn parse(text: &str) -> Result<Vec<Record>, Box<dyn std::error::Error>> {
    let mut lines = text.lines();
    if lines.next() != Some(HEADER) {
        return Err("native CSV header differs from the full soil schema".into());
    }
    let mut records: Vec<Record> = Vec::new();
    let mut seen = BTreeSet::new();
    for (line, content) in lines.enumerate() {
        let fields: Vec<_> = content.split(',').collect();
        if fields.len() != 35 {
            return Err(format!("line={} fields={}: require 35", line + 2, fields.len()).into());
        }
        let index: usize = fields[1].parse()?;
        let model: usize = fields[3].parse()?;
        if fields[0].is_empty()
            || !matches!(model, 2 | 3)
            || !matches!(fields[2], "INIT" | "STATE" | "STEP" | "F2")
            || !matches!(fields[4], "INIT" | "STATE")
        {
            return Err(format!("line={}: invalid native command metadata", line + 2).into());
        }
        let continuation = records.last().is_some_and(|r| r.case == fields[0]);
        let expected = if continuation {
            records.last().map_or(1, |r| r.index + 1)
        } else {
            1
        };
        if index != expected || (!continuation && !seen.insert(fields[0].to_owned())) {
            return Err(format!("line={}: noncontiguous case or command index", line + 2).into());
        }
        if index == 1 && (fields[2] != fields[4] || !matches!(fields[2], "INIT" | "STATE"))
            || index > 1 && !matches!(fields[2], "STEP" | "F2")
        {
            return Err(format!("line={}: invalid initialization ordering", line + 2).into());
        }
        let mut numeric = vec![None; 35];
        for i in 5..35 {
            if !fields[i].is_empty() {
                let value: f64 = fields[i].parse()?;
                if !value.is_finite() {
                    return Err(format!("line={} field={i}: nonfinite number", line + 2).into());
                }
                numeric[i] = Some(value);
            }
        }
        let row = Record {
            case: fields[0].into(),
            index,
            model,
            operation: fields[2].into(),
            initial_mode: fields[4].into(),
            fields: numeric,
        };
        for i in 5..35 {
            let common = (5..13).contains(&i) || i == 27 || (29..35).contains(&i);
            let before = index > 1 && (13..18).contains(&i);
            let command = match row.operation.as_str() {
                "STEP" => (18..21).contains(&i) || (25..27).contains(&i),
                "F2" => (21..25).contains(&i) || i == 28,
                _ => false,
            };
            if row.fields[i].is_some() != (common || before || command) {
                return Err(format!(
                    "line={} field={i}: presence differs from command schema",
                    line + 2
                )
                .into());
            }
        }
        for i in (5..13).chain(29..35) {
            row.value(i)?;
        }
        row.state(8)?;
        row.state(30)?;
        if index > 1 {
            row.state(13)?;
        }
        match row.operation.as_str() {
            "STEP" => {
                for i in 18..=20 {
                    row.value(i)?;
                }
                row.value(25)?;
                row.value(26)?;
            }
            "F2" => {
                for i in 21..=24 {
                    row.value(i)?;
                }
                row.value(28)?;
            }
            _ => {}
        }
        if row.fields[25].is_some() != (row.operation == "STEP")
            || row.fields[28].is_some() != (row.operation == "F2")
        {
            return Err(format!("line={}: optional output differs from command", line + 2).into());
        }
        if continuation {
            let previous = records.last().ok_or("missing preceding command")?;
            if row.model != previous.model || row.initial_mode != previous.initial_mode {
                return Err(
                    format!("case={}: model or initialization mode changed", row.case).into(),
                );
            }
            for i in 5..13 {
                if row.value(i)?.to_bits() != previous.value(i)?.to_bits() {
                    return Err(format!("case={} initial field={i} changed", row.case).into());
                }
            }
            if row.state(13)? != previous.state(30)? {
                return Err(format!("case={} preceding state differs", row.case).into());
            }
        }
        records.push(row);
    }
    if records.is_empty() {
        return Err("native CSV contains no cases".into());
    }
    Ok(records)
}
