//! Subprocess protocol client for the pinned EPA SWMM v5.2.4 worker.

use super::definition::SwmmSurchargeMethod;
use super::error::DrainageError;
use super::state::{
    FT_TO_M, FT3_TO_M3, SwmmContinuityLedger, SwmmDrainageState, SwmmInpUnitSystem, SwmmLinkState,
    SwmmNetworkSummary, SwmmNodeState,
};
use sha2::{Digest, Sha256};
use std::fs;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};

static INSTANCE_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Isolated `swmm-drainage-worker` child process and its temporary work directory.
pub(crate) struct SwmmWorker {
    child: Child,
    stdin: BufWriter<ChildStdin>,
    stdout: BufReader<ChildStdout>,
    work_dir: PathBuf,
}

impl SwmmWorker {
    /// Spawns `worker_binary`, writes `inp_content`, and executes `OPEN`.
    pub(crate) fn open(
        worker_binary: &Path,
        inp_content: &str,
    ) -> Result<(Self, SwmmNetworkSummary), DrainageError> {
        if !worker_binary.is_file() {
            return Err(DrainageError::unavailable(format!(
                "pinned SWMM worker binary `{}` does not exist; build it with `python3 tools/reference/swmm_drainage_build.py`",
                worker_binary.display()
            )));
        }
        let seq = INSTANCE_COUNTER.fetch_add(1, Ordering::Relaxed);
        let work_dir = std::env::temp_dir().join(format!("hydr-swmm-{}-{seq}", std::process::id()));
        fs::create_dir_all(&work_dir).map_err(|err| {
            DrainageError::process(format!(
                "failed to create SWMM temp directory `{}`: {err}",
                work_dir.display()
            ))
        })?;
        let inp_path = work_dir.join("model.inp");
        let rpt_path = work_dir.join("model.rpt");
        if let Err(err) = fs::write(&inp_path, inp_content) {
            let _ = fs::remove_dir_all(&work_dir);
            return Err(DrainageError::process(format!(
                "failed to write SWMM .inp file `{}`: {err}",
                inp_path.display()
            )));
        }

        let mut child = Command::new(worker_binary)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|err| {
                let _ = fs::remove_dir_all(&work_dir);
                DrainageError::unavailable(format!(
                    "failed to spawn SWMM worker `{}`: {err}",
                    worker_binary.display()
                ))
            })?;

        let Some(stdin) = child.stdin.take() else {
            let _ = child.kill();
            let _ = child.wait();
            let _ = fs::remove_dir_all(&work_dir);
            return Err(DrainageError::process("SWMM worker stdin pipe unavailable"));
        };
        let Some(stdout) = child.stdout.take() else {
            let _ = child.kill();
            let _ = child.wait();
            let _ = fs::remove_dir_all(&work_dir);
            return Err(DrainageError::process(
                "SWMM worker stdout pipe unavailable",
            ));
        };

        let mut worker = Self {
            child,
            stdin: BufWriter::new(stdin),
            stdout: BufReader::new(stdout),
            work_dir,
        };
        let cmd = format!("OPEN {} {}", inp_path.display(), rpt_path.display());
        let reply = worker.send_and_read_line(&cmd)?;
        let summary = parse_open_reply(&reply)?;
        Ok((worker, summary))
    }

    /// Sets `Node[node_index].apiExtInflow` in SWMM internal cfs.
    pub(crate) fn set_inflow(
        &mut self,
        node_index: usize,
        rate_cfs: f64,
    ) -> Result<(), DrainageError> {
        if !rate_cfs.is_finite() {
            return Err(DrainageError::input(format!(
                "non-finite inflow rate for node {node_index}: {rate_cfs}"
            )));
        }
        let cmd = format!("SET_INFLOW {node_index} {}", encode_f64_hex(rate_cfs));
        let reply = self.send_and_read_line(&cmd)?;
        if reply == "OK" {
            Ok(())
        } else {
            Err(DrainageError::process(format!(
                "SWMM SET_INFLOW failed for node {node_index}: {reply}"
            )))
        }
    }

    /// Advances the worker by `dt_s` seconds and returns the reported simulation time in seconds.
    pub(crate) fn step(&mut self, dt_s: f64) -> Result<f64, DrainageError> {
        if !dt_s.is_finite() || dt_s <= 0.0 {
            return Err(DrainageError::input(format!(
                "SWMM step dt_s must be finite and positive, got {dt_s}"
            )));
        }
        let cmd = format!("STEP {}", encode_f64_hex(dt_s));
        let reply = self.send_and_read_line(&cmd)?;
        let Some(hex) = reply.strip_prefix("OK ") else {
            return Err(DrainageError::numerical(format!(
                "SWMM STEP({dt_s}) failed: {reply}"
            )));
        };
        decode_f64_hex(hex.trim())
    }

    /// Queries the full worker state and converts all SWMM internal US units to canonical SI.
    pub(crate) fn state(&mut self) -> Result<SwmmDrainageState, DrainageError> {
        let header = self.send_and_read_line("STATE")?;
        let Some(rest) = header.strip_prefix("STATE ") else {
            return Err(DrainageError::process(format!(
                "unexpected STATE header from SWMM worker: {header}"
            )));
        };
        let vals = parse_hex_fields(rest, 10, "STATE")?;
        let time_s = vals[0];
        let variable_step_s = vals[1];
        let continuity = SwmmContinuityLedger {
            initial_storage_m3: vals[2] * FT3_TO_M3,
            external_inflow_m3: vals[3] * FT3_TO_M3,
            outflow_m3: vals[4] * FT3_TO_M3,
            flooding_m3: vals[5] * FT3_TO_M3,
            evaporation_loss_m3: vals[6] * FT3_TO_M3,
            seepage_loss_m3: vals[7] * FT3_TO_M3,
            final_storage_m3: vals[8] * FT3_TO_M3,
            continuity_error_pct: vals[9],
        };

        let mut nodes = Vec::new();
        let mut links = Vec::new();
        loop {
            let line = self.read_line()?;
            if line == "END_STATE" {
                break;
            }
            if let Some(body) = line.strip_prefix("NODE ") {
                nodes.push(parse_node_line(body)?);
            } else if let Some(body) = line.strip_prefix("LINK ") {
                links.push(parse_link_line(body)?);
            } else {
                return Err(DrainageError::process(format!(
                    "unexpected line inside STATE response: {line}"
                )));
            }
        }
        Ok(SwmmDrainageState {
            time_s,
            variable_step_s,
            continuity,
            nodes,
            links,
        })
    }

    /// Captures the complete hidden-state snapshot blob as lowercase hex.
    pub(crate) fn snapshot(&mut self) -> Result<String, DrainageError> {
        let reply = self.send_and_read_line("SNAPSHOT")?;
        let Some(hex) = reply.strip_prefix("SNAPSHOT ") else {
            return Err(DrainageError::process(format!(
                "SWMM SNAPSHOT failed: {reply}"
            )));
        };
        let hex = hex.trim();
        if hex.is_empty() || hex.len() % 2 != 0 {
            return Err(DrainageError::process(
                "SWMM SNAPSHOT returned invalid hex payload",
            ));
        }
        Ok(hex.to_string())
    }

    /// Restores the worker's complete hidden state from a previously captured hex blob.
    pub(crate) fn restore(&mut self, snapshot_hex: &str) -> Result<(), DrainageError> {
        let cmd = format!("RESTORE {snapshot_hex}");
        let reply = self.send_and_read_line(&cmd)?;
        if reply == "OK" {
            Ok(())
        } else {
            Err(DrainageError::process(format!(
                "SWMM RESTORE failed: {reply}"
            )))
        }
    }

    fn send_and_read_line(&mut self, command: &str) -> Result<String, DrainageError> {
        writeln!(self.stdin, "{command}").map_err(|err| {
            DrainageError::process(format!("failed writing command to SWMM worker: {err}"))
        })?;
        self.stdin.flush().map_err(|err| {
            DrainageError::process(format!("failed flushing command to SWMM worker: {err}"))
        })?;
        self.read_line()
    }

    fn read_line(&mut self) -> Result<String, DrainageError> {
        let mut line = String::new();
        let n = self.stdout.read_line(&mut line).map_err(|err| {
            DrainageError::process(format!("failed reading from SWMM worker: {err}"))
        })?;
        if n == 0 {
            return Err(DrainageError::process(
                "SWMM worker closed stdout unexpectedly",
            ));
        }
        Ok(line.trim_end_matches(['\r', '\n']).to_string())
    }
}

impl Drop for SwmmWorker {
    fn drop(&mut self) {
        let _ = writeln!(self.stdin, "CLOSE");
        let _ = self.stdin.flush();
        if self.child.wait().is_err() {
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
        let _ = fs::remove_dir_all(&self.work_dir);
    }
}

fn parse_open_reply(reply: &str) -> Result<SwmmNetworkSummary, DrainageError> {
    let Some(rest) = reply.strip_prefix("OK ") else {
        return Err(DrainageError::input(format!(
            "SWMM OPEN rejected input: {reply}"
        )));
    };
    let parts: Vec<&str> = rest.split_whitespace().collect();
    if parts.len() != 10 {
        return Err(DrainageError::process(format!(
            "expected 10 fields in SWMM OPEN reply, got {}: {reply}",
            parts.len()
        )));
    }
    let parse_usize = |idx: usize| -> Result<usize, DrainageError> {
        parts[idx].parse::<usize>().map_err(|err| {
            DrainageError::process(format!("invalid integer in SWMM OPEN[{idx}]: {err}"))
        })
    };
    let parse_i32 = |idx: usize| -> Result<i32, DrainageError> {
        parts[idx].parse::<i32>().map_err(|err| {
            DrainageError::process(format!("invalid integer in SWMM OPEN[{idx}]: {err}"))
        })
    };
    Ok(SwmmNetworkSummary {
        node_count: parse_usize(0)?,
        link_count: parse_usize(1)?,
        conduit_count: parse_usize(2)?,
        storage_count: parse_usize(3)?,
        outfall_count: parse_usize(4)?,
        orifice_count: parse_usize(5)?,
        weir_count: parse_usize(6)?,
        rule_count: parse_usize(7)?,
        surcharge_method: SwmmSurchargeMethod::from_worker_code(parse_i32(8)?)?,
        inp_unit_system: SwmmInpUnitSystem::from_worker_code(parse_i32(9)?)?,
    })
}

fn parse_node_line(body: &str) -> Result<SwmmNodeState, DrainageError> {
    let parts: Vec<&str> = body.split_whitespace().collect();
    if parts.len() != 9 {
        return Err(DrainageError::process(format!(
            "expected 9 fields in NODE line, got {}: {body}",
            parts.len()
        )));
    }
    let index = parts[0].parse::<usize>().map_err(|err| {
        DrainageError::process(format!("invalid node index `{}`: {err}", parts[0]))
    })?;
    let id = parts[1].to_string();
    let invert_m = decode_f64_hex(parts[2])? * FT_TO_M;
    let full_depth_m = decode_f64_hex(parts[3])? * FT_TO_M;
    let depth_m = decode_f64_hex(parts[4])? * FT_TO_M;
    let volume_m3 = decode_f64_hex(parts[5])? * FT3_TO_M3;
    let inflow_m3_per_s = decode_f64_hex(parts[6])? * FT3_TO_M3;
    let outflow_m3_per_s = decode_f64_hex(parts[7])? * FT3_TO_M3;
    let overflow_m3_per_s = decode_f64_hex(parts[8])? * FT3_TO_M3;
    Ok(SwmmNodeState {
        index,
        id,
        invert_m,
        full_depth_m,
        depth_m,
        head_m: invert_m + depth_m,
        volume_m3,
        inflow_m3_per_s,
        outflow_m3_per_s,
        overflow_m3_per_s,
    })
}

fn parse_link_line(body: &str) -> Result<SwmmLinkState, DrainageError> {
    let parts: Vec<&str> = body.split_whitespace().collect();
    if parts.len() != 8 {
        return Err(DrainageError::process(format!(
            "expected 8 fields in LINK line, got {}: {body}",
            parts.len()
        )));
    }
    let index = parts[0].parse::<usize>().map_err(|err| {
        DrainageError::process(format!("invalid link index `{}`: {err}", parts[0]))
    })?;
    let id = parts[1].to_string();
    Ok(SwmmLinkState {
        index,
        id,
        flow_m3_per_s: decode_f64_hex(parts[2])? * FT3_TO_M3,
        depth_m: decode_f64_hex(parts[3])? * FT_TO_M,
        volume_m3: decode_f64_hex(parts[4])? * FT3_TO_M3,
        setting: decode_f64_hex(parts[5])?,
        target_setting: decode_f64_hex(parts[6])?,
        time_last_set_days: decode_f64_hex(parts[7])?,
    })
}

fn parse_hex_fields(text: &str, count: usize, label: &str) -> Result<Vec<f64>, DrainageError> {
    let parts: Vec<&str> = text.split_whitespace().collect();
    if parts.len() != count {
        return Err(DrainageError::process(format!(
            "expected {count} hex fields in {label}, got {}",
            parts.len()
        )));
    }
    parts.into_iter().map(decode_f64_hex).collect()
}

/// Encodes an IEEE-754 `f64` as 16 lowercase hexadecimal digits.
pub(crate) fn encode_f64_hex(value: f64) -> String {
    format!("{:016x}", value.to_bits())
}

/// Decodes a 16-digit hexadecimal IEEE-754 `f64`, rejecting NaN and infinities.
pub(crate) fn decode_f64_hex(hex: &str) -> Result<f64, DrainageError> {
    let bits = u64::from_str_radix(hex, 16)
        .map_err(|err| DrainageError::process(format!("invalid f64 hex `{hex}`: {err}")))?;
    let value = f64::from_bits(bits);
    if !value.is_finite() {
        return Err(DrainageError::numerical(format!(
            "SWMM worker produced non-finite value from hex `{hex}`"
        )));
    }
    Ok(value)
}

/// Computes a lowercase hexadecimal SHA-256 digest.
pub(crate) fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut out = String::with_capacity(64);
    for byte in digest {
        use std::fmt::Write as _;
        let _ = write!(&mut out, "{byte:02x}");
    }
    out
}
