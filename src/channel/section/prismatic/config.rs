use super::super::CrossSectionError;

/// Maximum padded band entries in a compiled table or one query batch.
pub const PRISMATIC_BAND_WORK_LIMIT: usize = 1 << 20;

/// Accepted width of a recovered geometric-depth enclosure.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PrismaticRecoveryConfig {
    /// Absolute enclosure-width allowance, in metres.
    pub depth_atol_m: f64,
    /// Relative allowance multiplied by the largest absolute depth endpoint.
    pub depth_rtol: f64,
}

impl Default for PrismaticRecoveryConfig {
    fn default() -> Self {
        Self {
            depth_atol_m: 1e-12,
            depth_rtol: 1e-12,
        }
    }
}

impl PrismaticRecoveryConfig {
    pub(super) fn check(&self) -> Result<(), CrossSectionError> {
        for (name, value) in [
            ("depth_atol_m", self.depth_atol_m),
            ("depth_rtol", self.depth_rtol),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(CrossSectionError::input(format!(
                    "{name}={value}: supply a finite nonnegative depth-width allowance"
                )));
            }
        }
        Ok(())
    }
}

pub(super) fn check_work(rows: usize, bands: usize) -> Result<usize, CrossSectionError> {
    let entries = rows.checked_mul(bands).ok_or_else(|| {
        CrossSectionError::input(format!(
            "rows={rows}, bands={bands}: padded band work overflows usize. Split the batch"
        ))
    })?;
    if entries > PRISMATIC_BAND_WORK_LIMIT {
        return Err(CrossSectionError::input(format!(
            "rows={rows}, bands={bands}, padded entries={entries}: split the batch into at most {PRISMATIC_BAND_WORK_LIMIT} entries"
        )));
    }
    u32::try_from(entries).map_err(|_| {
        CrossSectionError::input("padded band count exceeds U32 indexing. Split the batch")
    })?;
    entries
        .checked_mul(std::mem::size_of::<f64>())
        .ok_or_else(|| {
            CrossSectionError::input("padded band allocation bytes overflow usize. Split the batch")
        })?;
    Ok(entries)
}
