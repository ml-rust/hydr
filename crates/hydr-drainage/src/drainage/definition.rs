//! Admission definition types for a pinned EPA SWMM v5.2.4 drainage network.

use super::error::DrainageError;
use super::protocol::sha256_hex;
use std::path::{Path, PathBuf};

/// Pressurized surcharge formulation selected for dynamic-wave conduit routing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SwmmSurchargeMethod {
    /// Classical SWMM EXTRAN surcharge algorithm (`SURCHARGE_METHOD EXTRAN`).
    Extran,
    /// Preissmann slot formulation (`SURCHARGE_METHOD SLOT`).
    Slot,
}

impl SwmmSurchargeMethod {
    /// Stable kebab-case identifier matching the capability option values.
    pub fn id(self) -> &'static str {
        match self {
            Self::Extran => "extran",
            Self::Slot => "slot",
        }
    }

    /// Parses the integer `SurchargeMethod` enum reported by the SWMM v5.2.4 worker.
    pub(crate) fn from_worker_code(code: i32) -> Result<Self, DrainageError> {
        match code {
            0 => Ok(Self::Extran),
            1 => Ok(Self::Slot),
            other => Err(DrainageError::process(format!(
                "unknown SWMM SurchargeMethod code `{other}` from worker OPEN"
            ))),
        }
    }
}

/// Control-authority mode for structures inside the SWMM drainage network.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SwmmControlAuthority {
    /// SWMM native `[CONTROLS]` own all internal drainage structure settings.
    NativeSwmm,
    /// `hydr` supplies target commands while SWMM native actuation moves settings.
    HydrCommandsNativeActuation,
    /// `hydr` actuator imposes link positions while SWMM native rules are excluded.
    HydrActuatorImposedPosition,
    /// `hydr` imposes a prescribed trajectory on drainage settings.
    PrescribedHydrTrajectory,
}

impl SwmmControlAuthority {
    /// Stable kebab-case identifier.
    pub fn id(self) -> &'static str {
        match self {
            Self::NativeSwmm => "native-swmm",
            Self::HydrCommandsNativeActuation => "hydr-commands-native-actuation",
            Self::HydrActuatorImposedPosition => "hydr-actuator-imposed-position",
            Self::PrescribedHydrTrajectory => "prescribed-hydr-trajectory",
        }
    }
}

/// Validated definition required to open a pinned EPA SWMM v5.2.4 drainage worker.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SwmmDrainageDefinition {
    inp_content: String,
    inp_sha256: String,
    vertical_datum: String,
    surcharge_method: SwmmSurchargeMethod,
    control_authority: SwmmControlAuthority,
    worker_binary: PathBuf,
    provenance: String,
    surcharge_evidence: String,
}

impl SwmmDrainageDefinition {
    /// Validates and constructs a SWMM drainage definition.
    ///
    /// Rejects empty `.inp` text, empty vertical datum, empty provenance or surcharge
    /// evidence, and unattached hybrid control-authority modes.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        inp_content: impl Into<String>,
        vertical_datum: impl Into<String>,
        surcharge_method: SwmmSurchargeMethod,
        control_authority: SwmmControlAuthority,
        worker_binary: impl Into<PathBuf>,
        provenance: impl Into<String>,
        surcharge_evidence: impl Into<String>,
    ) -> Result<Self, DrainageError> {
        let inp_content = inp_content.into();
        if inp_content.trim().is_empty() {
            return Err(DrainageError::input(
                "SWMM .inp content must not be empty; provide a complete dynamic-wave input file",
            ));
        }
        let vertical_datum = require_non_empty(vertical_datum.into(), "vertical_datum")?;
        let provenance = require_non_empty(provenance.into(), "provenance")?;
        let surcharge_evidence =
            require_non_empty(surcharge_evidence.into(), "surcharge_evidence")?;
        if control_authority != SwmmControlAuthority::NativeSwmm {
            return Err(DrainageError::input(format!(
                "control_authority `{}` requires coupled authority attachment (`native-control-authority`); use `SwmmControlAuthority::NativeSwmm` for standalone drainage",
                control_authority.id()
            )));
        }
        let worker_binary = worker_binary.into();
        if worker_binary.as_os_str().is_empty() {
            return Err(DrainageError::input(
                "worker_binary path must not be empty; point to the built `swmm-drainage-worker`",
            ));
        }
        let inp_sha256 = sha256_hex(inp_content.as_bytes());
        Ok(Self {
            inp_content,
            inp_sha256,
            vertical_datum,
            surcharge_method,
            control_authority,
            worker_binary,
            provenance,
            surcharge_evidence,
        })
    }

    /// Raw SWMM `.inp` file text.
    pub fn inp_content(&self) -> &str {
        &self.inp_content
    }

    /// Lowercase hex SHA-256 digest of `inp_content`.
    pub fn inp_sha256(&self) -> &str {
        &self.inp_sha256
    }

    /// Explicit vertical datum shared with any coupled surface or channel model.
    pub fn vertical_datum(&self) -> &str {
        &self.vertical_datum
    }

    /// Declared surcharge formulation (`Extran` or `Slot`).
    pub fn surcharge_method(&self) -> SwmmSurchargeMethod {
        self.surcharge_method
    }

    /// Declared control-authority mode.
    pub fn control_authority(&self) -> SwmmControlAuthority {
        self.control_authority
    }

    /// Path to the pinned `swmm-drainage-worker` executable.
    pub fn worker_binary(&self) -> &Path {
        &self.worker_binary
    }

    /// Provenance string naming the pinned SWMM build and source commit.
    pub fn provenance(&self) -> &str {
        &self.provenance
    }

    /// Evidence statement justifying the selected surcharge formulation.
    pub fn surcharge_evidence(&self) -> &str {
        &self.surcharge_evidence
    }
}

fn require_non_empty(value: String, field: &str) -> Result<String, DrainageError> {
    if value.trim().is_empty() {
        return Err(DrainageError::input(format!(
            "{field} must not be empty; supply an explicit {field} string"
        )));
    }
    Ok(value)
}
