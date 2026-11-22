//! Independent constant-rain reference for runoff delivered into the closed pond.
//!
//! Inputs are binary64 bits. Each row holds the nearest binary64 of the exact delivered volume
//! at one time. The largest error over the rows must fall at every refinement. The finest
//! run's relative error at the final row must stay within `FINEST_RELATIVE`.

// Transcribed from `resource/evidence/runoff/constant_rain.json`, which
// `constant_rain.py` beside it writes. Regenerate instead of editing.

/// SHA256 of the frozen `constant_rain.json`.
pub const RAIN_SHA256: &str = "36db13b34d4e5805ebe10144eb0651be800e87e727972d81250c6f24e47cfe40";

/// SI coefficient of the subarea the reference integrates.
pub const ALPHA: u64 = 0x40000b403359a5ca;
/// Constant rain rate, in m/s.
pub const RAIN: u64 = 0x3f50624dd2f1a9fc;
/// Common interval lengths of the refinement, in s.
pub const STEPS: [f64; 6] = [1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125];
/// Largest relative error of the finest run at the final row.
pub const FINEST_RELATIVE: f64 = 2e-2;
/// `(time bits, delivered volume bits)` rows, in s and m³.
pub const ROWS: [(u64, u64); 4] = [
    (0x3ff0000000000000, 0x3eff4d59cbcbc4c9),
    (0x4000000000000000, 0x3f286feb02427fac),
    (0x4008000000000000, 0x3f419ace222c9b86),
    (0x4010000000000000, 0x3f526efcb821b316),
];
