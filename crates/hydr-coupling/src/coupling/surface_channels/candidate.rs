//! The owned candidate of one common trial.

use super::batch::SurfaceChannelBatch;
use super::{SurfaceChannelError, SurfaceChannelStepDiagnostics};
use numr::dtype::DType;
use numr::runtime::Runtime;

/// Complete candidate of one common interval. Dropping it rejects every child together.
#[derive(Debug)]
pub struct SurfaceChannelTrial<R: Runtime<DType = DType>> {
    pub(super) next: SurfaceChannelBatch<R>,
}

impl<R: Runtime<DType = DType>> SurfaceChannelTrial<R> {
    /// Actual common endpoint after duration limits and recalculation, in seconds.
    pub fn time(&self) -> f64 {
        self.next.time()
    }

    /// Accepts the surface, every reach, and the link counters together.
    pub fn into_next(self) -> SurfaceChannelBatch<R> {
        self.next
    }

    /// Projects the candidate's common interval receipt.
    ///
    /// # Errors
    /// Returns runtime transfer errors and [`SurfaceChannelError::Numerical`] for a candidate
    /// without its receipt or with an unknown weir regime code.
    pub fn diagnostics(&self) -> Result<SurfaceChannelStepDiagnostics, SurfaceChannelError> {
        self.next.last_step()?.ok_or_else(|| {
            SurfaceChannelError::Numerical("candidate lacks its common interval receipt".into())
        })
    }
}
