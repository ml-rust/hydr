use crate::channel::flow::surveyed::error::HydraulicResult;
use numr::{
    dtype::DType,
    ops::{CompareOps, IndexingOps, TensorOps},
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};

/// Per-row Simpson panel levels of wet star rows and every level they use.
pub(super) struct WetStarLevels<R: Runtime<DType = DType>> {
    levels: Tensor<R>,
    policies: Vec<usize>,
}

impl<R: Runtime<DType = DType>> Clone for WetStarLevels<R> {
    fn clone(&self) -> Self {
        Self {
            levels: self.levels.clone(),
            policies: self.policies.clone(),
        }
    }
}

impl<R> WetStarLevels<R>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R> + CompareOps<R> + RuntimeClient<R>,
{
    /// Every row starts at level 0, the initial Simpson panels.
    pub(super) fn initial(c: &R::Client, count: usize) -> HydraulicResult<Self> {
        Ok(Self {
            levels: Tensor::zeros(&[count], DType::I64, c.device())?,
            policies: vec![0],
        })
    }

    #[cfg(test)]
    pub(super) fn levels(&self) -> &Tensor<R> {
        &self.levels
    }

    pub(super) fn policies(&self) -> &[usize] {
        &self.policies
    }

    /// Selected rows keep every policy, so grouping skips the empty ones.
    pub(super) fn selected(&self, c: &R::Client, rows: &Tensor<R>) -> HydraulicResult<Self> {
        if let [index] = self.policies() {
            return Ok(Self {
                levels: Tensor::full_scalar(rows.shape(), DType::I64, *index as f64, c.device())?,
                policies: vec![*index],
            });
        }
        Ok(Self {
            levels: c.index_select(&self.levels, 0, rows)?,
            policies: self.policies.clone(),
        })
    }

    /// Local rows at one panel level.
    pub(super) fn rows_at(&self, c: &R::Client, index: usize) -> HydraulicResult<Tensor<R>> {
        super::source_sonic_policy::policy_rows::<R>(c, &self.levels, index)
    }

    /// Tests move local rows to one admitted panel level.
    #[cfg(test)]
    pub(super) fn with_rows_at(
        &self,
        c: &R::Client,
        rows: &Tensor<R>,
        index: usize,
    ) -> HydraulicResult<Self> {
        use numr::ops::SortingOps;
        let level = Tensor::full_scalar(rows.shape(), DType::I64, index as f64, c.device())?;
        let levels = c.index_put(&self.levels, 0, rows, &level)?;
        let mut policies = Vec::with_capacity(self.policies.len() + 1);
        for policy in &self.policies {
            let value = Tensor::full_scalar(&[], DType::I64, *policy as f64, c.device())?;
            if c.nonzero(&c.eq(&levels, &value)?)?.numel() != 0 {
                policies.push(*policy);
            }
        }
        if !policies.contains(&index) {
            policies.push(index);
        }
        Ok(Self { levels, policies })
    }
}

#[cfg(test)]
#[path = "wet_star_levels_tests.rs"]
mod tests;
