//! Frozen corpus coverage shared by CPU and device tests.
//!
//! CPU tests replay every frozen case. Device tests replay one case per class, so each
//! regime and branch reaches its device kernels once against the same frozen values.

use std::collections::BTreeSet;

/// How much of a frozen corpus one test replays.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Coverage {
    /// Every frozen case.
    Full,
    /// The first frozen case of each class, in corpus order.
    Sample,
}

impl Coverage {
    /// Keeps the cases this coverage replays, in corpus order.
    pub fn select<T, K: Ord>(
        self,
        cases: impl IntoIterator<Item = T>,
        class: impl Fn(&T) -> K,
    ) -> Vec<T> {
        let mut seen = BTreeSet::new();
        cases
            .into_iter()
            .filter(|case| self == Self::Full || seen.insert(class(case)))
            .collect()
    }

    /// Returns `true` when every frozen case runs.
    pub fn is_full(self) -> bool {
        self == Self::Full
    }
}
