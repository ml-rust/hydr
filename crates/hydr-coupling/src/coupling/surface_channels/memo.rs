//! Child trials of one attempt, keyed by the exact bits of their exchange operands.
//!
//! One memo lives for one attempted interval. Its start state, duration, caller forcing,
//! reservations, port fluxes and inflow velocities stay fixed for that attempt. Port fluxes
//! enter the surface stage and each ported reach's reservation, so no key holds them. Each
//! child trial is then a pure function of its exchange operands:
//! - A linked reach: the inflow and withdrawal volume of each of its links.
//! - The surface: the inflow and withdrawal of every lane, and the withdrawal of every link.
//!   The link withdrawals carry the fixed channel velocity into the lane momentum.
//!
//! A trial whose operands match a kept entry in every bit takes that entry's result.
//! A converged reach repeats its last trial. A dry attempt repeats its zero-exchange trial.

use super::SurfaceChannelError;
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use std::cell::RefCell;
use std::collections::VecDeque;

/// Kept trials per child.
const CAPACITY: usize = 2;

/// Exact operand bits of one child trial.
#[derive(Clone, PartialEq, Eq, Debug)]
pub(super) struct TrialKey(Vec<u64>);

/// Operand bits of every child of one evaluation.
#[derive(Debug)]
pub(super) struct TrialKeys {
    pub surface: TrialKey,
    /// One key per linked reach, in `Context::linked` order.
    pub linked: Vec<TrialKey>,
}

impl TrialKeys {
    /// Reads `packed` in one copy and splits it into child keys.
    ///
    /// `packed` holds `[to_channel; L]`, `[to_surface; L]`, then the surface lane operands.
    /// `reach_links` lists the links of every linked reach.
    pub fn read<R: Runtime<DType = DType>>(
        packed: &Tensor<R>,
        link_count: usize,
        reach_links: &[&[usize]],
    ) -> Result<Self, SurfaceChannelError> {
        let bits: Vec<u64> = packed
            .contiguous()?
            .try_to_vec::<f64>()?
            .iter()
            .map(|v| v.to_bits())
            .collect();
        Self::split(&bits, link_count, reach_links)
    }

    /// Child keys from host bits in the layout of [`TrialKeys::read`].
    fn split(
        bits: &[u64],
        link_count: usize,
        reach_links: &[&[usize]],
    ) -> Result<Self, SurfaceChannelError> {
        let short = || SurfaceChannelError::Numerical("trial key read lacks a link volume".into());
        let to_channel = bits.get(..link_count).ok_or_else(short)?;
        let surface = bits.get(link_count..).ok_or_else(short)?;
        let to_surface = surface.get(..link_count).ok_or_else(short)?;
        let linked = reach_links
            .iter()
            .map(|links| {
                let mut key = Vec::with_capacity(2 * links.len());
                for &link in *links {
                    key.push(*to_channel.get(link).ok_or_else(short)?);
                    key.push(*to_surface.get(link).ok_or_else(short)?);
                }
                Ok(TrialKey(key))
            })
            .collect::<Result<_, SurfaceChannelError>>()?;
        Ok(Self {
            surface: TrialKey(surface.to_vec()),
            linked,
        })
    }
}

/// The most recent trials of one child, oldest first.
struct Kept<T>(VecDeque<(TrialKey, T)>);

impl<T: Clone> Kept<T> {
    fn new() -> Self {
        Self(VecDeque::with_capacity(CAPACITY))
    }

    /// The result kept under `key`, moved to the most recent place.
    fn find(&mut self, key: &TrialKey) -> Option<T> {
        let position = self.0.iter().position(|(kept, _)| kept == key)?;
        let entry = self.0.remove(position)?;
        let value = entry.1.clone();
        self.0.push_back(entry);
        Some(value)
    }

    /// Keeps `value` under `key` in place of the oldest entry.
    fn keep(&mut self, key: TrialKey, value: T) {
        if self.0.len() == CAPACITY {
            self.0.pop_front();
        }
        self.0.push_back((key, value));
    }
}

/// Kept surface results `S` and linked reach results `C` of one attempt.
pub(super) struct TrialMemo<S, C> {
    surface: RefCell<Kept<S>>,
    linked: RefCell<Vec<Kept<C>>>,
}

impl<S: Clone, C: Clone> TrialMemo<S, C> {
    /// An empty memo for `linked` reaches.
    pub fn new(linked: usize) -> Self {
        Self {
            surface: RefCell::new(Kept::new()),
            linked: RefCell::new((0..linked).map(|_| Kept::new()).collect()),
        }
    }

    /// The kept surface result of `key`, or a fresh `trial` kept under it.
    pub fn surface<E>(&self, key: &TrialKey, trial: impl FnOnce() -> Result<S, E>) -> Result<S, E> {
        if let Some(found) = self.surface.borrow_mut().find(key) {
            return Ok(found);
        }
        let value = trial()?;
        self.surface.borrow_mut().keep(key.clone(), value.clone());
        Ok(value)
    }

    /// The kept result of linked reach `index` under `key`, or a fresh `trial` kept under it.
    pub fn reach<E>(
        &self,
        index: usize,
        key: &TrialKey,
        trial: impl FnOnce() -> Result<C, E>,
    ) -> Result<C, E> {
        if let Some(found) = self
            .linked
            .borrow_mut()
            .get_mut(index)
            .and_then(|kept| kept.find(key))
        {
            return Ok(found);
        }
        let value = trial()?;
        if let Some(kept) = self.linked.borrow_mut().get_mut(index) {
            kept.keep(key.clone(), value.clone());
        }
        Ok(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use numr::runtime::cpu::{CpuDevice, CpuRuntime};
    use std::cell::Cell;

    type Result<T = ()> = std::result::Result<T, Box<dyn std::error::Error>>;

    fn key(values: &[f64]) -> TrialKey {
        TrialKey(values.iter().map(|v| v.to_bits()).collect())
    }

    /// Runs `memo.reach(0, ..)` and counts the trials it calculates.
    fn reach(memo: &TrialMemo<u32, u32>, at: &TrialKey, value: u32, runs: &Cell<u32>) -> u32 {
        let fresh = memo.reach(0, at, || {
            runs.set(runs.get() + 1);
            Ok::<_, ()>(value)
        });
        fresh.unwrap_or(u32::MAX)
    }

    #[test]
    fn a_repeated_key_reuses_its_trial() {
        let memo = TrialMemo::<u32, u32>::new(1);
        let runs = Cell::new(0);
        assert_eq!(reach(&memo, &key(&[0.25, 0.0]), 7, &runs), 7);
        assert_eq!(reach(&memo, &key(&[0.25, 0.0]), 8, &runs), 7);
        assert_eq!(runs.get(), 1);
    }

    #[test]
    fn a_one_ulp_change_misses() {
        let memo = TrialMemo::<u32, u32>::new(1);
        let runs = Cell::new(0);
        reach(&memo, &key(&[0.25, 0.0]), 7, &runs);
        let next = f64::from_bits(0.25f64.to_bits() + 1);
        assert_eq!(reach(&memo, &key(&[next, 0.0]), 8, &runs), 8);
        // A signed zero is a different operand bit pattern.
        assert_eq!(reach(&memo, &key(&[0.25, -0.0]), 9, &runs), 9);
        assert_eq!(runs.get(), 3);
    }

    #[test]
    fn the_least_recent_trial_leaves_first() {
        let memo = TrialMemo::<u32, u32>::new(1);
        let runs = Cell::new(0);
        let [a, b, c] = [key(&[1.0]), key(&[2.0]), key(&[3.0])];
        reach(&memo, &a, 1, &runs);
        reach(&memo, &b, 2, &runs);
        // A hit makes `a` the most recent, so `c` replaces `b`.
        assert_eq!(reach(&memo, &a, 0, &runs), 1);
        reach(&memo, &c, 3, &runs);
        assert_eq!(reach(&memo, &a, 0, &runs), 1);
        assert_eq!(reach(&memo, &b, 4, &runs), 4);
        assert_eq!(runs.get(), 4);
    }

    #[test]
    fn a_rejected_trial_is_not_kept() {
        let memo = TrialMemo::<u32, u32>::new(1);
        let at = key(&[1.0]);
        assert_eq!(
            memo.surface(&at, || Err::<u32, _>("shorter")),
            Err("shorter")
        );
        assert_eq!(memo.surface(&at, || Ok::<_, &str>(5)), Ok(5));
        assert_eq!(memo.surface(&at, || Ok::<_, &str>(6)), Ok(5));
    }

    #[test]
    fn reach_keys_hold_only_their_own_links() -> Result {
        // Links 0 and 2 feed reach 0. Link 1 feeds reach 1. Two surface lane operands follow.
        let values = [0.5, 0.0, 0.25, 0.0, 0.125, 0.0, 3.0, 4.0];
        let packed = Tensor::<CpuRuntime>::from_slice(&values, &[values.len()], &CpuDevice::new())?;
        let links: [&[usize]; 2] = [&[0, 2], &[1]];
        let keys = TrialKeys::read(&packed, 3, &links)?;
        assert_eq!(keys.linked[0], key(&[0.5, 0.0, 0.25, 0.0]));
        assert_eq!(keys.linked[1], key(&[0.0, 0.125]));
        assert_eq!(keys.surface, key(&[0.0, 0.125, 0.0, 3.0, 4.0]));
        // A change on a link of reach 1 leaves the key of reach 0 unchanged.
        let mut changed = values;
        changed[4] = f64::from_bits(0.125f64.to_bits() + 1);
        let packed =
            Tensor::<CpuRuntime>::from_slice(&changed, &[values.len()], &CpuDevice::new())?;
        let other = TrialKeys::read(&packed, 3, &links)?;
        assert_eq!(other.linked[0], keys.linked[0]);
        assert_ne!(other.linked[1], keys.linked[1]);
        assert_ne!(other.surface, keys.surface);
        Ok(())
    }

    #[test]
    fn a_short_read_is_an_error() {
        let links: [&[usize]; 1] = [&[3]];
        assert!(TrialKeys::split(&[0; 6], 3, &links).is_err());
        assert!(TrialKeys::split(&[0; 2], 3, &[]).is_err());
    }
}
