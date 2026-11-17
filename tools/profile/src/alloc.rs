//! Global allocator that counts allocation calls and requested bytes.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicU64, Ordering};

/// Forwards to [`System`] and counts every allocation and reallocation.
pub struct Counting;

static CALLS: AtomicU64 = AtomicU64::new(0);
static BYTES: AtomicU64 = AtomicU64::new(0);

/// Allocation calls and requested bytes since process start.
#[derive(Clone, Copy, Debug, Default)]
pub struct Totals {
    pub calls: u64,
    pub bytes: u64,
}

impl Totals {
    /// Current totals.
    pub fn now() -> Self {
        Self {
            calls: CALLS.load(Ordering::Relaxed),
            bytes: BYTES.load(Ordering::Relaxed),
        }
    }

    /// Totals accumulated since `earlier`.
    pub fn since(self, earlier: Self) -> Self {
        Self {
            calls: self.calls - earlier.calls,
            bytes: self.bytes - earlier.bytes,
        }
    }
}

fn record(size: usize) {
    CALLS.fetch_add(1, Ordering::Relaxed);
    BYTES.fetch_add(size as u64, Ordering::Relaxed);
}

// SAFETY: every method forwards its exact arguments to `System`, which upholds the contract.
#[allow(unsafe_code)]
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        record(layout.size());
        // SAFETY: the caller's layout contract passes through unchanged.
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        record(layout.size());
        // SAFETY: the caller's layout contract passes through unchanged.
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: `ptr` came from `System` with this `layout`.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        record(new_size);
        // SAFETY: `ptr` came from `System` with this `layout`.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}
