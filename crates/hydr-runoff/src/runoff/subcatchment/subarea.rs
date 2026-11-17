//! Stable catchment-major subarea ordering.

/// Fixed physical subarea role. Ordering defines complete state and receipt arrays.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Subarea {
    /// Impervious support without depression storage.
    ImperviousZeroStorage,
    /// Impervious support with declared depression storage.
    ImperviousStorage,
    /// Pervious support with complete native soil memory.
    Pervious,
}

impl Subarea {
    /// Fixed catchment-major physical order.
    pub const ALL: [Self; 3] = [
        Self::ImperviousZeroStorage,
        Self::ImperviousStorage,
        Self::Pervious,
    ];
    /// Stable zero-based index within a catchment.
    pub const fn index(self) -> usize {
        match self {
            Self::ImperviousZeroStorage => 0,
            Self::ImperviousStorage => 1,
            Self::Pervious => 2,
        }
    }
    /// Recover a physical subarea from its stable index.
    pub const fn from_index(index: usize) -> Option<Self> {
        match index {
            0 => Some(Self::ImperviousZeroStorage),
            1 => Some(Self::ImperviousStorage),
            2 => Some(Self::Pervious),
            _ => None,
        }
    }
}
