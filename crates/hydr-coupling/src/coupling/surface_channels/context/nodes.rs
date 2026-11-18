//! Resolved link endpoints and linked reaches of the coordinator topology.

use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// One resolved endpoint of a link.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(in super::super) enum NodeEnd {
    /// Surface lane.
    Surface(usize),
    /// Channel slot.
    Channel(usize),
    /// Pool in ascending identity.
    Pool(usize),
}

/// A reach that holds at least one link, with its slots and exchange bindings.
pub(in super::super) struct LinkedReach<R: Runtime<DType = DType>> {
    /// Setup reach index.
    pub reach: usize,
    /// Linked cells in ascending order. Their slots are consecutive.
    pub cells: Vec<usize>,
    /// `cells` as an I64 index tensor.
    pub cell_index: Tensor<R>,
    /// Positions of the links with a channel endpoint on this reach, in ascending link identity.
    pub links: Vec<usize>,
    /// Caller binding count. Exchange bindings follow, two per link.
    pub caller_bindings: usize,
}
