//! Runoff/pool fixtures under the sibling names their own `super::` paths expect.

#[path = "../runoff_pools/fixture.rs"]
pub mod fixture;

#[path = "../coupled_level_pools/laws.rs"]
pub mod laws;

use crate::coupled_pool_fixture as pool_fixture;
