//! Attachment cuts of every pool and the dyadic block covers that sum exports above each cut.
//!
//! Each distinct attachment bed of a pool is one cut. Every attachment at or above a cut exports
//! through it. Per pool, attachments sort by bed, then identity. Blocks are aligned dyadic runs
//! of that order. Each cut sums the fewest aligned blocks that cover its attachments.

use super::{LinkError, LinkOps};
use crate::numerics::water_transfers::TransferTree;
use hydr_storage::internal::LevelPoolCut;
use numr::dtype::DType;
use numr::runtime::Runtime;
use std::collections::BTreeMap;

/// One pool attachment: its pool, its bed, and its identity.
pub(crate) struct CutAttachment<'a> {
    pub pool: usize,
    pub bed_m: f64,
    pub id: &'a str,
}

/// Cuts of every pool with the trees that sum attachment exports above each cut.
pub(crate) struct CutCover<R: Runtime<DType = DType>> {
    /// Cuts in pool order, then ascending bed.
    pub cuts: Vec<LevelPoolCut>,
    /// Cut at each attachment's bed, in attachment order.
    pub attachment_cuts: Vec<usize>,
    /// Sums attachment exports per aligned block.
    pub blocks: TransferTree<R>,
    /// Sums blocks per cut.
    pub covers: TransferTree<R>,
}

/// Builds the cuts and covers of `attachments` over `pool_count` pools.
pub(crate) fn cut_cover<R: Runtime<DType = DType>>(
    client: &R::Client,
    pool_count: usize,
    attachments: &[CutAttachment<'_>],
) -> Result<CutCover<R>, LinkError>
where
    R::Client: LinkOps<R>,
{
    let mut by_pool = vec![Vec::new(); pool_count];
    for (index, attachment) in attachments.iter().enumerate() {
        by_pool
            .get_mut(attachment.pool)
            .ok_or_else(|| LinkError::Input("cut attachment names no declared pool".into()))?
            .push(index);
    }
    let beds: Vec<f64> = attachments.iter().map(|a| a.bed_m).collect();
    let mut cuts = Vec::new();
    let mut attachment_cuts = vec![0usize; attachments.len()];
    let mut blocks = Vec::new();
    let mut covers = Vec::new();
    let mut block_count = 0usize;
    for (pool, mut connections) in by_pool.into_iter().enumerate() {
        connections.sort_by(|&a, &b| {
            beds[a]
                .total_cmp(&beds[b])
                .then_with(|| attachments[a].id.cmp(attachments[b].id))
        });
        let mut block_ids = BTreeMap::new();
        let mut width = 1usize;
        while width <= connections.len() {
            for start in (0..connections.len()).step_by(width) {
                if start + width > connections.len() {
                    continue;
                }
                let group = block_count;
                block_count = block_count
                    .checked_add(1)
                    .ok_or_else(|| LinkError::Input("cut block count overflows".into()))?;
                block_ids.insert((start, width), group);
                for &connection in &connections[start..start + width] {
                    blocks.push((group, connection, attachments[connection].id.to_string()));
                }
            }
            match width.checked_mul(2) {
                Some(next) => width = next,
                None => break,
            }
        }
        let mut first = 0usize;
        while first < connections.len() {
            let elevation = beds[connections[first]];
            let cut = cuts.len();
            cuts.push(LevelPoolCut {
                pool_index: pool,
                elevation_m: elevation,
            });
            let mut cursor = first;
            while cursor < connections.len() {
                let mut size = 1usize;
                while let Some(next) = size.checked_mul(2) {
                    if !cursor.is_multiple_of(next) || next > connections.len() - cursor {
                        break;
                    }
                    size = next;
                }
                let block = *block_ids.get(&(cursor, size)).ok_or_else(|| {
                    LinkError::Input("cut cover lacks its cached dyadic block".into())
                })?;
                covers.push((cut, block, format!("{cursor:020}")));
                cursor += size;
            }
            let mut next = first + 1;
            while next < connections.len() && beds[connections[next]] == elevation {
                next += 1;
            }
            for &connection in &connections[first..next] {
                attachment_cuts[connection] = cut;
            }
            first = next;
        }
    }
    let borrowed_blocks = blocks
        .iter()
        .map(|(a, b, id)| (*a, *b, id.as_str()))
        .collect::<Vec<_>>();
    let borrowed_covers = covers
        .iter()
        .map(|(a, b, id)| (*a, *b, id.as_str()))
        .collect::<Vec<_>>();
    Ok(CutCover {
        blocks: TransferTree::new_grouped(
            client,
            block_count,
            attachments.len(),
            &borrowed_blocks,
        )?,
        covers: TransferTree::new_grouped(client, cuts.len(), block_count, &borrowed_covers)?,
        cuts,
        attachment_cuts,
    })
}
