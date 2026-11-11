use super::context_types::{CutAggregation, Topology};
use super::{ChannelPoolEndpoint, ChannelPoolError, ChannelPoolSetup};
use crate::channel::ChannelGeometry;
use crate::numerics::water_transfers::TransferTree;
use crate::storage::LevelPoolCut;
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, IndexingOps, ReduceOps, ScalarOps, ShapeOps, TwoSumOps,
    UnaryOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;
use std::collections::{BTreeMap, BTreeSet};

impl<R: Runtime<DType = DType>> Topology<R>
where
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + IndexingOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + ShapeOps<R>
        + TwoSumOps<R>
        + UnaryOps<R>,
{
    pub(super) fn new(
        client: &R::Client,
        setup: &ChannelPoolSetup,
    ) -> Result<Self, ChannelPoolError> {
        text(&setup.vertical_datum, "vertical_datum")?;
        text(&setup.nonoverlap_assertion, "nonoverlap_assertion")?;
        if setup.reaches.is_empty() || setup.pools.is_empty() || setup.connections.is_empty() {
            return Err(ChannelPoolError::input(
                "empty topology: supply reaches, pools, and connections",
            ));
        }
        let mut reach_by_id = BTreeMap::new();
        let gravity = setup.reaches[0].definition.config.gravity.to_bits();
        for (index, reach) in setup.reaches.iter().enumerate() {
            text(&reach.id, "reach_id")?;
            if reach_by_id.insert(reach.id.clone(), index).is_some() {
                return Err(ChannelPoolError::input(format!(
                    "duplicate reach_id={:?}: supply unique reach identities",
                    reach.id
                )));
            }
            if reach.definition.vertical_datum != setup.vertical_datum
                || reach.definition.config.gravity.to_bits() != gravity
            {
                return Err(ChannelPoolError::input(format!(
                    "reach_id={:?}: supply the common vertical datum and exact gravity",
                    reach.id
                )));
            }
        }
        let mut pool_by_id = BTreeMap::new();
        let mut supports = BTreeSet::new();
        for (index, pool) in setup.pools.iter().enumerate() {
            if pool_by_id.insert(pool.id.as_str(), index).is_some()
                || !supports.insert(pool.storage_support_identity.as_str())
            {
                return Err(ChannelPoolError::input(
                    "duplicate pool identity or storage support: supply distinct represented pools",
                ));
            }
            if pool.curve.vertical_reference() != setup.vertical_datum {
                return Err(ChannelPoolError::input(format!(
                    "pool_id={:?}: supply the common vertical datum",
                    pool.id
                )));
            }
        }
        let mut endpoint_connections = vec![[None, None]; setup.reaches.len()];
        let mut ids = BTreeSet::new();
        let mut endpoint_indices = Vec::new();
        let mut pool_indices = Vec::new();
        let mut beds = Vec::new();
        let mut losses = Vec::new();
        let mut pool_targets = Vec::new();
        let mut connections_by_pool = vec![Vec::new(); setup.pools.len()];
        for (index, connection) in setup.connections.iter().enumerate() {
            text(&connection.id, "connection_id")?;
            text(
                &connection.physical_connection_assertion,
                "physical_connection_assertion",
            )?;
            text(
                &connection.quiescent_reservoir_assertion,
                "quiescent_reservoir_assertion",
            )?;
            if !ids.insert(connection.id.as_str())
                || !connection.entrance_loss_coefficient.is_finite()
                || connection.entrance_loss_coefficient < 0.0
            {
                return Err(ChannelPoolError::input(format!(
                    "connection_id={:?}, K={}: supply unique identities and finite nonnegative entrance losses",
                    connection.id, connection.entrance_loss_coefficient
                )));
            }
            let reach = *reach_by_id.get(&connection.reach_id).ok_or_else(|| {
                ChannelPoolError::input(format!(
                    "reach_id={:?}: connect a declared reach",
                    connection.reach_id
                ))
            })?;
            let pool = *pool_by_id.get(connection.pool_id.as_str()).ok_or_else(|| {
                ChannelPoolError::input(format!(
                    "pool_id={:?}: connect a declared pool",
                    connection.pool_id
                ))
            })?;
            let endpoint = match connection.endpoint {
                ChannelPoolEndpoint::Left => 0,
                ChannelPoolEndpoint::Right => 1,
            };
            if endpoint_connections[reach][endpoint]
                .replace(index)
                .is_some()
            {
                return Err(ChannelPoolError::input(format!(
                    "connection_id={:?}: assign each reach endpoint exactly one owner",
                    connection.id
                )));
            }
            endpoint_indices.push(
                reach
                    .checked_mul(2)
                    .and_then(|x| x.checked_add(endpoint))
                    .ok_or_else(|| ChannelPoolError::input("endpoint index overflows"))?,
            );
            pool_indices.push(pool);
            beds.push(endpoint_bed(
                &setup.reaches[reach].definition.geometry,
                endpoint,
            )?);
            losses.push(connection.entrance_loss_coefficient);
            pool_targets.push((pool, index, connection.id.as_str()));
            connections_by_pool[pool].push(index);
        }
        let external_endpoints = endpoint_connections
            .iter()
            .enumerate()
            .flat_map(|(reach, endpoints)| {
                endpoints
                    .iter()
                    .enumerate()
                    .filter_map(move |(side, connection)| {
                        connection.is_none().then_some((reach, side))
                    })
            })
            .collect::<Vec<_>>();
        let external_indices = external_endpoints
            .iter()
            .map(|&(reach, side)| reach * 2 + side)
            .collect::<Vec<_>>();
        let mut cuts = Vec::new();
        let mut connection_cuts = vec![0usize; setup.connections.len()];
        let mut blocks = Vec::new();
        let mut covers = Vec::new();
        let mut block_count = 0usize;
        for (pool, mut connections) in connections_by_pool.into_iter().enumerate() {
            connections.sort_by(|&a, &b| {
                beds[a]
                    .total_cmp(&beds[b])
                    .then_with(|| setup.connections[a].id.cmp(&setup.connections[b].id))
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
                        .ok_or_else(|| ChannelPoolError::input("cut block count overflows"))?;
                    block_ids.insert((start, width), group);
                    for &connection in &connections[start..start + width] {
                        blocks.push((group, connection, setup.connections[connection].id.clone()));
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
                        ChannelPoolError::input("cut cover lacks its cached dyadic block")
                    })?;
                    covers.push((cut, block, format!("{cursor:020}")));
                    cursor += size;
                }
                let mut next = first + 1;
                while next < connections.len() && beds[connections[next]] == elevation {
                    next += 1;
                }
                for &connection in &connections[first..next] {
                    connection_cuts[connection] = cut;
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
        let cut_aggregation = CutAggregation {
            blocks: TransferTree::new_grouped(
                client,
                block_count,
                setup.connections.len(),
                &borrowed_blocks,
            )?,
            covers: TransferTree::new_grouped(client, cuts.len(), block_count, &borrowed_covers)?,
        };
        let values = |v: &[f64]| -> Result<Tensor<R>, ChannelPoolError> {
            Ok(Tensor::from_slice(v, &[v.len()], client.device())?)
        };
        Ok(Self {
            reach_by_id,
            endpoint_connections,
            external_endpoints,
            connection_endpoint_indices: indices(client, &endpoint_indices)?,
            external_endpoint_indices: indices(client, &external_indices)?,
            connection_pool_indices: indices(client, &pool_indices)?,
            connection_cut_indices: indices(client, &connection_cuts)?,
            entrance_loss_coefficients: values(&losses)?,
            pool_origins_m: values(
                &setup
                    .pools
                    .iter()
                    .map(|p| p.curve.origin())
                    .collect::<Vec<_>>(),
            )?,
            cuts,
            cut_aggregation,
            pool_transfers: TransferTree::new_grouped(
                client,
                setup.pools.len(),
                setup.connections.len(),
                &pool_targets,
            )?,
        })
    }
}
fn endpoint_bed(geometry: &ChannelGeometry, endpoint: usize) -> Result<f64, ChannelPoolError> {
    match geometry {
        ChannelGeometry::Rectangular { cells } => {
            let cell = if endpoint == 0 {
                cells.first()
            } else {
                cells.last()
            };
            cell.map(|c| c.bed_elevation_m)
                .ok_or_else(|| ChannelPoolError::input("empty channel: supply endpoint geometry"))
        }
        ChannelGeometry::PrismaticTriangular {
            bed_elevation_m, ..
        } => Ok(*bed_elevation_m),
    }
}
fn indices<R: Runtime<DType = DType>>(
    client: &R::Client,
    values: &[usize],
) -> Result<Tensor<R>, ChannelPoolError> {
    let values = values
        .iter()
        .map(|&v| {
            i64::try_from(v)
                .map_err(|_| ChannelPoolError::input("topology index exceeds runtime support"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Tensor::from_slice(
        &values,
        &[values.len()],
        client.device(),
    )?)
}
pub(super) fn text(value: &str, field: &str) -> Result<(), ChannelPoolError> {
    if value.trim().is_empty() || value.chars().any(char::is_control) {
        return Err(ChannelPoolError::input(format!(
            "{field}={value:?}: supply nonempty evidence without control characters"
        )));
    }
    Ok(())
}
