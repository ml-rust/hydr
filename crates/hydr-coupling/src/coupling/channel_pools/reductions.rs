use super::{
    ChannelPoolError, ChannelPoolSetup, accounting_math::nonnegative, context_types::ReductionPlans,
};
use crate::channel::{ChannelGeometry, ChannelLateralDirection, ChannelReach};
use crate::numerics::water_transfers::{AggregationKind, TransferTree, upward_add};
use crate::storage::LevelPoolBatch;
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, IndexingOps, ReduceOps, ScalarOps, ShapeOps, TwoSumOps,
    UnaryOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

impl<R: Runtime<DType = DType>> ReductionPlans<R>
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
        external_count: usize,
    ) -> Result<Self, ChannelPoolError> {
        let mut storage_keys = Vec::new();
        let mut lateral_keys = Vec::new();
        let mut face_keys = Vec::new();
        let mut assembly_keys = Vec::new();
        for reach in &setup.reaches {
            let cells = match &reach.definition.geometry {
                ChannelGeometry::Rectangular { cells } => cells
                    .iter()
                    .map(|cell| cell.id.as_str())
                    .collect::<Vec<_>>(),
                ChannelGeometry::PrismaticTriangular { cells, .. } => cells
                    .iter()
                    .map(|cell| cell.id.as_str())
                    .collect::<Vec<_>>(),
                ChannelGeometry::PrismaticSurveyed { cells, .. } => cells
                    .iter()
                    .map(|cell| cell.id.as_str())
                    .collect::<Vec<_>>(),
            };
            for cell in &cells {
                storage_keys.push(format!("channel/{}/{cell}", reach.id));
                assembly_keys.push(format!("channel/{}/{cell}", reach.id));
            }
            for face in 0..=cells.len() {
                face_keys.push((
                    format!("{}/{face:020}", reach.id),
                    face > 0 && face < cells.len(),
                ));
            }
            for binding in &reach.definition.lateral_bindings {
                let group = usize::from(binding.direction == ChannelLateralDirection::Withdrawal);
                lateral_keys.push((group, format!("{}/{}", reach.id, binding.boundary_id)));
            }
        }
        storage_keys.extend(setup.pools.iter().map(|pool| format!("pool/{}", pool.id)));
        let tree = |groups, count, targets: &[(usize, usize, &str)]| {
            TransferTree::new_grouped(client, groups, count, targets)
                .map_err(ChannelPoolError::from_transfer)
        };
        let one = |keys: &[String]| {
            tree(
                1,
                keys.len(),
                &keys
                    .iter()
                    .enumerate()
                    .map(|(index, key)| (0, index, key.as_str()))
                    .collect::<Vec<_>>(),
            )
        };
        let external_keys = (0..external_count)
            .map(|index| format!("{index:020}"))
            .collect::<Vec<_>>();
        let pool_keys = setup
            .pools
            .iter()
            .map(|pool| pool.id.clone())
            .collect::<Vec<_>>();
        let mut physical_targets = Vec::new();
        for (index, (key, interior)) in face_keys.iter().enumerate() {
            physical_targets.push((0, index, key.as_str()));
            if *interior {
                physical_targets.push((0, index, key.as_str()));
            }
        }
        let faces = face_keys.len();
        physical_targets.extend(
            assembly_keys
                .iter()
                .enumerate()
                .map(|(index, key)| (1, faces + index, key.as_str())),
        );
        let cells = assembly_keys.len();
        physical_targets.extend(
            pool_keys
                .iter()
                .enumerate()
                .map(|(index, key)| (2, faces + cells + index, key.as_str())),
        );
        Ok(Self {
            storage: one(&storage_keys)?,
            lateral: tree(
                2,
                lateral_keys.len(),
                &lateral_keys
                    .iter()
                    .enumerate()
                    .map(|(index, (group, key))| (*group, index, key.as_str()))
                    .collect::<Vec<_>>(),
            )?,
            external_counters: one(&external_keys)?,
            pool_counters: one(&pool_keys)?,
            physical_charges: tree(3, faces + cells + pool_keys.len(), &physical_targets)?,
        })
    }

    /// Composite storage and its reduction bound.
    ///
    /// The storage tree reduces the high parts of every cell and pool volume. When a reach
    /// cell carries a nonzero low part, the same tree reduces the low parts. The bound then
    /// adds both reduction bounds and the rounding of their sum. All-zero low parts skip
    /// that reduction, so single-part storage keeps the high-part result.
    pub(super) fn inventory(
        &self,
        client: &R::Client,
        reaches: &[ChannelReach<R>],
        pools: &LevelPoolBatch<R>,
    ) -> Result<(Tensor<R>, Tensor<R>), ChannelPoolError> {
        let mut values = Vec::with_capacity(reaches.len() + 1);
        let mut lows = Vec::with_capacity(reaches.len() + 1);
        for reach in reaches {
            let accounting = reach.coupling_accounting();
            values.push(accounting.cell_volume_m3);
            lows.push(accounting.cell_volume_low_m3);
        }
        let pool_volume = pools.coupling_native_accounting().volume_m3;
        let low_parts = nonzero_lows(client, &lows, pool_volume)?;
        values.push(pool_volume);
        let packed = client.cat(&values, 0)?;
        let (high, high_bound) = reduce(client, &self.storage, &packed)?;
        let Some(low_parts) = low_parts else {
            return Ok((high, high_bound));
        };
        let low = self
            .storage
            .aggregate(client, &low_parts, AggregationKind::Momentum)
            .map_err(ChannelPoolError::from_transfer)?;
        let (total, rounding) = client.two_sum(&high, &low.value)?;
        let bound = upward_add(
            client,
            &upward_add(client, &high_bound, &low.bound)
                .map_err(ChannelPoolError::from_transfer)?,
            &client.abs(&rounding)?,
        )
        .map_err(ChannelPoolError::from_transfer)?;
        Ok((total, bound))
    }
}

/// Signed low parts in storage-tree order, or `None` when every cell low part is zero.
///
/// Pools hold one F64 volume, so their low parts are zero.
fn nonzero_lows<R: Runtime<DType = DType>>(
    client: &R::Client,
    lows: &[&Tensor<R>],
    pool_volume: &Tensor<R>,
) -> Result<Option<Tensor<R>>, ChannelPoolError>
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
    if lows.is_empty() {
        return Ok(None);
    }
    let cells = client.cat(lows, 0)?;
    if cells.numel() == 0 {
        return Ok(None);
    }
    let zero = Tensor::<R>::zeros(&[1], DType::F64, client.device())?;
    if client
        .max(&client.ne(&cells, &zero)?, &[0], false)?
        .item::<f64>()?
        == 0.0
    {
        return Ok(None);
    }
    let pool_lows = Tensor::<R>::zeros(pool_volume.shape(), DType::F64, client.device())?;
    Ok(Some(client.cat(&[&cells, &pool_lows], 0)?))
}

pub(super) fn reduce<R: Runtime<DType = DType>>(
    client: &R::Client,
    tree: &TransferTree<R>,
    values: &Tensor<R>,
) -> Result<(Tensor<R>, Tensor<R>), ChannelPoolError>
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
    nonnegative(client, values, "nonnegative accounting reduction input")?;
    let aggregate = tree
        .aggregate(client, values, AggregationKind::Momentum)
        .map_err(ChannelPoolError::from_transfer)?;
    Ok((aggregate.value, aggregate.bound))
}

pub(super) fn upper_sum<R: Runtime<DType = DType>>(
    client: &R::Client,
    tree: &TransferTree<R>,
    values: &Tensor<R>,
) -> Result<Tensor<R>, ChannelPoolError>
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
    let (value, bound) = reduce(client, tree, values)?;
    upward_add(client, &value, &bound).map_err(ChannelPoolError::from_transfer)
}

#[cfg(test)]
mod tests {
    use super::super::batch::tests::{
        Batch, MASS_ABSOLUTE_M3, MASS_RELATIVE, TestResult, client, exterior, forcing, single,
    };
    use crate::coupling::ChannelPoolEndpoint;

    /// A three-band reach fed by a higher pool keeps two-part storage in the mass balance.
    ///
    /// Four 32 m cells start at rest at depth 1.5 m (5.125 m² each). The right end
    /// connects to a pool of plan area 16 m² at stage 2.5 m with `k = 0.5`. The left end
    /// holds the rest state.
    ///
    /// Frozen criteria, every accepted interval up to 2 s:
    /// - at least one reach cell carries a nonzero volume low part
    /// - `|residual| + bound <= mass_absolute + mass_relative · max(|before|, |after|)`
    ///   for the composite storage
    /// - the interval ledger reports the inventory of the accepted states
    #[test]
    fn surveyed_low_parts_stay_within_the_mass_bound() -> TestResult {
        let (setup, initial) = single(
            4,
            32.0,
            32.0 * 5.125,
            ChannelPoolEndpoint::Right,
            16.0,
            2.5,
            0.5,
        )?;
        let mut current = Batch::new(client(), setup.clone(), initial)?;
        let mut intervals = 0;
        while current.time() < 2.0 {
            let start = current.time();
            let end = (start + 0.125).min(2.0);
            current = current
                .trial_to(end, &forcing(&setup, start, end, &exterior(1.5)))?
                .into_next();
            assert!(current.time() > start && current.time() <= end);
            let reach = current.reach("reach").ok_or("Missing reach")?;
            let lows = reach
                .coupling_accounting()
                .cell_volume_low_m3
                .try_to_vec::<f64>()?;
            assert!(lows.iter().any(|low| *low != 0.0), "{lows:?}");
            let (storage, _) = current
                .context
                .inventory(&current.accepted.reaches, &current.accepted.pools)?;
            let step = current.last_step()?.ok_or("Missing interval receipt")?;
            assert_eq!(
                step.storage_after_m3.to_bits(),
                storage.item::<f64>()?.to_bits()
            );
            let scale = step
                .storage_before_m3
                .abs()
                .max(step.storage_after_m3.abs());
            assert!(
                step.mass_residual_m3.abs() + step.aggregation_roundoff_bound_m3
                    <= MASS_ABSOLUTE_M3 + MASS_RELATIVE * scale,
                "{step:?}"
            );
            intervals += 1;
        }
        assert!(intervals >= 16);
        Ok(())
    }
}
