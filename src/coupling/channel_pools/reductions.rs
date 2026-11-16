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
use numr::runtime::Runtime;
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
                ChannelGeometry::PrismaticSurveyed { .. } => {
                    return Err(ChannelPoolError::input(
                        "surveyed coordinator attachment is unsupported: use standalone reach advancement",
                    ));
                }
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
                .map_err(ChannelPoolError::from)
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

    pub(super) fn inventory(
        &self,
        client: &R::Client,
        reaches: &[ChannelReach<R>],
        pools: &LevelPoolBatch<R>,
    ) -> Result<(Tensor<R>, Tensor<R>), ChannelPoolError> {
        let mut values = Vec::with_capacity(reaches.len() + 1);
        for reach in reaches {
            values.push(reach.coupling_accounting().cell_volume_m3);
        }
        values.push(pools.coupling_native_accounting().volume_m3);
        let packed = client.cat(&values, 0)?;
        reduce(client, &self.storage, &packed)
    }
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
    let aggregate = tree.aggregate(client, values, AggregationKind::Momentum)?;
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
    Ok(upward_add(client, &value, &bound)?)
}
