use super::inflows::InflowContext;
use super::topology::TransferTree;
use super::{CoupledLevelPoolError, CoupledLevelPoolSetup};
use crate::structures::{DirectionPolicy, StructureLaw};
use hydr_structures::internal::CouplingStructures;
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, CorrectlyRoundedOps, IndexingOps, ReduceOps, ScalarOps,
    ShapeOps, TwoSumOps, UnaryOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug)]
pub(super) struct Context<R: Runtime<DType = DType>> {
    pub client: R::Client,
    pub inflows: InflowContext<R>,
    pub setup: CoupledLevelPoolSetup,
    pub structures: CouplingStructures<R>,
    pub a: Tensor<R>,
    pub b: Tensor<R>,
    pub bed_a: Tensor<R>,
    pub bed_b: Tensor<R>,
    pub cutoff_a: Tensor<R>,
    pub cutoff_b: Tensor<R>,
    pub origin: Tensor<R>,
    pub zero: Tensor<R>,
    pub one: Tensor<R>,
    pub tree: TransferTree<R>,
}

impl<R: Runtime<DType = DType>> Context<R>
where
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + CorrectlyRoundedOps<R>
        + IndexingOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + ShapeOps<R>
        + TwoSumOps<R>
        + UnaryOps<R>
        + numr::ops::TypeConversionOps<R>,
{
    pub fn new(
        client: R::Client,
        setup: CoupledLevelPoolSetup,
    ) -> Result<Self, CoupledLevelPoolError> {
        check_config(&setup)?;
        let n = setup.pools.len();
        let e = setup.structures.len();
        if n == 0 || e == 0 || setup.pool_support.len() != n || setup.connections.len() != e {
            return Err(CoupledLevelPoolError::input(
                "require nonempty pools/structures and complete ordered support/connection declarations",
            ));
        }
        let ids: BTreeMap<_, _> = setup
            .pools
            .iter()
            .enumerate()
            .map(|(i, p)| (p.id.as_str(), i))
            .collect();
        let support: BTreeSet<_> = setup
            .pools
            .iter()
            .map(|p| p.storage_support_identity.as_str())
            .collect();
        if ids.len() != n || support.len() != n {
            return Err(CoupledLevelPoolError::input(
                "pool and physical storage-support identities must be unique",
            ));
        }
        let mut origins = Vec::new();
        for (pool, support) in setup.pools.iter().zip(&setup.pool_support) {
            check_text(
                &support.zero_volume_limit_assertion,
                "zero-volume limit evidence",
            )?;
            let origin = pool
                .curve
                .knots()
                .first()
                .ok_or_else(|| {
                    CoupledLevelPoolError::input("pool curve lacks its zero-volume knot")
                })?
                .stage_m;
            if support.pool_id != pool.id
                || !support.zero_volume_stage_m.is_finite()
                || support.zero_volume_stage_m.to_bits() != origin.to_bits()
            {
                return Err(CoupledLevelPoolError::input(format!(
                    "pool={}: supply matching identity and original physical zero-volume limit",
                    pool.id
                )));
            }
            origins.push(origin);
        }
        let mut endpoints = Vec::new();
        let mut a = Vec::new();
        let mut b = Vec::new();
        let mut ba = Vec::new();
        let mut bb = Vec::new();
        let mut ca = Vec::new();
        let mut cb = Vec::new();
        for (element, connection) in setup.structures.iter().zip(&setup.connections) {
            if element.id != connection.structure_id {
                return Err(CoupledLevelPoolError::input(
                    "connections must match original structure order and identities",
                ));
            }
            let mut positions = [0usize; 2];
            let mut beds = [0.0; 2];
            for (j, (attachment, expected)) in [
                (&connection.endpoint_a, &element.endpoint_a),
                (&connection.endpoint_b, &element.endpoint_b),
            ]
            .into_iter()
            .enumerate()
            {
                check_text(
                    &attachment.physical_connection_assertion,
                    "physical connection evidence",
                )?;
                let index = *ids.get(attachment.pool_id.as_str()).ok_or_else(|| {
                    CoupledLevelPoolError::input(format!(
                        "unknown attached pool={}",
                        attachment.pool_id
                    ))
                })?;
                if &attachment.pool_id != expected
                    || element.datum != setup.pools[index].curve.vertical_reference()
                    || !attachment.bed_elevation_m.is_finite()
                    || attachment.bed_elevation_m < origins[index]
                {
                    return Err(CoupledLevelPoolError::input(format!(
                        "structure={}: require matching pool endpoint/datum and a finite local bed at or above its physical pool limit",
                        element.id
                    )));
                }
                positions[j] = index;
                beds[j] = attachment.bed_elevation_m;
            }
            if positions[0] == positions[1] {
                return Err(CoupledLevelPoolError::input(
                    "a structure must connect distinct physical pools",
                ));
            }
            let mut cuts = beds;
            match &element.law {
                StructureLaw::SuppressedRectangularWeir(p) => {
                    cuts[0] = cuts[0].max(p.crest_m());
                    cuts[1] = cuts[1].max(p.crest_m());
                }
                StructureLaw::HydrostaticRectangularGate(p) => {
                    cuts[0] = cuts[0].max(p.sill_m());
                    cuts[1] = cuts[1].max(p.sill_m());
                }
                StructureLaw::SubmergedOrifice(p) => {
                    cuts[0] = cuts[0].max(p.crown_m());
                    cuts[1] = cuts[1].max(p.crown_m());
                }
                StructureLaw::Pump(p) => {
                    let source = if element.direction == DirectionPolicy::BtoAOnly {
                        1
                    } else {
                        0
                    };
                    cuts[source] = cuts[source].max(p.minimum_suction_stage_m());
                }
            }
            a.push(index_value(positions[0])?);
            b.push(index_value(positions[1])?);
            ba.push(beds[0]);
            bb.push(beds[1]);
            ca.push(cuts[0]);
            cb.push(cuts[1]);
            endpoints.push((positions[0], positions[1]));
        }
        let tensor = |v: &[f64]| Tensor::<R>::from_slice(v, &[v.len()], client.device());
        let ai = Tensor::<R>::from_slice(&a, &[a.len()], client.device())?;
        let bi = Tensor::<R>::from_slice(&b, &[b.len()], client.device())?;
        let structure_ids = setup
            .structures
            .iter()
            .map(|s| s.id.clone())
            .collect::<Vec<_>>();
        let tree = TransferTree::new(&client, n, &endpoints, &structure_ids)?;
        let structures = CouplingStructures::new(
            client.clone(),
            setup.structures.clone(),
            setup.operations.clone(),
        )?;
        let inflows = InflowContext::<R>::new(&client, &setup.pools, &setup.external_inflows)?;
        Ok(Self {
            inflows,
            a: ai,
            b: bi,
            bed_a: tensor(&ba)?,
            bed_b: tensor(&bb)?,
            cutoff_a: tensor(&ca)?,
            cutoff_b: tensor(&cb)?,
            origin: tensor(&origins)?,
            zero: Tensor::zeros(&[], DType::F64, client.device())?,
            one: Tensor::ones(&[], DType::F64, client.device())?,
            tree,
            structures,
            setup,
            client,
        })
    }
    pub fn pool_count(&self) -> usize {
        self.setup.pools.len()
    }
    pub fn edge_count(&self) -> usize {
        self.setup.structures.len()
    }
}

fn check_config(setup: &CoupledLevelPoolSetup) -> Result<(), CoupledLevelPoolError> {
    let c = &setup.config;
    c.storage.check()?;
    if !c.head_absolute_m.is_finite()
        || c.head_absolute_m <= 0.0
        || !c.transfer_absolute_m3.is_finite()
        || c.transfer_absolute_m3 <= 0.0
        || !c.mass_absolute_m3.is_finite()
        || c.mass_absolute_m3 < 0.0
        || !c.mass_relative.is_finite()
        || c.mass_relative < 0.0
        || (c.mass_absolute_m3 == 0.0 && c.mass_relative == 0.0)
        || !c.minimum_step_s.is_finite()
        || c.minimum_step_s <= 0.0
        || !c.maximum_step_s.is_finite()
        || c.maximum_step_s < c.minimum_step_s
        || c.iteration_limit == 0
    {
        return Err(CoupledLevelPoolError::input(
            "require positive finite head/volume tolerances and step limits, a positive mass budget, and a nonzero iteration limit",
        ));
    }
    Ok(())
}
fn check_text(value: &str, field: &str) -> Result<(), CoupledLevelPoolError> {
    if value.trim().is_empty() || value.chars().any(char::is_control) {
        return Err(CoupledLevelPoolError::input(format!(
            "{field}={value:?}: supply explicit nonempty physical evidence"
        )));
    }
    Ok(())
}
fn index_value(value: usize) -> Result<i64, CoupledLevelPoolError> {
    i64::try_from(value)
        .map_err(|_| CoupledLevelPoolError::input("pool count exceeds runtime indexing support"))
}
