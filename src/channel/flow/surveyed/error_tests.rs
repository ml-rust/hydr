use super::error::{
    HydraulicError, PrismaticRiemannError, PrismaticRiemannRejection, checked, into_public,
};
use super::work::PrismaticRiemannWork;
use crate::channel::flow::ChannelError;
use numr::runtime::{
    RuntimeClient,
    cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig},
};
use numr::tensor::Tensor;

#[test]
fn solver_numerical_diagnostic_preserves_public_error_category() {
    let context = "root arithmetic rejected nonfinite residual bounds at stage=star";
    let error = into_public(
        HydraulicError::Numerical {
            context: context.into(),
        },
        PrismaticRiemannWork::default(),
    );
    assert_eq!(
        error,
        PrismaticRiemannError::Channel(ChannelError::Numerical {
            cell: None,
            context: context.into(),
        })
    );
}

#[test]
fn arithmetic_completion_precedes_retained_geometry_rejection()
-> Result<(), Box<dyn std::error::Error>> {
    let client =
        CpuClient::new(CpuDevice::new()).with_parallelism(ParallelismConfig::new(Some(1), None));
    let lower = Tensor::<CpuRuntime>::from_slice(&[1.0_f64], &[1], client.device())?;
    let upper = Tensor::<CpuRuntime>::from_slice(&[0.0_f64], &[1], client.device())?;
    let result = checked::<CpuRuntime, ()>(&client, "surveyed sticky arithmetic test", |a| {
        let _invalid = a.bounds(&lower, &upper)?;
        Err(HydraulicError::Rejected {
            reason: PrismaticRiemannRejection::GeometryUnresolved,
            query: Some(0),
            section_owner: Some(0),
            context: "depth support remains unresolved. Check profile conditioning".into(),
        })
    });
    let error = match result {
        Ok(()) => return Err("invalid bounds escaped arithmetic completion".into()),
        Err(error) => into_public(error, PrismaticRiemannWork::default()),
    };
    assert!(
        matches!(
            error,
            PrismaticRiemannError::Channel(ChannelError::Numerical { cell: None, .. })
        ),
        "{error}"
    );
    Ok(())
}
