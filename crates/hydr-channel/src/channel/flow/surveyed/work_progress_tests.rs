use super::*;

#[test]
fn resumed_root_work_counts_deltas_and_keeps_the_session_maximum() -> HydraulicResult<()> {
    let ledger = WorkLedger::new(&PrismaticRiemannConfig::default());
    ledger.begin_root()?;
    ledger.finish_root_progress(3, 3)?;
    ledger.begin_root()?;
    ledger.finish_root_progress(2, 5)?;
    ledger.begin_root()?;
    ledger.finish_root_progress(1, 6)?;
    ledger.charge(QueryWork::Residual, 17)?;
    assert_eq!(
        ledger.snapshot(),
        PrismaticRiemannWork {
            root_solves: 3,
            root_iterations: 6,
            maximum_root_iterations: 6,
            residual_points: 17,
            ..PrismaticRiemannWork::default()
        }
    );
    let before = ledger.snapshot();
    ledger.finish_root_progress(0, 6)?;
    assert_eq!(ledger.snapshot(), before);
    Ok(())
}

#[test]
fn independent_root_sessions_keep_separate_iteration_maxima() -> HydraulicResult<()> {
    let ledger = WorkLedger::new(&PrismaticRiemannConfig::default());
    ledger.finish_root_progress(5, 5)?;
    ledger.finish_root_progress(2, 2)?;
    assert_eq!(ledger.snapshot().root_iterations, 7);
    assert_eq!(ledger.snapshot().maximum_root_iterations, 5);
    Ok(())
}

#[test]
fn invalid_or_overflowing_root_progress_preserves_all_counters() {
    let ledger = WorkLedger::new(&PrismaticRiemannConfig::default());
    let before = PrismaticRiemannWork {
        root_solves: 2,
        root_iterations: 7,
        maximum_root_iterations: 5,
        residual_points: 13,
        ..PrismaticRiemannWork::default()
    };
    ledger.counts.set(before);
    assert!(ledger.finish_root_progress(4, 3).is_err());
    assert_eq!(ledger.snapshot(), before);
    let overflow = PrismaticRiemannWork {
        root_iterations: usize::MAX,
        ..before
    };
    ledger.counts.set(overflow);
    assert!(matches!(
        ledger.finish_root_progress(1, 1),
        Err(HydraulicError::Rejected {
            reason: PrismaticRiemannRejection::WorkLimit,
            ..
        })
    ));
    assert_eq!(ledger.snapshot(), overflow);
}

#[test]
fn legacy_root_completion_matches_one_progress_advance() -> HydraulicResult<()> {
    let config = PrismaticRiemannConfig::default();
    let legacy = WorkLedger::new(&config);
    let progress = WorkLedger::new(&config);
    legacy.finish_root(7)?;
    progress.finish_root_progress(7, 7)?;
    assert_eq!(legacy.snapshot(), progress.snapshot());
    Ok(())
}
