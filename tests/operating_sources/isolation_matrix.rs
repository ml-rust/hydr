//! Individual cases from the frozen isolation policy.

use super::{
    fixture::Result,
    isolation_matrix_fixture::{Mutation, Recipe},
    isolation_matrix_run::run,
};

macro_rules! frozen_case {
    ($name:ident, $recipe:ident, $mutation:ident) => {
        #[test]
        fn $name() -> Result {
            run(Recipe::$recipe, Mutation::$mutation)
        }
    };
}

frozen_case!(manual_zero_unknown_heads, ManualZero, None);
frozen_case!(manual_positive_minsub_unknown_heads, ManualMinsub, None);
frozen_case!(manual_positive_quarter_unknown_heads, ManualQuarter, None);
frozen_case!(prescribed_zero_knot_unknown_heads, PrescribedZero, None);
frozen_case!(
    prescribed_zero_plateau_unknown_heads,
    PrescribedPlateau,
    None
);
frozen_case!(
    prescribed_positive_minsub_unknown_heads,
    PrescribedMinsub,
    None
);
frozen_case!(
    actuated_zero_active_target_positive_unknown_heads,
    ActualActive,
    None
);
frozen_case!(
    actuated_zero_stalled_target_positive_unknown_heads,
    ActualStalled,
    None
);
frozen_case!(
    actuated_positive_target_zero_unknown_heads,
    ActualPositive,
    None
);
frozen_case!(
    actuated_closure_unresolved_unknown_heads,
    ActualUnresolved,
    None
);
frozen_case!(manual_zero_malformed_nonfinite_input, ManualZero, Nonfinite);
frozen_case!(manual_zero_malformed_datum, ManualZero, Datum);
frozen_case!(
    manual_zero_malformed_head_convention,
    ManualZero,
    Convention
);
frozen_case!(manual_zero_malformed_identity, ManualZero, Identity);
frozen_case!(
    prescribed_zero_knot_malformed_nonfinite_input,
    PrescribedZero,
    Nonfinite
);
frozen_case!(prescribed_zero_knot_malformed_datum, PrescribedZero, Datum);
frozen_case!(
    prescribed_zero_knot_malformed_head_convention,
    PrescribedZero,
    Convention
);
frozen_case!(
    prescribed_zero_knot_malformed_identity,
    PrescribedZero,
    Identity
);
frozen_case!(
    actuated_zero_active_target_positive_malformed_nonfinite_input,
    ActualActive,
    Nonfinite
);
frozen_case!(
    actuated_zero_active_target_positive_malformed_datum,
    ActualActive,
    Datum
);
frozen_case!(
    actuated_zero_active_target_positive_malformed_head_convention,
    ActualActive,
    Convention
);
frozen_case!(
    actuated_zero_active_target_positive_malformed_identity,
    ActualActive,
    Identity
);
frozen_case!(stopped_pump_unknown_heads, PumpStopped, None);
frozen_case!(running_pump_unknown_heads, PumpRunning, None);
frozen_case!(
    manual_positive_not_isolated_by_prescribed_padding,
    Padding,
    None
);
