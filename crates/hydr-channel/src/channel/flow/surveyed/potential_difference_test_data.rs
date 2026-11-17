#[path = "potential_difference_case_data.rs"]
mod cases;
pub(crate) use cases::*;
#[path = "potential_difference_derivative_data.rs"]
mod derivatives;
pub(crate) use derivatives::*;
#[path = "potential_difference_panel_data_0.rs"]
mod panel_0;
#[path = "potential_difference_panel_data_1.rs"]
mod panel_1;
#[path = "potential_difference_panel_data_2.rs"]
mod panel_2;
#[path = "potential_difference_panel_data_3.rs"]
mod panel_3;
#[path = "potential_difference_panel_data_4.rs"]
mod panel_4;
#[path = "potential_difference_panel_data_5.rs"]
mod panel_5;
#[path = "potential_difference_panel_data_6.rs"]
mod panel_6;
pub(crate) const PANELS: &[&DerivativePanel] = &[
    &panel_0::PANEL,
    &panel_1::PANEL,
    &panel_2::PANEL,
    &panel_3::PANEL,
    &panel_4::PANEL,
    &panel_5::PANEL,
    &panel_6::PANEL,
];
