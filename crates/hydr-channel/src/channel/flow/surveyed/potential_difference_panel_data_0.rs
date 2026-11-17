use super::derivatives::{DerivativePanel, DerivativePoint};
pub(crate) const PANEL: DerivativePanel = DerivativePanel {
    id: "weak_panel",
    area: [0x4023f45a904b7a1f, 0x4024000000003a1f],
    squared_width: [0x404fee87d871072e, 0x4050000000004398],
    absolute_delta: [0x3f974adf68978400, 0x3f974adf68978400],
    points: &[
        DerivativePoint {
            area: [0x4024000000000000, 0x4024000000000000],
            squared_width: [0x4050000000000000, 0x4050000000000000],
            delta: [0xbf974adf68978400, 0xbf974adf68978400],
            slope: [0x4008000000000000, 0x4008000000000000],
            convex_margin: [0x4041000000000000, 0x4041000000000000],
            reference: [0xbd8768a88a1c19a2, 0xbd8768a88a1c19a1],
        },
        DerivativePoint {
            area: [0x4023fa2d4825da1f, 0x4023fa2d4825da1f],
            squared_width: [0x404ff743ec38c72e, 0x404ff743ec38c72f],
            delta: [0xbf974adf68978400, 0xbf974adf68978400],
            slope: [0x4008000000000000, 0x4008000000000000],
            convex_margin: [0x4040fba1f61c6397, 0x4040fba1f61c6398],
            reference: [0xbd87888633cf34c3, 0xbd87888633cf34c2],
        },
        DerivativePoint {
            area: [0x4023f45a904bb43e, 0x4023f45a904bb43e],
            squared_width: [0x404fee87d8718e5d, 0x404fee87d8718e5d],
            delta: [0xbf974adf68978400, 0xbf974adf68978400],
            slope: [0x4008000000000000, 0x4008000000000000],
            convex_margin: [0x4040f743ec38c72e, 0x4040f743ec38c72f],
            reference: [0xbd87a89875c97451, 0xbd87a89875c97450],
        },
    ],
};
