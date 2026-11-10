# Connect level pools

Coupled level pools advance connected storage through declared hydraulic structures for Rust applications. Start with the physical requirements below.

Read [level pools](level-pool.md) for storage curves and [structure laws](structures.md) for supported flow relations.

## Check the physical model

- Storage: Each pool represents connected, approximately level water with its own stage-volume curve.
- Ownership: Pool volumes must describe separate physical storage. Do not count the same water in multiple pools.
- Units: Supply volumes in m³, elevations in metres, and time in seconds.
- Datum: Each structure and its connected pools must use the same vertical reference.
- Dry support: Declare the physical zero-volume stage limit and the bed at each connection.

Zero volume has no reported water elevation. Midpoint estimates use the declared zero-volume limit when water enters or leaves a dry pool. This limit must match the storage curve origin.

A connection's local bed can lie above that origin. Water below a raised connection cannot supply that outlet.

## Define connections and operations

1. Give each pool and structure a stable identity.
2. Bind each structure endpoint to its declared pool.
3. Supply evidence supporting the local bed and physical connection.
4. Supply one explicit static operation for each structure.
5. Choose timestep limits and separate head, transfer-volume, and water-balance tolerances.

Passive structures, stopped pumps, running pumps, and fixed gate openings use their existing operation types. No implicit pump operation applies.

## Supply local rain and losses

Use `new_with_sources` to enable local sources. Supply one `LevelPoolSourcePolicy` per pool, including evidence supporting rainfall area and supplied bottom demand.

`CoupledLevelPoolSourceInterval` supplies constant rates for every pool over one declared interval. Rain and potential evaporation use m/s. Potential bottom loss uses m³/s. Supply explicit zero rates where a process does not apply.

Call `trial_with_sources_to` with an interval covering the accepted time and requested endpoint. Source-enabled pools reject `trial_to` because it supplies no forcing.

The calculation freezes rainfall and evaporation area at the accepted start. An exact storage knot selects its right segment. Dry pools use the declared rain-receiving basal footprint.

Rain enters first. Bottom loss then consumes available water, followed by evaporation. Each loss is capped by remaining water. Structure withdrawals use that post-source availability. Water arriving through another structure cannot supply withdrawals during the same interval.

Shortened trials recalculate every source process from the original accepted state. They preserve the original accepted heads for midpoint estimates.

## Advance an interval

`trial_to` calculates a candidate from the accepted state. It checks storage, structure support, shared transfers, and convergence before returning that candidate.

| Result | Caller action |
| --- | --- |
| Successful trial | Read the accepted endpoint and consume `into_next` to retain the candidate. |
| Shortened interval | Continue from the actual accepted time. The requested endpoint remains incomplete. |
| Rejected interval | Read the error. The original accepted state remains unchanged. |

The calculation estimates connection flow from midpoint water levels. It repeats the trial until head and transfer-volume changes meet their separate tolerances. Support flags must also agree exactly.

Head means water elevation. A support flag records whether water reaches a pool or connection.

Nonconvergence shortens the interval and recalculates it. Iteration, retry, and minimum-step limits stop unresolved calculations.

## Understand the water balance

Each connection retains one signed transfer volume. Positive volume moves from endpoint A to endpoint B. Both endpoints use that same volume.

Competing outlets share the donor's post-source available water. Hydraulic inflows cannot finance another outgoing transfer during that interval.

The calculation preserves represented positive water films. A mathematically exhausted donor can retain a small positive volume after conservative floating-point calculations.

Unresolved positive transfers return an error instead of disappearing during aggregation. Numerical bounds describe arithmetic uncertainty without adding water to close a balance.

## Save and restore

Trusted snapshots retain numerical arrays on the selected runtime. Owned checkpoint data copies those arrays into application-owned values for persistence. Copying data from the runtime can return errors.

Use `snapshot()` and `restore()` within a running process. Use `snapshot_data()` and `restore_data()` when your application stores checkpoint data.

Restore checks the physical definition and numerical controls. It retains the destination's configured client. Preserve every checkpoint field and floating-point bit when encoding data.

Source checkpoints also retain physical source policies, cumulative counters, and the original forcing interval. Import recalculates the final interval's sources before checking its transfers and accepted state.

## Current limits

- Sources: Explicit local rain, supplied bottom loss, and evaporation are supported. External inflows and runoff attachment remain excluded.
- Operations: Static manual operations are supported. Control schedules, trajectories, and actuator frames remain excluded.
- Regimes: Existing structure restrictions still apply. Running pumps require wet support. Submerged orifices require water above the opening's top elevation at both endpoints.
- Crossings: Unresolved bed or hydraulic-cutoff transitions return errors after bounded interval shortening.
- Geometry: Multiple-elevation closure events require later event localization. Availability scaling does not resolve those events.
- Accuracy: Synthetic numerical checks establish no measured-event accuracy or universal GPU qualification.
