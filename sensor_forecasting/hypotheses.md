# Task 1 — predictions recorded before execution

Recorded on 2026-10-04, before running the implementation checks, fitting any
forecaster, or inspecting Outputs 1.1–4.3. These are hypotheses, not measured
results. Preserve this record; discuss revisions separately in the report.

## Response 1A — expected validation ordering

The order below runs from lowest to highest expected aggregate validation error.
The sensor-specific/shared ordering is a weak prediction; I expect their errors
to be close.

| Population | Predicted ordering |
| --- | --- |
| Established stations 1–3 | Period-routed ridge → Raw Attention → Shared ridge → Sensor-specific ridge |
| Held-out station 4 | Period-routed ridge → Raw Attention → Shared ridge; Sensor-specific ridge is unavailable |

### Reasoning

- **Period-routed ridge:** the operating period changes the phase advance needed
  for a forecast. Routing by a period estimated from the current context gives
  each fitted linear map a narrower range of continuation rules to learn. This
  explicit structure should be useful in the supplied drift-plus-repetition world.
- **Raw Attention:** its learned projections can construct different mixing
  weights for different contexts. I expect that flexibility to improve on a
  single fixed linear map, but learning around the large slow component may be
  harder than using an explicit period estimator and router.
- **Shared ridge:** it pools established-station training windows and benefits
  from anchoring away offsets, but must compromise across incompatible operating
  paces. Pooling may make its estimate more stable than separate station fits.
- **Sensor-specific ridge:** station identity does not identify the current
  operating pace. Each station still spans all three products and many periods,
  while each matrix has fewer fitting windows. It could nevertheless beat the
  shared fit if station-specific amplitude, phase, or drift differences matter
  enough; that is why their predicted ordering is uncertain.
- **Station 4:** shared, period-routed, and neural models can use its observed
  context without fitting on it. A sensor-specific model has no fitted matrix
  for this station, so it must be marked unavailable rather than ranked last or
  assigned an invented error.

I expect the benefit of context-dependent rules to appear across Products A, B,
and C, but I do not predict a fixed product difficulty ordering: each product
trades the number of visible cycles against the number of future cycles and
the sensitivity of phase extrapolation to period-estimation error.

## Response 4(a) — hypothesis before Output 4.1

On contexts with especially prominent slow drift, I expect decomposition to help
more than switching to delay mixing alone, because it assigns the slow component
to a separate forecast head and exposes a cleaner repeating remainder to the
encoder. I expect both upgrades to help overall and the combined
Autoformer-inspired model to improve on Raw Attention, but with an MSE reduction
smaller than the sum of the two individual reductions because the upgrades
address overlapping difficulties in the same drift-plus-repetition signal.

## How the hypotheses will be evaluated

- Compare the four Part 1 forecasters using Output 1.1, separately by population
  and product, and quantify the differences before revising the expected order.
- Use Outputs 1.2 and 1.3 for the mechanism explanation, without interpreting
  attention maps as proof that the physical period was recovered.
- For the four neural models, compare decomposition under each mixer and delay
  mixing under each representation. Assess overlapping or additive gains in
  **validation MSE**, using Raw Attention as the common reference.
- Use the full preset for all reported evidence. Choose deployment models from
  validation errors and measured costs before accessing final-test errors.
