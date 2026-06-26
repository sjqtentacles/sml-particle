# sml-particle

[![CI](https://github.com/sjqtentacles/sml-particle/actions/workflows/ci.yml/badge.svg)](https://github.com/sjqtentacles/sml-particle/actions/workflows/ci.yml)

Sequential Importance Resampling (SIR) particle filter for Standard ML, with a
Kalman filter comparison via **sml-kalman**. Models 1-D position tracking with
Gaussian process and observation noise, exposes a configurable `params` record,
weighted statistics, effective sample size, and a per-step trace.

## Algorithm (SIR)

For each observation the filter:

1. **Predicts** — perturbs each particle by `N(0, q)` process noise (Box-Muller
   over `sml-prng` SplitMix64, threaded purely for reproducibility).
2. **Updates** — weights each particle by the Gaussian observation likelihood
   `exp(-(z-x)^2 / 2r)`, normalized for numerical stability.
3. **Summarizes** — records the weighted mean and effective sample size.
4. **Resamples** — systematic resampling from the normalized weights.

## API

```sml
type params = { n : int, q : real, r : real, initVar : real }
val defaultParams : params      (* n=200, q=0.01, r=0.1, initVar=1.0 *)

type step = { particles : real list, weights : real list, mean : real, ess : real }

(* weighted statistics *)
val weightedMean        : real list -> real list -> real
val weightedVariance    : real list -> real list -> real
val effectiveSampleSize : real list -> real

(* filters *)
val runSIRWith  : params -> real list -> real -> Word64.word -> real
val runSIR      : real list -> real -> Word64.word -> real
val runSIRTrace : params -> real list -> real -> Word64.word -> step list
val runKalman   : real list -> real -> real

(* legacy *)
val filterStep    : real -> real -> real
val compareKalman : unit -> real * real
```

## Examples

```sml
val obs = [1.0, 1.0, 1.0, 1.0, 1.0]

(* default 200-particle filter, seeded for reproducibility *)
val est = Particle.runSIR obs 0.0 0wx1234           (* ~= 1.0 *)

(* tune the filter *)
val p   = { n = 500, q = 0.01, r = 0.1, initVar = 1.0 }
val est2 = Particle.runSIRWith p obs 0.0 0wx7

(* inspect convergence + degeneracy step by step *)
val trace = Particle.runSIRTrace p obs 0.0 0wx7
val esss  = List.map #ess trace                     (* ESS per step *)

(* Kalman reference solution on the same system *)
val k = Particle.runKalman obs 0.0                  (* ~= 1.0 *)
```

## Notes and limitations

- **1-D state.** The tracker is scalar; multi-dimensional state requires
  generalising predict/update (and the Kalman comparison uses 1x1 matrices).
- **Reproducible.** All randomness threads a caller-supplied `Word64.word` seed,
  so the same seed gives byte-identical results across MLton and Poly/ML.
- **Systematic resampling** mitigates weight degeneracy but does not eliminate
  sample impoverishment over very long sequences.
- ESS is reported from the **pre-resample** normalized weights; after
  resampling weights are uniform by construction.

## Installing with smlpkg

```sh
smlpkg add github.com/sjqtentacles/sml-particle
smlpkg sync
```

Reference from your `.mlb`:

```
lib/github.com/sjqtentacles/sml-particle/particle.mlb
```

## Building and testing

```sh
make test        # MLton
make test-poly   # Poly/ML
make all-tests   # both
make clean
```

## Project layout

```
sml.pkg
Makefile
lib/github.com/sjqtentacles/sml-particle/
  particle.sig     PARTICLE signature
  particle.sml     SIR filter (params/trace/ESS) + Kalman comparison
  particle.mlb
test/
  test.sml         weighted stats, ESS bounds, determinism, convergence, configurable N
```

## License

MIT. See [LICENSE](LICENSE).
