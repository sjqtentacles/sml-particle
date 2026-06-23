# sml-particle

[![CI](https://github.com/sjqtentacles/sml-particle/actions/workflows/ci.yml/badge.svg)](https://github.com/sjqtentacles/sml-particle/actions/workflows/ci.yml)

Sequential Importance Resampling (SIR) particle filter for Standard ML, with
a Kalman filter comparison via **sml-kalman**. Models 1D position tracking with
Gaussian process and observation noise.

## Algorithm (SIR)

Each call to `runSIR obs x0 seed` runs a particle filter with N=200 particles:

1. **Initialise**: all particles at `x0`.
2. **Predict**: perturb each particle by N(0, q) process noise (Box-Muller via
   sml-prng SplitMix64).
3. **Update**: weight each particle by the observation likelihood exp(−(z−x)²/2r²).
4. **Resample**: systematic resampling from the normalised weight distribution.
5. Return the posterior mean after processing all observations.

Parameters: q = 0.01 (process noise variance), r = 0.1 (observation noise std).

## API sketch

```sml
(* SIR particle filter: track position through 5 observations near 1.0 *)
val obs = [1.0, 1.0, 1.0, 1.0, 1.0]
val estimate : real = Particle.runSIR obs 0.0 0wx1234
(* ≈ 1.0 — converges to true position *)

(* Kalman filter on the same system (exact solution) *)
val kalman : real = Particle.runKalman obs 0.0
(* ≈ 1.0 — reference solution *)

(* Single-step filter *)
val x1 : real = Particle.filterStep 1.0 0.0
(* starts shifting estimate toward 1.0 *)

(* Compare both filters on a 5-step observation sequence *)
val (kEst, pEst) : real * real = Particle.compareKalman ()
```

## Known limitations

- **1D only**: the implementation tracks a scalar state. Multi-dimensional
  state spaces require generalising the prediction/update steps.
- **Fixed noise parameters**: q and r are compile-time constants; the API
  does not accept runtime noise parameters.
- **N=200 particles**: sufficient for 1D tracking; for high-dimensional or
  multimodal posteriors, N should be much larger.
- **Systematic resampling**: avoids weight impoverishment better than
  multinomial but is still subject to sample degeneracy over many steps.
- **Seeded, reproducible**: `runSIR` uses a caller-supplied seed for
  deterministic results in tests.

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
  particle.sml     SIR filter + Kalman comparison
  particle.mlb
test/
  test.sml         convergence-to-true-state tests (Kalman ±0.1, SIR ±0.3)
```

## License

MIT. See [LICENSE](LICENSE).
