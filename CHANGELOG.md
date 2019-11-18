# Unreleased

- fix penalty sign in initial optimization (#97)

- add example for skipping stepsize search (#104)

# v2.1.0

- add experimental “iterator” interface (#94)

- use `randexp` for Metropolis acceptance draws

- remove dependence on StatsFuns.jl

# v2.0.2

Default keyword arguments for LogProgressReport.

# v2.0.1

Don't print `chain_id` when it is `nothing`.

# v2.0.0

Note: the interface was redesigned. You probably want to review the docs, especially the worked example.

## API changes

- major API change: entry point is now `mcmc_with_warmup`

- refactor warmup code, add initial optimizer

- use the LogDensityProblems v0.9.0 API

- use Julia's Logging module for progress messages

- diagnostics moved to `DynamicHMC.Diagnostics`

  - report turning and divergence positions

  - add `leapfrog_trajectory` for exploration

## Implementation changes

- factor out the tree traversal code

  - abstract trajectory interface

  - separate random and non-random parts

  - stricter and more exact unit tests

- refactor Hamiltonian code slightly

  - caching is now in EvaluatedLogDensity

  - functions renamed

- misc

  - remove dependency on DataStructures, Suppressor

  - cosmetic changes to dual averaging code

  - large test cleanup

# v1.0.6

- fix LogDensityProblems version bounds

# v1.0.5

- fix tuning with singular covariance matrices

# v1.0.4

- minor fixes in tests and coverage

# v1.0.3 and prior

No CHANGELOG available.
