```@meta
CurrentModule = DynamicHMC
```

# Notation

Notation follows [Betancourt (2017)](https://arxiv.org/abs/1701.02434), with some differences.

Instead of energies, *negative* energies are used in the code.

The following are used consistently for variables:

- `ℓ`: log density we sample from, supports the interface of [LogDensityProblems.AbstractLogDensityProblem](https://github.com/tpapp/LogDensityProblems.jl)
- `κ`: distribution/density that corresponds to kinetic energy
- `H`: Hamiltonian
- `q`: position
- `p`: momentum
- `z`: point in phase space (q,p)
- `ϵ`: stepsize
- `a`: acceptance rate
- `A`: acceptance tuning state
- `ζ`: proposal from trajectory (phase point and weight)
- `τ`: turn statistic
- `d`: divergence statistic
- `π`: log density (**different from papers**)
- `Δ`: logdensity relative to initial point of trajectory

# Low-level building blocks

This is documented only for developers. These are not part of the public API, if you are using them you should reconsider or file an issue.

## Hamiltonian and leapfrog

```@docs
Hamiltonian
PhasePoint
phasepoint_in
rand_phasepoint
neg_energy
get_p♯
loggradient
leapfrog
```

## Finding initial stepsize ``\epsilon``

Local stepsize tuning.

The local acceptance ratio is technically a probability, but for finding the initial stepsize, it is not capped at ``1``.

Also, the values are cached as this is assumed to be moderately expensive to calculate.

```@docs
find_initial_stepsize
InitialStepsizeSearch
find_crossing_stepsize
bisect_stepsize
local_acceptance_ratio
```

## Dual averaging

```@docs
DualAveragingParameters
DualAveragingAdaptation
get_ϵ
adapting_ϵ
adapt_stepsize
```

## Abstract trajectory interface

In contrast to other reference implementations, the algorithm is implemented in a functional style using immutable values. The intention is to provide more transparency and facilitate fine-grained unit testing.

```@docs
adjacent_tree
Termination
sample_trajectory
```

## Proposals

```@docs
Proposal
combined_logprob_logweight
combine_proposals
```

## Divergence statistics

```@docs
DivergenceStatistic
divergence_statistic
isdivergent
combine_divstats
```

## Turn analysis

```@docs
TurnStatistic
combine_turnstats
isturning
```

## Sampling

```@docs
Trajectory
leaf
move
NUTS_transition
```

## Tuning

```@docs
DynamicHMC.AbstractTuner
```

## [Diagnostics](@id diagnostics_lowlevel)

```@docs
NUTS_Statistics
ACCEPTANCE_QUANTILES
explore_local_acceptance_ratios
```

## Reporting information during runs

Samplers take an [`AbstractReport`](@ref) argument, which is then used for reporting. The interface is as follows.

```@docs
DynamicHMC.AbstractReport
DynamicHMC.report!
DynamicHMC.start_progress!
DynamicHMC.end_progress!
```

The default is
```@docs
ReportIO
```

Reporting information can be suppressed with
```@docs
ReportSilent
```

Other interfaces should define similar types.

## Utilities and miscellanea

```@docs
rand_bool
```
