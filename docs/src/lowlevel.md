```@meta
CurrentModule = DynamicHMC
```

# Notation

Notation follows [Betancourt (2017)](https://arxiv.org/abs/1701.02434), with some differences.

Instead of energies, *negative* energies are used in the code.

The following are used consistently for variables:

- `ℓ`: log density we sample from, see [this explanation](@ref ell-tutorial)
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

This is documented only for package developers. These are not part of the public API, if you are using them you should reconsider or file an issue.

## Hamiltonian and leapfrog

```@docs
Hamiltonian
PhasePoint
get_ℓq
phasepoint_in
rand_phasepoint
neg_energy
get_p♯
leapfrog
is_valid_ℓq
```

## Finding initial stepsize ``\epsilon``

General rootfinding algorithms.

```@docs
bracket_zero
find_zero
bracket_find_zero
```

Local stepsize tuning.

```@docs
logϵ_residual
find_reasonable_logϵ
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

## Utilities and miscellanea

```@docs
rand_bool
ACCEPTANCE_QUANTILES
```
