# Sampling and accessors

Most users would use this function, which initializes and tunes the parameters of the algorithm, then samples. Parameters can be set manually for difficult posteriors.

```@docs
NUTS_init_tune_mcmc
```

!!! important

    The [`NUTS`](@ref) sampler saves a random number generator and uses it for random draws. When running in parallel, you should initialize [`NUTS_init_tune_mcmc`](@ref) with a random number generator as its first argument explicitly, making sure that each thread has its own one.

These functions can be used use to perform the steps above manually.

```@docs
NUTS_init
tune
mcmc
```

The resulting sample is a vector of [`NUTS_Transition`](@ref) objects, for which the following accessors exist:

```@docs
NUTS_Transition
get_position
get_neg_energy
get_depth
get_termination
get_acceptance_rate
get_steps
get_position_matrix
```

# Diagnostics

These are NUTS-specific diagnostics and statistics (except for [`sample_cov`](@ref), which is a convenience function). It is also prudent to use generic MCMC convergence diagnostics, as suggested in the [introduction](@ref Introduction).

```@docs
NUTS_statistics
sample_cov
EBFMI
```

# Fine-grained control

## Kinetic energy

```@docs
KineticEnergy
EuclideanKE
GaussianKE
```

## [NUTS parameters and tuning](@id tuning)

```@docs
NUTS
StepsizeTuner
StepsizeCovTuner
TunerSequence
mcmc_adapting_ϵ
bracketed_doubling_tuner
```
