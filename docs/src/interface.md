# User interface

## Sampling

The main entry point for sampling is

```@docs
mcmc_with_warmup
```

## Warmup

Warmup can be customized.

### Default warmup sequences

A warmup sequence is just a tuple of [warmup building blocks](@ref wbb). Two commonly used sequences are predefined.

```@docs
default_warmup_stages
fixed_stepsize_warmup_stages
```

### [Warmup building blocks](@id wbb)

```@docs
InitialStepsizeSearch
FindLocalOptimum
DualAveraging
TuningNUTS
GaussianKineticEnergy
```

## Progress report

Progress reports can be explicit or silent.

```@docs
NoProgressReport
LogProgressReport
```

## Algorithm and parameters

You probably won't need to change these options with normal usage, except possibly increasing the maximum tree depth.

```@docs
DynamicHMC.NUTS
```

## Inspecting warmup

!!! note
    The warmup interface below is not considered part of the exposed API, and may change with just minor version bump. It is intended for interactive use; the docstrings and the field names of results should be informative.

```@docs
DynamicHMC.mcmc_keep_warmup
```

## Stepwise sampling

!!! note
    The stepwise sampling interface below is not considered part of the exposed API, and may change with just minor version bump.

An experimental interface is available to users who wish to do MCMC one step at a time, eg until some desired criterion about effective sample size or mixing is satisfied. See the docstrings below for an example.

```@docs
DynamicHMC.mcmc_steps
DynamicHMC.mcmc_next_step
```

# Diagnostics

!!! note
    Strictly speaking the `Diagnostics` submodule API is not considered part of the exposed interface, and may change with just minor version bump. It is intended for interactive use.

```@docs
DynamicHMC.Diagnostics.explore_log_acceptance_ratios
DynamicHMC.Diagnostics.summarize_tree_statistics
DynamicHMC.Diagnostics.leapfrog_trajectory
DynamicHMC.Diagnostics.EBFMI
DynamicHMC.PhasePoint
```
