# DynamicHMC

Bare-bones implementation of robust dynamic Hamiltonian Monte Carlo methods.

[![Project Status: WIP – Initial development is in progress, but there has not yet been a stable, usable release suitable for the public.](http://www.repostatus.org/badges/latest/wip.svg)](http://www.repostatus.org/#wip)
[![Build Status](https://travis-ci.org/tpapp/DynamicHMC.jl.svg?branch=master)](https://travis-ci.org/tpapp/DynamicHMC.jl)
[![Coverage Status](https://coveralls.io/repos/tpapp/DynamicHMC.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/tpapp/DynamicHMC.jl?branch=master)
[![codecov.io](http://codecov.io/github/tpapp/DynamicHMC.jl/coverage.svg?branch=master)](http://codecov.io/github/tpapp/DynamicHMC.jl?branch=master)
[![Documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://tpapp.github.io/DynamicHMC.jl/dev)

## Overview

This package implements a modern version of the “No-U-turn sampler” in the Julia language, mostly as described in [Betancourt (2017)](https://arxiv.org/abs/1701.02434), with some tweaks.

In contrast to [Mamba.jl](https://github.com/brian-j-smith/Mamba.jl) and [Klara.jl](https://github.com/JuliaStats/Klara.jl), which provide an integrated framework for building up a Bayesian model from small components, this package requires that you code a *log-density function* of the posterior, which also provides derivatives (for which of course you would use [automatic differentiation](http://www.juliadiff.org/)).

Since most of the runtime is spent on calculating the log-likelihood, this allows the use of standard tools like [profiling](https://docs.julialang.org/en/latest/stdlib/profile/) and [benchmarking](https://github.com/JuliaCI/BenchmarkTools.jl) to optimize its [performance](https://docs.julialang.org/en/latest/manual/performance-tips/).

Consequently, this package requires that the user is comfortable with the basics of the theory of Bayesian inference, to the extent of coding a (log) posterior density in Julia. Gelman et al (2013) and Gelman and Hill (2007) are excellent introductions.

Also, the building blocks of the algorithm are implemented using a *functional* (non-modifying) approach whenever possible, allowing extensive unit testing of components, while at the same time also intended to serve as a transparent, pedagogical introduction to the low-level mechanics of current Hamiltonian Monte Carlo samplers.

## Examples

Examples are available in [DynamicHMCExamples.jl](https://github.com/tpapp/DynamicHMCExamples.jl).

## Support and participation

For general questions, open an issue or ask on [the Discourse forum](https://discourse.julialang.org/).

The API is in the process of being refined to accommodate various modeling approaches. Users who wish to participate in the discussion should subscribe to the Github notifications (“watching” the package). Also, I will do my best to accommodate feature requests, just open issues.

## Bibliography

Betancourt, M. J., Byrne, S., & Girolami, M. (2014). Optimizing the integrator step size for Hamiltonian Monte Carlo. [arXiv preprint arXiv:1411.6669](https://arxiv.org/pdf/1411.6669).

Betancourt, M. (2016). Diagnosing suboptimal cotangent disintegrations in Hamiltonian Monte Carlo. [arXiv preprint arXiv:1604.00695](https://arxiv.org/abs/1604.00695).

Betancourt, M. (2017). A Conceptual Introduction to Hamiltonian Monte Carlo. [arXiv preprint arXiv:1701.02434](https://arxiv.org/abs/1701.02434).

Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D. B., Vehtari, A., & Rubin, D. B. (2013). Bayesian data analysis. : CRC Press.

Gelman, A., & Hill, J. (2007). Data analysis using regression and multilevel/hierarchical models.

Hoffman, M. D., & Gelman, A. (2014). The No-U-turn sampler: adaptively setting path lengths in Hamiltonian Monte Carlo. Journal of Machine Learning Research, 15(1), 1593-1623.
