# DynamicHMC

Implementation of robust dynamic Hamiltonian Monte Carlo methods in Julia.

[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![Build Status](https://travis-ci.org/tpapp/DynamicHMC.jl.svg?branch=master)](https://travis-ci.org/tpapp/DynamicHMC.jl)
[![codecov.io](http://codecov.io/github/tpapp/DynamicHMC.jl/coverage.svg?branch=master)](http://codecov.io/github/tpapp/DynamicHMC.jl?branch=master)
[![Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://tpapp.github.io/DynamicHMC.jl/stable)
[![Documentation](https://img.shields.io/badge/docs-master-blue.svg)](https://tpapp.github.io/DynamicHMC.jl/latest)
[![DOI](https://zenodo.org/badge/93741413.svg)](https://zenodo.org/badge/latestdoi/93741413)

## Overview

This package implements a modern version of the “No-U-turn sampler” in the Julia language, mostly as described in [Betancourt (2017)](https://arxiv.org/abs/1701.02434), with some tweaks.

In contrast to frameworks which utilize a directed acyclic graph to build a posterior for a Bayesian model from small components, this package requires that you code a *log-density function* of the posterior in Julia. Derivatives can be provided manually, or using [automatic differentiation](http://www.juliadiff.org/).

Consequently, this package requires that the user is comfortable with the basics of the theory of Bayesian inference, to the extent of coding a (log) posterior density in Julia. This approach allows the use of standard tools like [profiling](https://docs.julialang.org/en/v1/manual/profile/) and [benchmarking](https://github.com/JuliaCI/BenchmarkTools.jl) to optimize its [performance](https://docs.julialang.org/en/v1/manual/performance-tips/).

The building blocks of the algorithm are implemented using a *functional* (non-modifying) approach whenever possible, allowing extensive unit testing of components, while at the same time also intended to serve as a transparent, pedagogical introduction to the low-level mechanics of current Hamiltonian Monte Carlo samplers, and as a platform for research into MCMC methods.

Please start with the [documentation](https://tpapp.github.io/DynamicHMC.jl/latest).

## Examples

- Some basic examples are available in [DynamicHMCExamples.jl](https://github.com/tpapp/DynamicHMCExamples.jl).

- [DynamicHMCModels.jl](https://github.com/StatisticalRethinkingJulia/DynamicHMCModels.jl) contains worked examples from the [Statistical Rethinking](https://xcelab.net/rm/statistical-rethinking/) book.

## Support and participation

For general questions, open an issue or ask on [the Discourse forum](https://discourse.julialang.org/). I am happy to help with models.

Users who rely on this package and want to participate in discussions are recommended to subscribe to the Github notifications (“watching” the package). Also, I will do my best to accommodate feature requests, just open issues.

## Bibliography

Betancourt, M. J., Byrne, S., & Girolami, M. (2014). Optimizing the integrator step size for Hamiltonian Monte Carlo. [arXiv preprint arXiv:1411.6669](https://arxiv.org/pdf/1411.6669).

Betancourt, M. (2016). Diagnosing suboptimal cotangent disintegrations in Hamiltonian Monte Carlo. [arXiv preprint arXiv:1604.00695](https://arxiv.org/abs/1604.00695).

Betancourt, M. (2017). A Conceptual Introduction to Hamiltonian Monte Carlo. [arXiv preprint arXiv:1701.02434](https://arxiv.org/abs/1701.02434).

Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D. B., Vehtari, A., & Rubin, D. B. (2013). Bayesian data analysis. : CRC Press.

Gelman, A., & Hill, J. (2007). Data analysis using regression and multilevel/hierarchical models.

Hoffman, M. D., & Gelman, A. (2014). The No-U-turn sampler: adaptively setting path lengths in Hamiltonian Monte Carlo. Journal of Machine Learning Research, 15(1), 1593-1623.

McElreath, R. (2018). Statistical rethinking: A Bayesian course with examples in R and Stan. Chapman and Hall/CRC.
