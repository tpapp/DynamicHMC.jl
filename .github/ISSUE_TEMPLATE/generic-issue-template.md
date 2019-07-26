---
name: generic issue template
about: question, feature request, or bug report

---

*Please make sure you are using **the latest tagged versions** and **the last stable release of Julia** before proceeding.* Support for other versions is very limited.

If you need *help* using this package for Bayesian inference, please provide a self-contained description of your inference problem (simplifying if possible) and preferably the code you have written so far to code the log likelihood. If you found a *bug*, please provide a self-contained working example, complete with (simulated) data. Please make sure you set a random seed (`Random.seed!`) at the beginning to make your example reproducible. If you are requesting a new *feature*, please provide a description and a rationale.

## Self-contained example that demonstrates the problem

```julia
using DynamicHMC
```

## Output, expected outcome, comparison to other samplers

Did the sampler fail to run, produce incorrect results, …?

## Contributing code for tests

You can contribute to the development of this package by allowing that your example is used as a test. Please indicate whether your code can be incorporated into this package under the MIT "Expat" license, found in the root directory of this package.
