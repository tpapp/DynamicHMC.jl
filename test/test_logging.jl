# NOTE currently we just check that logging does not error, more explicit testing might make
# sense

ℓ = multivariate_normal(ones(1))
κ = GaussianKineticEnergy(1)
Q = evaluate_ℓ(ℓ, [1.0])

reporters_1 = [
    NoProgressReport(),
    ProgressMeterReport(),
]

reporters_2 = [
    LogProgressReport(),
]

with_logger(NullLogger()) do   # suppress logging in CI
    for reporter in deepcopy(vcat(reporters_1, reporters_2))
        results = mcmc_with_warmup(RNG, ℓ, 10_000; reporter = reporter)
    end

    for reporter in deepcopy(vcat(reporters_1, reporters_2))
        DynamicHMC.report(reporter, "")

        mcmc_reporter_1 = DynamicHMC.make_mcmc_reporter(reporter, 1_000; currently_warmup = true)
        DynamicHMC.report(mcmc_reporter_1, "")
        DynamicHMC.report(mcmc_reporter_1, 1)

        mcmc_reporter_2 = DynamicHMC.make_mcmc_reporter(reporter, 1_000; currently_warmup = false)
        DynamicHMC.report(mcmc_reporter_2, "")
        DynamicHMC.report(mcmc_reporter_2, 1)
    end

    for reporter in deepcopy(reporters_1)
        DynamicHMC.report(reporter, "")
        DynamicHMC.report(reporter, 1)

        mcmc_reporter_1 = DynamicHMC.make_mcmc_reporter(reporter, 1_000; currently_warmup = true)
        DynamicHMC.report(mcmc_reporter_1, "")
        DynamicHMC.report(mcmc_reporter_1, 1)

        mcmc_reporter_2 = DynamicHMC.make_mcmc_reporter(reporter, 1_000; currently_warmup = false)
        DynamicHMC.report(mcmc_reporter_2, "")
        DynamicHMC.report(mcmc_reporter_2, 1)
    end
end
