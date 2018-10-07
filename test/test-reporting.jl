@testset "reporting" begin
    ℓ = DistributionLogDensity(MvNormal, 3)
    output = @color_output false begin
        @capture_err begin
            sample, nuts = NUTS_init_tune_mcmc(RNG, ℓ, 1000;
                                               report = ReportIO(; countΔ = 100, time_nsΔ = -1))
        end
    end
    float_regex = raw"(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?"
    function expectedA(msg, n)
        r = "$(msg) \\($(n) steps\\)\\n"
        for i in 100:100:n
            r *= "step $(i) \\(of $(n)\\), $(float_regex) s/step\\n"
        end
        r *= "$(float_regex) s/step \\.\\.\\.done\\n"
    end
    raw_regex = join(expectedA.(vcat(fill("MCMC, adapting ϵ", 7), ["MCMC"]),
                                [75, 25, 50, 100, 200, 400, 50, 1000]), "")
    @test occursin(Regex(raw_regex), output)
end
