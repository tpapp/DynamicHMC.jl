@testset "reporting" begin
    ℓ = MvNormal(zeros(3), ones(3))
    @color_output false begin
        output = @capture_err begin
            sample, nuts = NUTS_init_tune_mcmc(RNG, ℓ, length(ℓ), 1000;
                                               report = ReportIO())
        end
    end
    function expectedA(msg, n)
        r = "$msg \\($(n) steps\\)\\n"
        for i in 100:100:n
            r *= "step $(i)/$(n), \\d+\\.\\d+ s/step\\n"
        end
        r *=  " \\.\\.\\.done\\n"
    end
    raw_regex = join(expectedA.(vcat(fill("MCMC, adapting ϵ", 7), ["MCMC"]),
                                [75, 25, 50, 100, 200, 400, 50, 1000]), "")
    @test ismatch(Regex(raw_regex), output)
end
