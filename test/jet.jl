if VERSION >= v"1.7"

    using JET

    # insert!(LOAD_PATH, 2, "..")

    @testset "static analysis with JET.jl" begin
        @test isempty(JET.get_reports(report_package(DynamicHMC, target_modules=(DynamicHMC,))))
    end

end
