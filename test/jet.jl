using JET
@testset "static analysis with JET.jl" begin
    @test isempty(JET.get_reports(report_package(DynamicHMC, target_modules=(DynamicHMC,))))
end
