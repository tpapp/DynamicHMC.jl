@testset "NUTS statistics" begin
    N = 1000
    t = collect(instances(Termination))
    directions = Directions(UInt32(0))
    sample = [TreeStatisticsNUTS(randn(), rand(1:5), rand(t), rand(), rand(1:30),
                                 directions) for _ in 1:N]
    stats = NUTS_statistics(sample)
    @test stats.N == N
    @test stats.a_mean ≈ mean(x -> x.acceptance_statistic, sample)
    @test stats.a_quantiles ==
        quantile((x -> x.acceptance_statistic).(sample), ACCEPTANCE_QUANTILES)
    @test stats.termination_counts == counter(map(x -> x.termination, sample))
    @test stats.depth_counts == counter(map(x -> x.depth, sample))
    @test 1.8 ≤ EBFMI(sample) ≤ 2.2 # nonsensical value, just checking calculation
    @test repr(stats) isa AbstractString # just test that it prints w/o error
end
