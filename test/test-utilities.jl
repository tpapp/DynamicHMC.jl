@testset "MvNormal loggradient" begin
    for _ in 1:10
        n = rand(2:6)
        ℓ = MvNormal(randn(n), rand_PDMat(n))
        for _ in 1:10
            test_loggradient(ℓ, randn(n))
        end
    end
end

struct Foo{T}
    a::T
    b::T
end

@testset "≂ comparisons" begin
    @test 1 ≂ 1
    @test 1 ≂ 1.0
    @test [1,2] ≂ [1,2]
    @test [1.0,2.0] ≂ [1,2]
    @test !(1 ≂ 2)
    @test Foo(1,2) ≂ Foo(1,2)
    @test !(Foo(1,2) ≂ Foo(1,3))
    @test !(Foo{Any}(1,2) ≂ Foo(1,2))
end
