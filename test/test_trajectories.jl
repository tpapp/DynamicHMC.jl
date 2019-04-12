#####
##### Tests for the trajectories module
#####

using Test, DynamicHMC.Trajectories, StatsFuns, ArgCheck, DocStringExtensions

Base.@kwdef struct DummyTrajectory
    visited::Vector = Any[]
    turning_left = Inf
    turning_right = -Inf
end

"""
$(TYPEDEF)

Mock trajectory type. `z` is a vector containing a single integer. Saves visited positions.
Initialized buffers with `Union{Missing,...}` types, so that it can be checked.
"""
function Trajectories.move!(z, trajectory::DummyTrajectory, is_forward::Bool)
    z[] += is_forward ? 1 : -1
    push!(trajectory.visited, z[])
    nothing
end

_arr0(T, v = missing) = (a = Array{Union{Missing,T}}(undef, ()); a[] = v; a)
_z(𝑧) = _arr0(Int64, 𝑧)
_p♯(𝑧) = _arr0(Float64, 𝑧)
_ρ(𝑧) = _arr0(Float64, 𝑧 * √2)

Trajectories.log_probability(::DummyTrajectory, z) = z[] * 0.1
Trajectories.get_p♯!(p♯, ::DummyTrajectory, z) = (p♯ .= _p♯(z[]); nothing)
Trajectories.get_ρ!(ρ, ::DummyTrajectory, z) = (ρ .= _ρ(z[]); nothing)
Trajectories.add_ρ!(ρ, ::DummyTrajectory, ρ′) = (ρ .+= ρ′; nothing)
Trajectories.empty_z(::DummyTrajectory) = _arr0(Int)
Trajectories.empty_ρ(::DummyTrajectory) = _arr0(Float64)
Trajectories.empty_p♯(::DummyTrajectory) = _arr0(Float64)
function Trajectories.is_turning(t::DummyTrajectory, _, p♯₋, p♯₊)
    l, r = p♯₋[], p♯₊[]
    if l > r
        l, r = r, l
    end
    t.turning_left ≤ l && r ≤ t.turning_right
end


@testset "fresh buffer consistency" begin
    t = DummyTrajectory()
    max_depth = 5
    b = Buffers(t, max_depth)
    v = Visited()
    @test length(b.ẑs) == length(b.ρs) == length(b.p♯s) == max_depth + 1
    initialize_buffers!(b, t, _z(0))
    @test b.z₋ == b.z₊ == b.ẑ == _z(0)
    @test b.p♯₋ == b.p♯₊ == _p♯(0)
    @test b.ρ == _ρ(0)
    initialize_visited!(v)
    @test v.leapfrog_steps == 0
end

@testset "building adjacent 0" begin
    t = DummyTrajectory()
    max_depth = 5
    b = Buffers(t, max_depth)
    v = Visited()
    initialize_buffers!(b, t, _z(0))
    # rng = nothing, test that it is never called
    @test build_adjacent!(nothing, t, -1000, 0.0, 0, true, v, b, 1) == 0.1
    @test t.visited == [1]
    @test v.leapfrog_steps == 1
    @test v.log∑a ≈ log(2)      # log(exp(0) + exp(0))
    @test b.z₋ == _z(0)
    @test b.z₊ == _z(1)
    @test b.ẑs[1] == _z(1)
    @test b.ρs[1] == _ρ(1)
    @test b.p♯s[1] == _p♯(1)
    @test all(getindex.(b.ẑs[1:1]) .≢ missing)
    @test all(getindex.(b.ẑs[2:max_depth]) .≡ missing)
end

@testset "building adjacent 1" begin
    t = DummyTrajectory()
    max_depth = 5
    b = Buffers(t, max_depth)
    v = Visited()
    initialize_buffers!(b, t, _z(0))
    r = DummyRNG(zeros(10))     # always accept
    @test build_adjacent!(r, t, -1000, 0.0, 1, true, v, b, 1) ≈ log(exp(0.1) + exp(0.2))
    @test r.index == 1
    @test t.visited == 1:2
    @test v.leapfrog_steps == 2
    @test v.log∑a ≈ log(3 * exp(0))
    @test b.z₋ == _z(0)
    @test b.z₊ == _z(2)
    @test b.ẑs[1] == _z(2)
    @test b.ρs[1] ≈ _ρ(sum(1:2))
    @test b.p♯s[1] == _p♯(2)
    @test all(getindex.(b.ẑs[1:2]) .≢ missing)
    @test all(getindex.(b.ẑs[3:max_depth]) .≡ missing)
end

@testset "building adjacent 2" begin
    t = DummyTrajectory()
    max_depth = 5
    b = Buffers(t, max_depth)
    v = Visited()
    initialize_buffers!(b, t, _z(0))
    r = DummyRNG(zeros(10))
    @test build_adjacent!(r, t, -1000, 0.0, 2, true, v, b, 1) ≈ logsumexp((1:4) .* 0.1)
    @test r.index == 3
    @test t.visited == 1:4
    @test v.leapfrog_steps == 4
    @test v.log∑a ≈ log(5 * exp(0))
    @test b.z₋ == _z(0)
    @test b.z₊ == _z(4)
    @test b.ẑs[1] == _z(4)
    @test b.ρs[1] ≈ _ρ(sum(1:4))
    @test b.p♯s[1] == _p♯(4)
    @test all(getindex.(b.ẑs[1:3]) .≢ missing)
    @test all(getindex.(b.ẑs[4:max_depth]) .≡ missing)
end
