isinteractive() && include("common.jl")

####
#### test directions mechanism
####

@testset "directions" begin
    directions = Directions(0b110101)
    is_forwards = Vector{Bool}()
    for i in 1:6
        is_forward, directions = next_direction(directions)
        push!(is_forwards, is_forward)
    end
    @test collect(is_forwards) == [true, false, true, false, true, true]
    @test rand(RNG, Directions).flags isa UInt32
end

####
#### dummy trajectory for unit testing
####

"""
Trajectory type that is easy to inspect and reason about, for unit testing.

The field `visited` keeps track of visited nodes, can be reset with `empty!`.
"""
struct DummyTrajectory{L,T,D}
    "Positions that trigger turning."
    turning::T
    "Positions that trigger/mimic divergence."
    divergent::D
    "Log density."
    ℓ::L
    "Visited positions."
    visited::Vector{Int}
end

function DummyTrajectory(ℓ; turning = Set(Int[]), divergent = Set(Int[]))
    DummyTrajectory(turning, divergent, ℓ, Int[])
end

Base.empty!(trajectory) = empty(trajectory.visited)

move(::DummyTrajectory, z, is_forward) = z + (is_forward ? 1 : -1)

function is_turning(::DummyTrajectory, τ)
    turn_flag, positions = τ
    @test length(positions) > 1 # not called on a leaf
    turn_flag
end

function combine_turn_statistics(::DummyTrajectory, τ₁, τ₂)
    turn_flag1, positions1 = τ₁
    turn_flag2, positions2 = τ₂
    @test last(positions1) + 1 == first(positions2) # adjacency and order
    (turn_flag1 && turn_flag2, first(positions1):last(positions2))
end

function combine_visited_statistics(::DummyTrajectory, v₁, v₂)
    a1, s1 = v₁
    a2, s2 = v₂
    (a1 + a2, s1 + s2)
end

# copyied from StatsFuns.jl
log1mexp(x::Real) = x < log(0.5) ? log1p(-exp(x)) : log(-expm1(x))

function combine_proposals(_, ::DummyTrajectory, zeta1, zeta2, logprob2, is_forward)
    lp2 = logprob2 > 0 ? 0.0 : logprob2
    lp1 = logprob2 > 0 ? oftype(lp2, -Inf) : log1mexp(lp2)
    if !is_forward
        # exchange so that we can test for adjacency, and join as a UnitRange
        zeta1, zeta2 = zeta2, zeta1
        lp1, lp2 = lp2, lp1
    end
    z1, p1 = zeta1
    z2, p2 = zeta2
    @test last(z1) + 1 == first(z2) # adjacency and order
    (first(z1):last(z2), vcat(p1 .+ lp1, p2 .+ lp2))
end

function calculate_logprob2(::DummyTrajectory, is_doubling, ω₁, ω₂, ω)
    biased_progressive_logprob2(is_doubling, ω₁, ω₂, ω)
end

function leaf(trajectory::DummyTrajectory, z, is_initial)
    @unpack turning, divergent, ℓ, visited = trajectory
    d = z ∈ divergent
    is_initial && @argcheck !d                              # don't start with divergent
    Δ = ℓ(z)
    v = is_initial ? (0.0, 0) : (min(exp(Δ), 1), 1)
    !is_initial && push!(visited, z)                        # save position
    if d
        nothing, v
    else
        (((z:z, [0.0]),          # ζ = nodes, log prob. within tree
          Δ,                     # ω = Δ for leaf
          (z ∈ turning, z:z)),   # τ
         v)
    end
end

"A log density for testing."
testℓ(z) = -abs2(z - 3) * 0.1

"Total acceptance rate of `ℓ` over `z`"
testA(ℓ, z) = sum(min.(exp.(ℓ.(z)), 1))

"sum of acceptance rates for trajectory."
testA(trajectory::DummyTrajectory) = testA(trajectory.ℓ, trajectory.visited)

@testset "dummy adjacent tree full" begin
    trajectory = DummyTrajectory(testℓ)
    (ζ, ω, τ, z′, i′), v = adjacent_tree(nothing, trajectory, 0, 0, 2, true)
    @test first(ζ) == 1:4
    @test sum(exp, last(ζ)) ≈ 1
    @test trajectory.visited == 1:4
    @test !is_turning(trajectory, τ)
    @test v[1] ≈ testA(trajectory)
    @test v[2] == 4
    @test z′ = i′ == 4
end

@testset "dummy adjacent tree turning" begin
    trajectory = DummyTrajectory(testℓ; turning = 5:7)
    t, v = adjacent_tree(nothing, trajectory, 0, 0, 3, true)
    @test trajectory.visited == 1:6 # [5,6] is turning
    @test t == InvalidTree(5, 6)
    @test v[1] == testA(trajectory)
    @test v[2] == 6
end

@testset "dummy adjacent tree divergent" begin
    trajectory = DummyTrajectory(testℓ; divergent = 5:7)
    t, v = adjacent_tree(nothing, trajectory, 0, 0, 3, true)
    @test trajectory.visited == 1:5 # 5 is divergent
    @test t == InvalidTree(5)
    @test v[1] ≈ testA(testℓ, 1:5)
    @test v[2] == 5
end

@testset "dummy adjacent tree full backward" begin
    trajectory = DummyTrajectory(testℓ)
    (ζ, ω, τ, z′, i′), v = adjacent_tree(nothing, trajectory, 0, 0, 3, false)
    @test first(ζ) == -8:-1
    @test sum(exp, last(ζ)) ≈ 1
    @test trajectory.visited == -(1:8)
    @test !is_turning(trajectory, τ)
    @test v[1] ≈ testA(testℓ, -(1:8))
    @test v[2] == 8
    @test z′ == i′ == -8
end

@testset "dummy sampled tree" begin
    trajectory = DummyTrajectory(testℓ)
    ζ, v, termination, depth = sample_trajectory(nothing, trajectory, 0, 3, Directions(0b101))
    @test trajectory.visited == [1, -1, -2, 2, 3, 4, 5]
    @test first(ζ) == -2:5
    @test sum(exp, last(ζ)) ≈ 1
    @test termination == DynamicHMC.REACHED_MAX_DEPTH
    @test v[1] ≈ testA(trajectory)
    @test v[2] == 7             # initial node does not participate in acceptance rate
end

####
#### Detailed balance tests
####

"An accumulator for log probabilities associated with a position on a trajectory."
empty_accumulator() = Dict{Int,Float64}()

"Add log probabilities `πs` at positions `zs` into `accumulator`."
function add_log_probabilities!(accumulator, zs, πs)
    for (z, π) in zip(zs, πs)
        accumulator[z] = haskey(accumulator, z) ? logaddexp(accumulator[z], π) : π
    end
    accumulator
end

"Normalize an accumulator by depth."
function normalize_accumulator(accumulator, depth)
    D = log(0.5) * depth
    Dict((k => v + D) for (k, v) in pairs(accumulator))
end

"""
An accumulator with the probability of visiting nodes for all trees with `depth`, strarting
from `z`, on `trajectory`.
"""
function visited_log_probabilities(trajectory, z, depth)
    accumulator = empty_accumulator()
    for flags in 0:(2^depth - 1)
        ζ = first(sample_trajectory(nothing, trajectory, z, depth, Directions(UInt32(flags))))
        add_log_probabilities!(accumulator, ζ...)
    end
    normalize_accumulator(accumulator, depth)
end

"""
The probability of visiting node `z′` for all trees with `depth`, strarting from `z`, on
`trajectory`.
"""
function transition_log_probability(trajectory, z, z′, depth)
    p = -Inf
    for flags in 0:(2^depth - 1)
        zs, πs = first(sample_trajectory(nothing, trajectory, z, depth,
                                         Directions(UInt32(flags))))
        ix = findfirst(isequal(z′), zs)
        if ix ≠ nothing
            p = logaddexp(p, πs[ix])
        end
    end
    p + depth * log(0.5)
end

@testset "transition calculations consistency check" begin
    trajectory = DummyTrajectory(testℓ)
    depth = 5
    z = 9
    for (z′, π) in pairs(visited_log_probabilities(trajectory, z, depth))
        @test π ≈ transition_log_probability(trajectory, z, z′, depth)
    end
end

"""
$(SIGNATURES)

For all transitions from `z`, test the detailed balance condition, ie

``ℙ(z) ℙ(j ∣ z) = ℙ(j) ℙ(z ∣ j)``

where ``ℙ(z) = exp(ℓ(z))`` and the transition probabilities ``ℙ(⋅∣⋅)`` are
calculated using `visited_log_probabilities` and `transition_log_probability`.

(We use logs for more accurate calculation.)
"""
function test_detailed_balance(trajectory, z, depth; atol = √eps())
    @unpack ℓ = trajectory
    ℓz = ℓ(z)
    for (z′, π) in pairs(visited_log_probabilities(trajectory, z, depth))
        π′ = transition_log_probability(trajectory, z′, z, depth)
        @test (π + ℓz) ≈ (π′ + ℓ(z′)) atol = atol
    end
end

@testset "detailed balance" begin
    for max_depth in 1:5
        test_detailed_balance(DummyTrajectory(testℓ), 0, max_depth)
    end
    for max_depth in 1:5
        test_detailed_balance(DummyTrajectory(testℓ; turning = 1:2), 3, max_depth)
    end
    for max_depth in 1:6
        test_detailed_balance(DummyTrajectory(testℓ; divergent = 10:11), 3, max_depth)
    end
    for max_depth in 1:6
        test_detailed_balance(DummyTrajectory(testℓ; divergent = 10:12, turning = -3:-2), 3,
                              max_depth)
    end
end
