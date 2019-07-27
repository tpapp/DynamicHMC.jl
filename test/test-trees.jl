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
    @test length(positions) > 1
    turn_flag
end

function combine_turn_statistics(::DummyTrajectory, τ₁, τ₂)
    turn_flag1, positions1 = τ₁
    turn_flag2, positions2 = τ₂
    @test last(positions1) + 1 == first(positions2) # adjacency and order
    (turn_flag1 && turn_flag2, first(positions1):last(positions2))
end

is_divergent(::DummyTrajectory, d) = first(d)

function combine_divergence_statistics(::DummyTrajectory, d₁, d₂)
    f1, a1, s1 = d₁
    f2, a2, s2 = d₂
    (f1 || f2, a1 + a2, s1 + s2)
end

function combine_proposals(_, ::DummyTrajectory, zeta1, zeta2, logprob2, is_forward)
    lp2 = logprob2 > 0 ? 1.0 : logprob2
    lp1 = logprob2 > 0 ? zero(lp2) : log1mexp(lp2)
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
    biased_progressive_logprob2(false #=is_doubling =#, ω₁, ω₂, ω)
end

function leaf(trajectory::DummyTrajectory, z, is_initial)
    @unpack turning, divergent, ℓ, visited = trajectory
    Δ = ℓ(z)
    d = z ∈ divergent
    is_initial && @argcheck !d                              # don't start with divergent
    !is_initial && push!(visited, z)                        # save position
    ((z:z, [0.0]),                                          # ζ = nodes, log prob. within tree
     Δ,                                                     # ω = Δ for leaf
     (z ∈ turning, z:z),                                    # τ
     is_initial ? (false, 0.0, 0) : (d, min(exp(Δ), 1), 1)) # d
end

testℓ(z) = -abs2(z - 3) * 0.1

testA(ℓ, z) = sum(min.(exp.(ℓ.(z))))

"sum of acceptance rates for trajectory."
testA(trajectory::DummyTrajectory) = testA(trajectory.ℓ, trajectory.visited)

@testset "dummy adjacent tree full" begin
    trajectory = DummyTrajectory(testℓ)
    ζ, ω, τ, d, z′ = adjacent_tree(nothing, trajectory, 0, 2, true)
    @test first(ζ) == 1:4
    @test sum(exp, last(ζ)) ≈ 1
    @test trajectory.visited == 1:4
    @test !is_turning(trajectory, τ)
    @test !is_divergent(trajectory, d)
    @test d[2] ≈ testA(trajectory)
    @test d[3] == 4
    @test z′ == 4
end

@testset "dummy adjacent tree turning" begin
    trajectory = DummyTrajectory(testℓ; turning = 5:7)
    ζ, ω, τ, d, z′ = adjacent_tree(nothing, trajectory, 0, 3, true)
    @test trajectory.visited == 1:6 # [5,6] is turning
    @test is_turning(trajectory, τ)
    @test !is_divergent(trajectory, d)
    @test d[2] == testA(trajectory)
    @test d[3] == 6
    @test z′ == 6
end

@testset "dummy adjacent tree divergent" begin
    trajectory = DummyTrajectory(testℓ; divergent = 5:7)
    ζ, ω, τ, d, z′ = adjacent_tree(nothing, trajectory, 0, 3, true)
    @test trajectory.visited == 1:5 # 5 is divergent
    @test is_divergent(trajectory, d)
    @test d[2] ≈ testA(testℓ, 1:5)
    @test d[3] == 5
    @test z′ == 5
end

@testset "dummy adjacent tree full backward" begin
    trajectory = DummyTrajectory(testℓ)
    ζ, ω, τ, d, z′ = adjacent_tree(nothing, trajectory, 0, 3, false)
    @test first(ζ) == -8:-1
    @test sum(exp, last(ζ)) ≈ 1
    @test trajectory.visited == -(1:8)
    @test !is_turning(trajectory, τ)
    @test !is_divergent(trajectory, d)
    @test d[2] ≈ testA(testℓ, -(1:8))
    @test d[3] == 8
    @test z′ == -8
end

@testset "dummy sampled tree" begin
    trajectory = DummyTrajectory(testℓ)
    ζ, d, termination, depth = sample_trajectory(nothing, trajectory, 0, 3, Directions(0b101))
    @test trajectory.visited == [1, -1, -2, 2, 3, 4, 5]
    @test first(ζ) == -2:5
    @test sum(exp, last(ζ)) ≈ 1
    @test termination == DynamicHMC.MaxDepth
    @test !is_divergent(trajectory, d)
    @test d[2] ≈ testA(trajectory)
    @test d[3] == 7             # initial node does not participate in acceptance rate
end

####
#### Detailed balance tests
####

empty_accumulator() = Dict{Int,Float64}()

function add_log_probabilities!(accumulator, zs, πs)
    for (z, π) in zip(zs, πs)
        accumulator[z] = haskey(accumulator, z) ? logaddexp(accumulator[z], π) : π
    end
    accumulator
end

function normalize_accumulator(accumulator, depth)
    D = log(0.5) * depth
    Dict((k => v + D) for (k, v) in pairs(accumulator))
end

function visited_log_probabilities(trajectory, z, depth)
    accumulator = empty_accumulator()
    for flags in 0:(2^depth - 1)
        ζ = first(sample_trajectory(nothing, trajectory, z, depth, Directions(UInt32(flags))))
        add_log_probabilities!(accumulator, ζ...)
    end
    normalize_accumulator(accumulator, depth)
end

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
