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

function combine_proposals(_, ::DummyTrajectory, zeta1, zeta2, is_forward, is_doubling)
    if !is_forward
        zeta2, zeta1 = zeta1, zeta2
    end
    w1, z1 = zeta1
    w2, z2 = zeta2
    @test last(z1) + 1 == first(z2) # adjacency and order
    (logaddexp(w1, w2), first(z1):last(z2))
end

function leaf(trajectory::DummyTrajectory, z, is_initial)
    @unpack turning, divergent, ℓ, visited = trajectory
    w = ℓ(z)
    d = z ∈ divergent
    is_initial && @argcheck !d                              # don't start with divergent
    !is_initial && push!(visited, z)                        # save position
    ((w, z:z),                                              # ζ
     (z ∈ turning, z:z),                                    # τ
     is_initial ? (false, 0.0, 0) : (d, min(exp(w), 1), 1)) # d
end

testℓ(z) = -abs2(z - 3) * 0.1

# implicitly, visited region has to be contiguous
testA(ℓ, z) = sum(min.(exp.(ℓ.(UnitRange(extrema(z)...)))))

"sum of acceptance rates for trajectory."
testA(trajectory::DummyTrajectory) = testA(trajectory.ℓ, trajectory.visited)

@testset "dummy adjacent tree full" begin
    trajectory = DummyTrajectory(testℓ)
    ζ, τ, d, z′ = adjacent_tree(nothing, trajectory, 0, 2, true)
    @test trajectory.visited == 1:4
    @test first(ζ) ≈ logsumexp(testℓ.(1:4))
    @test last(ζ) == 1:4
    @test !is_turning(trajectory, τ)
    @test !is_divergent(trajectory, d)
    @test d[2] ≈ testA(trajectory)
    @test d[3] == 4
    @test z′ == 4
end

@testset "dummy adjacent tree turning" begin
    trajectory = DummyTrajectory(testℓ; turning = 5:7)
    ζ, τ, d, z′ = adjacent_tree(nothing, trajectory, 0, 3, true)
    @test trajectory.visited == 1:6 # [5,6] is turning
    @test is_turning(trajectory, τ)
    @test !is_divergent(trajectory, d)
    @test d[2] == testA(trajectory)
    @test d[3] == 6
    @test z′ == 6
end

@testset "dummy adjacent tree divergent" begin
    trajectory = DummyTrajectory(testℓ; divergent = 5:7)
    ζ, τ, d, z′ = adjacent_tree(nothing, trajectory, 0, 3, true)
    @test trajectory.visited == 1:5 # 5 is divergent
    @test is_divergent(trajectory, d)
    @test d[2] ≈ testA(testℓ, 1:5)
    @test d[3] == 5
    @test z′ == 5
end

@testset "dummy adjacent tree full backward" begin
    trajectory = DummyTrajectory(testℓ)
    ζ, τ, d, z′ = adjacent_tree(nothing, trajectory, 0, 3, false)
    @test trajectory.visited == -(1:8)
    @test !is_turning(trajectory, τ)
    @test !is_divergent(trajectory, d)
    @test d[2] ≈ testA(testℓ, -(1:8))
    @test d[3] == 8
    @test z′ == -8
end


## a realized trajectory, with positions and their probabilities

# struct TrajectoryDistribution{Tz, Tf}
#     "ordered positions"
#     positions::Vector{Tz}
#     "probabilities"
#     probabilities::Vector{Tf}
# end

# function TrajectoryDistribution(positions, probabilities)
#     pos = collect(positions)
#     probs = collect(probabilities)
#     @argcheck length(pos) == length(probs)
#     @argcheck sum(probs) ≈ 1
#     p = sortperm(pos)
#     TrajectoryDistribution(pos[p], probs[p])
# end

# """
# Combine two `TrajectoryDistributions` such that the second one has the given
# probability.
# """
# function mix(x::TrajectoryDistribution, y::TrajectoryDistribution, y_prob)
#     @argcheck isempty(x.positions ∩ y.positions)
#     @argcheck 0 ≤ y_prob ≤ 1
#     positions = vcat(x.positions, y.positions)
#     probabilities = vcat((1-y_prob) * x.probabilities, y_prob * y.probabilities)
#     p = sortperm(positions)
#     TrajectoryDistribution(positions[p], probabilities[p])
# end

# @testset "trajectory distributions" begin
#     x = TrajectoryDistribution(0:1, [0.2, 0.8])
#     y = TrajectoryDistribution(2:3, [0.3, 0.7])
#     z = mix(x, y, 0.8)
#     @test z.positions == collect(0:3)
#     @test z.probabilities ≈ vcat(0.2*[0.2, 0.8], 0.8*[0.3, 0.7])
# end

# "Keep track of all probabilities."
# struct ProposalDistribution{Tz, Tf}
#     dist::TrajectoryDistribution{Tz, Tf}
#     ω::Tf
# end

# support(pd::ProposalDistribution) = pd.dist.positions

# function DynamicHMC.combine_proposals(rng, x::ProposalDistribution,
#                                       y::ProposalDistribution, bias_y)
#     logprob_y, ω = combined_logprob_logweight(x.ω, y.ω, bias_y)
#     prob_y = logprob_y ≥ 0 ? one(logprob_y) : exp(logprob_y)
#     dist = mix(x.dist, y.dist, prob_y)
#     ProposalDistribution(dist, ω)
# end

# function DynamicHMC.leaf(trajectory::DummyTrajectory, z, isinitial)
#     @unpack z₀, collecting, positions, turning, divergent, ℓ = trajectory
#     Δ = isinitial ? 0.0 : ℓ(z) - ℓ(z₀)
#     collecting && push!(positions, z)
#     d = isinitial ? divergence_statistic() : divergence_statistic(z ∈ divergent, Δ)
#     τ = DummyTurnStatistic(z ∈ turning)
#     ζ = d.divergent ? nothing : ProposalDistribution(TrajectoryDistribution([z],[1.0]), Δ)
#     ζ, τ, d
# end

# DynamicHMC.move(trajectory::DummyTrajectory, z, fwd) = z + (fwd ? one(z) : -(one(z)))

# """
#     sample_dists(trajectory, z::T, max_depth)

# Return a vector of sampled `TrajectoryDistribution`s along `trajectory`,
# starting from `z`.

# They have uniform probability `0.5^max_depth`.
# """
# function sample_dists(trajectory, z::T, max_depth) where T
#     N = (2^max_depth)
#     function distribution(rng_state)
#         rng = DummyRNG(rng_state, 0)
#         ζ, d, termination, depth = sample_trajectory(rng, trajectory, z, max_depth)
#         @test depth ≤ rng.count ≤ depth+1
#         ζ.dist
#     end
#     [distribution(rng_state) for rng_state in 0:(N-1)]
# end

# """
#     transprob(dists, j)

# Transition probability to `j` according to `dists`.
# """
# function transprob(dists, j)
#     p = 0.0
#     for d in dists
#         ix = findfirst(isequal(d.positions), j)
#         ix ≠ nothing && (p += d.probabilities[ix])
#     end
#     p / length(dists)
# end

# """
#     test_detailed_balance(ℓ, z::T, max_depth; args...)

# For all transitions from `z`, test the detailed balance condition, ie

# ``ℙ(z) ℙ(j ∣ z) = ℙ(j) ℙ(z ∣ j)``

# where ``ℙ(z) = exp(ℓ(z))`` and the transition probabilities ``ℙ(⋅∣⋅)`` are
# calculated using `sample_dist` and `transprob`.

# `z` is the starting point, and all `j`s are checked such that ``|z-j| <
# 2^max_depth``.

# `args` can be used to specify extra keyword arguments (turning, divergence) to
# the `DummyTrajectory` constructor.
# """
# function test_detailed_balance(ℓ, z::T, max_depth; args...) where T
#     DT(z) = DummyTrajectory(z; ℓ = ℓ, collecting = false, args...)
#     K = (max_depth)-1
#     dists = sample_dists(DT(z), z, max_depth)
#     for j in (z-K):(z+K)
#         p1 = exp(ℓ(z)) * transprob(dists, j)
#         p2 = exp(ℓ(j)) * transprob(sample_dists(DT(j), j, max_depth), z)
#         @test p1 ≈ p2
#     end
# end

# @testset "detailed balance flat" begin
#     for max_depth in 1:5
#         test_detailed_balance(_->0.0, 0, max_depth)
#     end
# end

# @testset "detailed balance non-flat" begin
#     for max_depth in 1:5
#         test_detailed_balance(x->x*0.02, 0, max_depth)
#     end
# end

# @testset "detailed balance non-flat turning" begin
#     for max_depth in 1:5
#         test_detailed_balance(x->x*0.02, 0, max_depth; turning = 2:3)
#     end
# end
