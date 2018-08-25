
# trajectory type that is easy to inspect and reason about, for unit testing

"""
Structure that can take the place of a trajectory for testing tree traversal and
proposals.
"""
struct DummyTrajectory{Tz}
    "Initial position."
    z₀::Tz
    "Flag for collecting values."
    collecting::Bool
    "History of positions `leaf` was called with, when `collecting`."
    positions::Vector{Tz}
    "Positions that mimic turning."
    turning
    "Positions that mimic divergence."
    divergent
    "Log density."
    ℓ
end

"Convenience constructor for DummyTrajectory."
function DummyTrajectory(z₀::T; turning = Set{T}(), divergent = Set{T}(),
                 collecting = true, ℓ = (z) -> 0.0) where T
    DummyTrajectory(z₀, collecting, Vector{T}(), turning, divergent, ℓ)
end

"Dummy Turn statistic."
struct DummyTurnStatistic
    turning::Bool
end

DynamicHMC.combine_turnstats(τ₁::T, τ₂::T) where {T <: DummyTurnStatistic} =
    T(τ₁.turning && τ₂.turning)

DynamicHMC.isturning(τ::DummyTurnStatistic) = τ.turning


# a realized trajectory, with positions and their probabilities

struct TrajectoryDistribution{Tz, Tf}
    "ordered positions"
    positions::Vector{Tz}
    "probabilities"
    probabilities::Vector{Tf}
end

function TrajectoryDistribution(positions, probabilities)
    pos = collect(positions)
    probs = collect(probabilities)
    @argcheck length(pos) == length(probs)
    @argcheck sum(probs) ≈ 1
    p = sortperm(pos)
    TrajectoryDistribution(pos[p], probs[p])
end

"""
Combine two `TrajectoryDistributions` such that the second one has the given
probability.
"""
function mix(x::TrajectoryDistribution, y::TrajectoryDistribution, y_prob)
    @argcheck isempty(x.positions ∩ y.positions)
    @argcheck 0 ≤ y_prob ≤ 1
    positions = vcat(x.positions, y.positions)
    probabilities = vcat((1-y_prob) * x.probabilities, y_prob * y.probabilities)
    p = sortperm(positions)
    TrajectoryDistribution(positions[p], probabilities[p])
end

@testset "trajectory distributions" begin
    x = TrajectoryDistribution(0:1, [0.2, 0.8])
    y = TrajectoryDistribution(2:3, [0.3, 0.7])
    z = mix(x, y, 0.8)
    @test z.positions == collect(0:3)
    @test z.probabilities ≈ vcat(0.2*[0.2, 0.8], 0.8*[0.3, 0.7])
end

"Keep track of all probabilities."
struct ProposalDistribution{Tz, Tf}
    dist::TrajectoryDistribution{Tz, Tf}
    ω::Tf
end

support(pd::ProposalDistribution) = pd.dist.positions

function DynamicHMC.combine_proposals(rng, x::ProposalDistribution,
                                      y::ProposalDistribution, bias_y)
    logprob_y, ω = combined_logprob_logweight(x.ω, y.ω, bias_y)
    prob_y = logprob_y ≥ 0 ? one(logprob_y) : exp(logprob_y)
    dist = mix(x.dist, y.dist, prob_y)
    ProposalDistribution(dist, ω)
end

function DynamicHMC.leaf(trajectory::DummyTrajectory, z, isinitial)
    @unpack z₀, collecting, positions, turning, divergent, ℓ = trajectory
    Δ = isinitial ? 0.0 : ℓ(z) - ℓ(z₀)
    collecting && push!(positions, z)
    d = isinitial ? divergence_statistic() : divergence_statistic(z ∈ divergent, Δ)
    τ = DummyTurnStatistic(z ∈ turning)
    ζ = d.divergent ? nothing : ProposalDistribution(TrajectoryDistribution([z],[1.0]), Δ)
    ζ, τ, d
end

DynamicHMC.move(trajectory::DummyTrajectory, z, fwd) = z + (fwd ? one(z) : -(one(z)))

@testset "dummy adjacent tree full" begin
    trajectory = DummyTrajectory(0)
    ζ, τ, d, z′ = adjacent_tree(nothing, trajectory, 0, 2, true)
    @test trajectory.positions == collect(1:4)
    @test z′ == 4
    @test !isturning(τ)
    @test !isdivergent(d)
    @test d.∑a == 4.0
    @test d.steps == 4
    @test support(ζ) == collect(1:4)
end

@testset "dummy adjacent tree turning" begin
    trajectory = DummyTrajectory(0; turning = 5:7)
    ζ, τ, d, z′ = adjacent_tree(nothing, trajectory, 0, 3, true)
    @test trajectory.positions == collect(1:6) # [5,6] is turning
    @test z′ == 6
    @test isturning(τ)
    @test !isdivergent(d)
    @test d.∑a == 6.0
    @test d.steps == 6
end

@testset "dummy adjacent tree divergent" begin
    trajectory = DummyTrajectory(0; divergent = 5:7)
    ζ, τ, d, z′ = adjacent_tree(nothing, trajectory, 0, 3, true)
    @test trajectory.positions == collect(1:5)
    @test z′ == 5
    @test isdivergent(d)
    @test !isturning(τ)
    @test d.∑a == 5.0
    @test d.steps == 5
end

@testset "dummy adjacent tree full backward" begin
    trajectory = DummyTrajectory(0)
    ζ, τ, d, z′ = adjacent_tree(nothing, trajectory, 0, 3, false)
    @test trajectory.positions == collect(-(1:8))
    @test z′ == -8
    @test !isdivergent(d)
    @test !isturning(τ)
    @test d.∑a == 8.0
    @test d.steps == 8
    @test support(ζ) == collect(-8:-1)
end

"Random number generator replacement with predefined boolean results."
mutable struct DummyRNG{T}
    state::T
    count::Int
end

"Generate specified random number placeholders using DummyRNG."
function DynamicHMC.rand_bool(rng::DummyRNG, ::AbstractFloat)
    value = rng.state & 1 > 0
    rng.state >>= 1
    rng.count += 1
    value
end

@testset "dummy RNG" begin
    rng = DummyRNG(0b1001101, 0)
    b = [true, false, true, true, false, false, true, false]
    @test collect(rand_bool(rng, 0.0) for _ in 1:8) == b
end

"""
    sample_dists(trajectory, z::T, max_depth)

Return a vector of sampled `TrajectoryDistribution`s along `trajectory`,
starting from `z`.

They have uniform probability `0.5^max_depth`.
"""
function sample_dists(trajectory, z::T, max_depth) where T
    N = (2^max_depth)
    function distribution(rng_state)
        rng = DummyRNG(rng_state, 0)
        ζ, d, termination, depth = sample_trajectory(rng, trajectory, z, max_depth)
        @test depth ≤ rng.count ≤ depth+1
        ζ.dist
    end
    [distribution(rng_state) for rng_state in 0:(N-1)]
end

"""
    transprob(dists, j)

Transition probability to `j` according to `dists`.
"""
function transprob(dists, j)
    p = 0.0
    for d in dists
        ix = findfirst(isequal(d.positions), j)
        ix ≠ nothing && (p += d.probabilities[ix])
    end
    p / length(dists)
end

"""
    test_detailed_balance(ℓ, z::T, max_depth; args...)

For all transitions from `z`, test the detailed balance condition, ie

``ℙ(z) ℙ(j ∣ z) = ℙ(j) ℙ(z ∣ j)``

where ``ℙ(z) = exp(ℓ(z))`` and the transition probabilities ``ℙ(⋅∣⋅)`` are
calculated using `sample_dist` and `transprob`.

`z` is the starting point, and all `j`s are checked such that ``|z-j| <
2^max_depth``.

`args` can be used to specify extra keyword arguments (turning, divergence) to
the `DummyTrajectory` constructor.
"""
function test_detailed_balance(ℓ, z::T, max_depth; args...) where T
    DT(z) = DummyTrajectory(z; ℓ = ℓ, collecting = false, args...)
    K = (max_depth)-1
    dists = sample_dists(DT(z), z, max_depth)
    for j in (z-K):(z+K)
        p1 = exp(ℓ(z)) * transprob(dists, j)
        p2 = exp(ℓ(j)) * transprob(sample_dists(DT(j), j, max_depth), z)
        @test p1 ≈ p2
    end
end

@testset "detailed balance flat" begin
    for max_depth in 1:5
        test_detailed_balance(_->0.0, 0, max_depth)
    end
end

@testset "detailed balance non-flat" begin
    for max_depth in 1:5
        test_detailed_balance(x->x*0.02, 0, max_depth)
    end
end

@testset "detailed balance non-flat turning" begin
    for max_depth in 1:5
        test_detailed_balance(x->x*0.02, 0, max_depth; turning = 2:3)
    end
end
