import DynamicHMC: logdensity, leapfrog, isturning, adjacent_tree, getp♯,
    PhasePoint, phasepoint, DoublingMultinomialSampler, sample_trajectory,
    rand_bool, leaf_proposal, TurnStatistic, combine_proposals,
    transition_logprob_logweight

import Base.Random: GLOBAL_RNG

struct DummyPosition{T}
    x::T
end

Base.:+(x::DummyPosition, y::Real) = DummyPosition(x.x+y)

Base.:+(x::DummyPosition, y::DummyPosition) = DummyPosition(x.x+y.x)

"""
Structure that can take the place of a Hamiltonian for testing tree
traversal and proposals.
"""
struct DummyHamiltonian{Tz, Tf}
    zs::Vector{Tz}
    forbidden::Tf
    collecting::Bool
end

DummyHamiltonian(T::Type; forbidden = Set{T}(), collecting = true) =
    DummyHamiltonian(PhasePoint{T}[], forbidden, collecting)

phasepoint(::DummyHamiltonian, q) = PhasePoint(q, q, q)

function leapfrog(H::DummyHamiltonian, z, ϵ)
    z′ = PhasePoint(z.q + ϵ, z.p + ϵ, z.q)
    H.collecting && push!(H.zs, z′)
    z′
end

logdensity(H::DummyHamiltonian, z) = z.q.x ≤ 1000 ? 0.0 : -Inf

function getp♯{T}(H::DummyHamiltonian, z::PhasePoint{DummyPosition{T}})
    turning = z.p.x ∈ H.forbidden
    DummyPosition(one(T) * (turning ? 1 : -1))
end

function isturning{T <: DummyPosition}(stats::TurnStatistic{T})
    stats.p♯₋.x > 0 || stats.p♯₊.x > 0
end

function dummy_H_sampler_z(; z_index = 0, args...)
    H = DummyHamiltonian(DummyPosition{Int}; args...)
    sampler = DoublingMultinomialSampler(GLOBAL_RNG, H, 0.0, 1)
    q = DummyPosition(z_index)
    z = phasepoint(H, q)
    H, sampler, z
end

index(z::PhasePoint{<: DummyPosition}) = z.q.x

@testset "dummy adjacent tree full" begin
    H, sampler, z = dummy_H_sampler_z()
    t, d, z′ = adjacent_tree(sampler, z, 2, true)
    @test index.(H.zs) == collect(1:4)
    @test index(z′) == 4
    @test !d.divergent
    @test d.∑a == 4.0
    @test d.steps == 4
    @test index(t.proposal.z) ∈ 1:4
end

@testset "dummy adjacent tree turning" begin
    H, sampler, z = dummy_H_sampler_z(; forbidden = 5:7)
    t, d, z′ = adjacent_tree(sampler, z, 3, true)
    @test index.(H.zs) == collect(1:6) # [5,6] is turning
    @test index(z′) == 6
    @test !d.divergent
    @test d.∑a == 6.0
    @test d.steps == 6
    @test t == nothing
end

@testset "dummy adjacent tree divergent" begin
    H, sampler, z = dummy_H_sampler_z(; z_index = 1000)
    t, d, z′ = adjacent_tree(sampler, z, 1, true)
    @test index.(H.zs) == [1001] # 1001 is divergent
    @test index(z′) == 1001
    @test d.divergent
    @test d.∑a == 0.0
    @test d.steps == 1
    @test t == nothing
end

@testset "dummy adjacent tree full backward" begin
    H, sampler, z = dummy_H_sampler_z()
    t, d, z′ = adjacent_tree(sampler, z, 3, false)
    @test index.(H.zs) == collect(-(1:8))
    @test index(z′) == -8
    @test !d.divergent
    @test d.∑a == 8.0
    @test d.steps == 8
    @test index(t.proposal.z) ∈ -(1:8)
end

struct ProposalDistribution{Tz, Tf}
    probabilities::Dict{Tz,Tf}
    logweight::Tf
end

leaf_proposal(::Type{ProposalDistribution}, z, Δ::T) where T =
    ProposalDistribution(Dict(index(z) => one(T)), Δ)

function combine_proposals(rng, x::ProposalDistribution, y::ProposalDistribution, bias_y)
    logprob_y, logweight = transition_logprob_logweight(x.logweight, y.logweight, bias_y)
    prob_y = logprob_y ≥ 0 ? one(logprob_y) : exp(logprob_y)
    prob(probabilities::Dict{Tz,Tf}, z) where {Tz,Tf} = get(probabilities, z, zero(Tf))
    prob(x::ProposalDistribution, z) = prob(x.probabilities, z)
    probability(key) =  prob(x, key) * (1-prob_y) + prob(y, key) * prob_y
    probabilities = Dict(key => probability(key)
                         for key in keys(x.probabilities) ∪ keys(y.probabilities))
    ProposalDistribution(probabilities, logweight)
end

mutable struct DummyRNG{T}
    state::T
    count::Int
end

function rand_bool(rng::DummyRNG, ::Any)
    value = rng.state & 1 > 0
    rng.state >>= 1
    rng.count += 1
    value
end

@testset "dummy RNG" begin
    rng = DummyRNG(0b1001101, 0)
    @test collect(rand_bool(rng, nothing) for _ in 1:8) == [true, false, true, true,
                                                            false, false, true, false]
end

function transition_probabilities(rng_state::Int, H, i, max_depth)
    rng = DummyRNG(rng_state, 0)
    z = phasepoint(H, DummyPosition(i))
    sampler = DoublingMultinomialSampler(rng, H, 0.0, 1;
                                         proposal_type = ProposalDistribution,
                                         max_depth = max_depth)
    t, d, termination, depth = sample_trajectory(sampler, z)
    @test rng.count == depth
    t.proposal.probabilities
end

function transition_probabilities(H, i::T, max_depth; args...) where T
    probabilities = Dict{T, Float64}()
    N = (2^max_depth)
    for rng_state in 0:(N - 1)
        probabilities′ = transition_probabilities(rng_state, H, i, max_depth)
        for (z, p) in probabilities′
            probabilities[z] = get(probabilities, z, 0) + p
        end
    end
    Dict(z => p/N for (z,p) in probabilities if p > 0)
end

function test_detailed_balance(i::T, max_depth) where T
    H = DummyHamiltonian(DummyPosition{T})
    for (i′, p) in transition_probabilities(H, i, max_depth)
        probabilities′ = transition_probabilities(H, i′, max_depth)
        @test haskey(probabilities′, i)
        @test probabilities′[i] ≈ p
    end
end

@testset "detailed balance flat" begin
    for max_depth in 1:5
        test_detailed_balance(0, max_depth)
    end
end
