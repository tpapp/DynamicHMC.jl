#####
##### Abstract tree/trajectory interface
#####

####
#### Directions
####

"Maximum number of iterations [`next_direction`](@ref) supports."
const MAX_DIRECTIONS_DEPTH = 32

"""
Internal type implementing random directions. Draw a new value with `rand`, see
[`next_direction`](@ref).

Serves two purposes: a fixed value of `Directions` is useful for unit testing, and drawing a
single bit flag collection economizes on the RNG cost.
"""
struct Directions
    flags::UInt32
end

Base.rand(rng::AbstractRNG, ::Type{Directions}) = Directions(rand(rng, UInt32))

"""
$(SIGNATURES)

Return the next direction flag and the new state of directions. Results are undefined for
more than [`MAX_DIRECTIONS_DEPTH`](@ref) updates.
"""
function next_direction(directions::Directions)
    @unpack flags = directions
    Bool(flags & 0x01), Directions(flags >>> 1)
end

####
#### Trajectory interface
####

"""
    $(FUNCTIONNAME)(trajectory, z, is_forward)

Move along the trajectory in the specified direction. Return the new position.
"""
function move end

"""
    $(FUNCTIONNAME)(trajectory, τ)

Test if the turn statistics indicate that the corresponding tree is turning.

Will only be called on nontrivial trees (at least two nodes).
"""
function is_turning end

"""
    $(FUNCTIONNAME)(trajectory, τ₁, τ₂)

Combine turn statistics on trajectory. Implementation can assume that the trees that
correspond to the turn statistics have the same ordering.
"""
function combine_turn_statistics end

"""
    $(FUNCTIONNAME)(trajectory, d)

Test if the divergence statistics indicate that the corresponding tree is divergent.
"""
function is_divergent end

"""
    $(FUNCTIONNAME)(trajectory, d₁, d₂)

Combine divergence statistics for adjacent trees trajectory. Implementation should be
invariant to the ordering of `d₁` and `d₂` (ie the operation is commutative).
"""
function combine_divergence_statistics end

"""
    $(FUNCTIONNAME)(trajectory, is_doubling::Bool, ω₁, ω₂, ω)

Calculate the log probability if selecting the subtree corresponding to `ω₁`. When
`is_doubling`, the tree corresponding to `ω₂` was obtained from a doubling step (this can be
relevant eg for biased progressive sampling).

The value `ω = logaddexp(ω₁, ω₂)` is provided for avoiding redundant calculations.

See [`biased_progressive_logprob1`](@ref) for an implementation.
"""
function calculate_logprob1 end

"""
    $(FUNCTIONNAME)(rng, trajectory, ζ₁, ζ₂, logprob1::Real, is_forward::Bool)

Combine two proposals `ζ₁, ζ₂` on `trajectory`, with log probability `logprob1` for
selecting `ζ1`. `ζ₁` is before `ζ₂` iff `is_forward`.
"""
function combine_proposals end

"""
    ζ, ω, τ, d = $(FUNCTIONNAME)(trajectory, z, is_initial)

Return

- the proposal `ζ`,
- the log weight (probability) of node `ω`,
- turn statistics `τ` (never tested as with `is_turning` for leafs), and
- divergence statistics `d`

for a tree made of a single node. When `is_initial == true`, this is the first node.
"""
function leaf end

####
#### utilities
####

"""
$(SIGNATURES)

Combine turn statistics with the given direction. When `is_forward`, `τ₁` is before `τ₂`,
otherwise after.

Internal helper function.
"""
function combine_turn_statistics_in_direction(trajectory, τ₁, τ₂, is_forward::Bool)
    if is_forward
        combine_turn_statistics(trajectory, τ₁, τ₂)
    else
        combine_turn_statistics(trajectory, τ₂, τ₁)
    end
end

function combine_proposals_and_logweights(rng, trajectory, ζ₁, ζ₂, ω₁::Real, ω₂::Real,
                                          is_forward::Bool, is_doubling::Bool)
    ω = logaddexp(ω₁, ω₂)
    logprob1 = calculate_logprob1(trajectory, is_doubling, ω₁, ω₂, ω)
    ζ = combine_proposals(rng, trajectory, ζ₁, ζ₂, logprob1, is_forward)
    ζ, ω
end

"""
$(SIGNATURES)

Given (relative) log probabilities `ω₁` and `ω₂`, return the log probabiliy of
drawing a sample from the second (`logprob1`).

When `bias`, biases towards the second argument, introducing anti-correlations.
"""
function biased_progressive_logprob1(bias::Bool, ω₁::Real, ω₂::Real, ω = logaddexp(ω₁, ω₂))
    ω₂ - (bias ? ω₁ : ω)
end

####
#### abstract trajectory interface
####

"""
    ζ, τ, d, z = adjacent_tree(rng, trajectory, z, depth, is_forward)

Traverse the tree of given `depth` adjacent to point `z` in `trajectory`.

`is_forward` specifies the direction, `rng` is used for random numbers in
[`combine_proposals`](@ref).

Return:

- `ζ`: the proposal from the tree. Only valid when `!isdivergent(d) && !isturning(τ)`,
otherwise the value should not be used.

- `ω`: the log weight of the subtree that corresponds to the proposal; valid if `ζ` is.

- `τ`: turn statistics. Only valid when `!isdivergent(d)`.

- `d`: divergence statistics, always valid.

- `z`: the point at the edge of the tree (depending on the direction).
"""
function adjacent_tree(rng, trajectory, z, depth, is_forward)
    if depth == 0
        z = move(trajectory, z, is_forward)
        ζ, ω, τ, d = leaf(trajectory, z, false)
        ζ, ω, τ, d, z
    else
        ζ₋, ω₋, τ₋, d₋, z = adjacent_tree(rng, trajectory, z, depth - 1, is_forward)
        (is_divergent(trajectory, d₋) || (depth > 1 && is_turning(trajectory, τ₋))) &&
            return ζ₋, ω₋, τ₋, d₋, z
        ζ₊, ω₊, τ₊, d₊, z = adjacent_tree(rng, trajectory, z, depth - 1, is_forward)
        d = combine_divergence_statistics(trajectory, d₋, d₊)
        (is_divergent(trajectory, d) || (depth > 1 && is_turning(trajectory, τ₊))) &&
            return ζ₊, ω₋, τ₊, d, z
        τ = combine_turn_statistics_in_direction(trajectory, τ₋, τ₊, is_forward)
        ζ, ω = is_turning(trajectory, τ) ? (ζ₋, ω₋) :
            combine_proposals_and_logweights(rng, trajectory, ζ₋, ζ₊, ω₋, ω₊, is_forward, false)
        ζ, ω, τ, d, z
    end
end

"Reason for terminating a trajectory."
@enum Termination MaxDepth AdjacentDivergent AdjacentTurn DoubledTurn

"""
    ζ, d, termination, depth = sample_trajectory(rng, trajectory, z, max_depth)

Sample a `trajectory` starting at `z`.

Return:

- `ζ`: proposal from the tree

- `d`: divergence statistics

- `termination`: reason for termination (see [`Termination`](@ref))

- `depth`: the depth of the tree that was sampled from. Doubling steps that lead
  to an invalid adjacent tree do not contribute to `depth`.
"""
function sample_trajectory(rng, trajectory, z, max_depth::Integer, directions::Directions)
    @argcheck max_depth ≤ MAX_DIRECTIONS_DEPTH
    ζ, ω, τ, d = leaf(trajectory, z, true)
    z₋ = z₊ = z
    depth = 0
    termination = MaxDepth
    while depth < max_depth
        is_forward, directions = next_direction(directions)
        ζ′, ω′, τ′, d′, z = adjacent_tree(rng, trajectory, is_forward ? z₊ : z₋,
                                          depth, is_forward)
        d = combine_divergence_statistics(trajectory, d, d′)
        is_divergent(trajectory, d) && (termination = AdjacentDivergent; break)
        (depth > 0 && is_turning(trajectory, τ′)) && (termination = AdjacentTurn; break)
        ζ, ω = combine_proposals_and_logweights(rng, trajectory, ζ, ζ′, ω, ω′,
                                                is_forward, true)
        τ = combine_turn_statistics_in_direction(trajectory, τ, τ′, is_forward)
        is_forward ? z₊ = z : z₋ = z
        depth += 1
        is_turning(trajectory, τ) && (termination = DoubledTurn; break)
    end
    ζ, d, termination, depth
end
