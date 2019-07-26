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
    $(FUNCTIONNAME)(rng, trajectory, ζ₁, ζ₂, is_forward::Bool, is_doubling::Bool)

Combine two proposals `ζ₁, ζ₂` on `trajectory`. `ζ₁` is before `ζ₂` iff `is_forward`.

When `is_doubling`, `ζ₂` was obtained from a doubling step (this can be relevant eg for
biased progressive sampling).
"""
function combine_proposals end

"""
    ζ, τ, d = $(FUNCTIONNAME)(trajectory, z, is_initial)

Proposal, turn statistics, and divergence statistics for a tree made of a single node. When
`is_initial == true`, this is the first node.
"""
function leaf end

####
#### utilities
####

"""
$(SIGNATURES)

Random boolean which is `true` with the given probability `prob`.

**All random numbers in this library are obtained from this function.**
"""
rand_bool(rng::AbstractRNG, prob::T) where {T <: AbstractFloat} =
    rand(rng, T) ≤ prob

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

- `τ`: turn statistics. Only valid when `!isdivergent(d)`.

- `d`: divergence statistics, always valid.

- `z`: the point at the edge of the tree (depending on the direction).
"""
function adjacent_tree(rng, trajectory, z, depth, is_forward)
    if depth == 0
        z = move(trajectory, z, is_forward)
        ζ, τ, d = leaf(trajectory, z, false)
        ζ, τ, d, z
    else
        ζ₋, τ₋, d₋, z = adjacent_tree(rng, trajectory, z, depth - 1, is_forward)
        (is_divergent(trajectory, d₋) || (depth > 1 && is_turning(trajectory, τ₋))) &&
            return ζ₋, τ₋, d₋, z
        ζ₊, τ₊, d₊, z = adjacent_tree(rng, trajectory, z, depth - 1, is_forward)
        d = combine_divergence_statistics(trajectory, d₋, d₊)
        (is_divergent(trajectory, d) || (depth > 1 && is_turning(trajectory, τ₊))) &&
            return ζ₊, τ₊, d, z
        τ = combine_turn_statistics_in_direction(trajectory, τ₋, τ₊, is_forward)
        ζ = is_turning(trajectory, τ) ? nothing :
            combine_proposals(rng, trajectory, ζ₋, ζ₊, is_forward, false)
        ζ, τ, d, z
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
    ζ, τ, d = leaf(trajectory, z, true)
    z₋ = z₊ = z
    depth = 0
    termination = MaxDepth
    while depth < max_depth
        is_forward, directions = next_direction(directions)
        ζ′, τ′, d′, z = adjacent_tree(rng, trajectory, is_forward ? z₊ : z₋,
                                      depth, is_forward)
        d = combine_divergence_statistics(trajectory, d, d′)
        is_divergent(trajectory, d) && (termination = AdjacentDivergent; break)
        (depth > 0 && is_turning(trajectory, τ′)) && (termination = AdjacentTurn; break)
        ζ = combine_proposals(rng, trajectory, ζ, ζ′, is_forward, true)
        τ = combine_turn_statistics_in_direction(trajectory, τ, τ′, is_forward)
        is_forward ? z₊ = z : z₋ = z
        depth += 1
        is_turning(trajectory, τ) && (termination = DoubledTurn; break)
    end
    ζ, d, termination, depth
end
