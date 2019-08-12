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

Test if the turn statistics `τ` indicate that the corresponding tree is turning.

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
    $(FUNCTIONNAME)(trajectory, v₁, v₂)

Combine visited node statistics for adjacent trees trajectory. Implementation should be
invariant to the ordering of `v₁` and `v₂` (ie the operation is commutative).
"""
function combine_visited_statistics end

"""
    $(FUNCTIONNAME)(trajectory, is_doubling::Bool, ω₁, ω₂, ω)

Calculate the log probability if selecting the subtree corresponding to `ω₂`. When
`is_doubling`, the tree corresponding to `ω₂` was obtained from a doubling step (this can be
relevant eg for biased progressive sampling).

The value `ω = logaddexp(ω₁, ω₂)` is provided for avoiding redundant calculations.

See [`biased_progressive_logprob2`](@ref) for an implementation.
"""
function calculate_logprob2 end

"""
    $(FUNCTIONNAME)(rng, trajectory, ζ₁, ζ₂, logprob2::Real, is_forward::Bool)

Combine two proposals `ζ₁, ζ₂` on `trajectory`, with log probability `logprob2` for
selecting `ζ₂`.

 `ζ₁` is before `ζ₂` iff `is_forward`.
"""
function combine_proposals end

"""
    ζωτ_or_nothing, v = $(FUNCTIONNAME)(trajectory, z, is_initial)

Information for a tree made of a single node. When `is_initial == true`, this is the first
node.

The first value is either

1. `nothing` for a divergent node,

2. a tuple containing the proposal `ζ`, the log weight (probability) of the node `ω`, the
turn statistics `τ` (never tested as with `is_turning` for leafs).

The second value is the visited node information.
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
    logprob2 = calculate_logprob2(trajectory, is_doubling, ω₁, ω₂, ω)
    ζ = combine_proposals(rng, trajectory, ζ₁, ζ₂, logprob2, is_forward)
    ζ, ω
end

"""
$(SIGNATURES)

Given (relative) log probabilities `ω₁` and `ω₂`, return the log probabiliy of
drawing a sample from the second (`logprob2`).

When `bias`, biases towards the second argument, introducing anti-correlations.
"""
function biased_progressive_logprob2(bias::Bool, ω₁::Real, ω₂::Real, ω = logaddexp(ω₁, ω₂))
    ω₂ - (bias ? ω₁ : ω)
end

####
#### abstract trajectory interface
####

"""
$(SIGNATURES)

Information about an invalid (sub)tree, using positions relative to the starting node.

1. When `left < right`, this tree was *turning*.

2. When `left == right`, this is a *divergent* node.

3. `left == 1 && right == 0` is used as a sentinel value for reaching maximum depth without
encountering any invalid trees (see [`REACHED_MAX_DEPTH`](@ref). All other `left > right`
values are disallowed.
"""
struct InvalidTree
    left::Int
    right::Int
end

InvalidTree(i::Integer) = InvalidTree(i, i)

is_divergent(invalid_tree::InvalidTree) = invalid_tree.left == invalid_tree.right

function Base.show(io::IO, invalid_tree::InvalidTree)
    msg = if is_divergent(invalid_tree)
        "divergence at position $(invalid_tree.left)"
    elseif invalid_tree == REACHED_MAX_DEPTH
        "reached maximum depth without divergence or turning"
    else
        @unpack left, right = invalid_tree
        "turning at positions $(left):$(right)"
    end
    print(io, msg)
end

"Sentinel value for reaching maximum depth."
const REACHED_MAX_DEPTH = InvalidTree(1, 0)

"""
    result, v = adjacent_tree(rng, trajectory, z, i, depth, is_forward)

Traverse the tree of given `depth` adjacent to point `z` in `trajectory`.

`is_forward` specifies the direction, `rng` is used for random numbers in
[`combine_proposals`](@ref). `i` is an integer position relative to the initial node (`0`).

The *first value* is either

1. an `InvalidTree`, indicating the first divergent node or turning subtree that was
encounteted and invalidated this tree.

2. a tuple of `(ζ, ω, τ, z′, i′), with

    - `ζ`: the proposal from the tree.

    - `ω`: the log weight of the subtree that corresponds to the proposal

    - `τ`: turn statistics

    - `z′`: the last node of the tree

    - `i′`: the position of the last node relative to the initial node.

The *second value* is always the visited node statistic.
"""
function adjacent_tree(rng, trajectory, z, i, depth, is_forward)
    i′ = i + (is_forward ? 1 : -1)
    if depth == 0
        z′ = move(trajectory, z, is_forward)
        ζωτ, v = leaf(trajectory, z′, false)
        if ζωτ ≡ nothing
            InvalidTree(i′), v
        else
            (ζωτ..., z′, i′), v
        end
    else
        # “left” tree
        t₋, v₋ = adjacent_tree(rng, trajectory, z, i, depth - 1, is_forward)
        t₋ isa InvalidTree && return t₋, v₋
        ζ₋, ω₋, τ₋, z₋, i₋ = t₋

        # “right” tree — visited information from left is kept even if invalid
        t₊, v₊ = adjacent_tree(rng, trajectory, z₋, i₋, depth - 1, is_forward)
        v = combine_visited_statistics(trajectory, v₋, v₊)
        t₊ isa InvalidTree && return t₊, v
        ζ₊, ω₊, τ₊, z₊, i₊ = t₊

        # turning invalidates
        τ = combine_turn_statistics_in_direction(trajectory, τ₋, τ₊, is_forward)
        is_turning(trajectory, τ) && return InvalidTree(i′, i₊), v

        # valid subtree, combine proposals
        ζ, ω = combine_proposals_and_logweights(rng, trajectory, ζ₋, ζ₊, ω₋, ω₊, is_forward, false)
        (ζ, ω, τ, z₊, i₊), v
    end
end

"""
$(SIGNATURES)

Sample a `trajectory` starting at `z`, up to `max_depth`. `directions` determines the tree
expansion directions.

Return the following values

- `ζ`: proposal from the tree

- `v`: visited node statistics

- `termination`: an `InvalidTree` (this includes the last doubling step turning, which is
  technically a valid tree) or `REACHED_MAX_DEPTH` when all subtrees were valid and no
  turning happens.

- `depth`: the depth of the tree that was sampled from. Doubling steps that lead to an
  invalid adjacent tree do not contribute to `depth`.
"""
function sample_trajectory(rng, trajectory, z, max_depth::Integer, directions::Directions)
    @argcheck max_depth ≤ MAX_DIRECTIONS_DEPTH
    (ζ, ω, τ), v = leaf(trajectory, z, true)
    z₋ = z₊ = z
    depth = 0
    termination = REACHED_MAX_DEPTH
    i₋ = i₊ = 0
    while depth < max_depth
        is_forward, directions = next_direction(directions)
        t′, v′ = adjacent_tree(rng, trajectory, is_forward ? z₊ : z₋, is_forward ? i₊ : i₋,
                               depth, is_forward)
        v = combine_visited_statistics(trajectory, v, v′)

        # invalid adjacent tree: stop
        t′ isa InvalidTree && (termination = t′; break)

        # extract information from adjacent tree
        ζ′, ω′, τ′, z′, i′ = t′

        # update edges and combine proposals
        if is_forward
            z₊, i₊ = z′, i′
        else
            z₋, i₋ = z′, i′
        end

        # tree has doubled successfully
        ζ, ω = combine_proposals_and_logweights(rng, trajectory, ζ, ζ′, ω, ω′,
                                                is_forward, true)
        depth += 1

        # when the combined tree is turning, stop
        τ = combine_turn_statistics_in_direction(trajectory, τ, τ′, is_forward)
        is_turning(trajectory, τ) && (termination = InvalidTree(i₋, i₊); break)
    end
    ζ, v, termination, depth
end
