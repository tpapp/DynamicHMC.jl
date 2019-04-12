module Trajectories

export Buffers, initialize_buffers!, Visited, initialize_visited!, random_direction_flags,
    build_adjacent!, transition!

using ArgCheck: @argcheck
using DocStringExtensions: FIELDS, FUNCTIONNAME, SIGNATURES, TYPEDEF
using StatsFuns: logaddexp
using Parameters: @unpack

####
#### utilities
####

"""
Upper bound for the MAX_DEPTH supported by these routines.

This is required because we use an integer to store the bits.
"""
const SUPPORTED_MAX_DEPTH = 32

"""
$(SIGNATURES)

Random Metropolis acceptance with log proposal ratio `Δ`.

Will only generate a random number if necessary.
"""
accept_Δ(rng, Δ) = Δ ≥ 0 || rand(rng) < exp(Δ)

"""
$(SIGNATURES)

Generate random direction flags, returned as an integer.

`max_depth` is limited to [`SUPPORTED_MAX_DEPTH`](@ref).
"""
function random_direction_flags(rng, max_depth)
    @argcheck 0 < max_depth ≤ SUPPORTED_MAX_DEPTH
    rand(rng, UInt32)
end

####
#### exposed API for trajectories
####

"""
$(FUNCTIONNAME)(z, trajectory, is_forward)

Update phasepoint `z` on `trajectory`, moving forward or backward as determined by the
direction argument. Modifies `z`, returned value is unused.

Buffers for `z` can be allocated with [`empty_z`](@ref).
"""
function move! end

"""
$(FUNCTIONNAME)(trajectory, z)

Return the log probability of `z` on `trajectory`.
"""
function log_probability(trajectory, z) end

"""
$(FUNCTIONNAME)(p♯, trajectory, z)

Calculate `p♯`, a statistic related to `z`, useful for evaluating turning, on the
`trajectory`.

Buffers for `p♯` can be allocated with [`empty_p♯`](@ref).
"""
function get_p♯! end

"""
$(FUNCTIONNAME)(ρ, trajectory, z)

Calculate `ρ`, a statistic related to `z`, useful for evaluating turning, on the
`trajectory`. This is used for leafs on the tree, and then added with [`add_ρ!`](@ref).

Buffers for `ρ` can be allocated with [`empty_ρ`](@ref).
"""
function get_ρ!(ρ, trajectory, z) end

"""
$(FUNCTIONNAME)(ρ, trajectory, ρ′)

Combine `ρ` and `ρ′` into `ρ`, overwriting the latter.

Buffers for `ρ` can be allocated with [`empty_ρ`](@ref).
"""
function add_ρ! end

"""
$(FUNCTIONNAME)(trajectory)

Allocate a mutable buffer for phasepoints `z`.
"""
function empty_z end

"""
$(FUNCTIONNAME)(trajectory)

Allocate a mutable buffer for tree turn statistic `ρ`.
"""
function empty_ρ end

"""
$(FUNCTIONNAME)(trajectory)

Allocate a mutable buffer for phasepoint turn statistic `p♯`.
"""
function empty_p♯ end

"""
$(FUNCTIONNAME)(trajectory, ρ, p♯₋, p♯₊)

Test if the relevant section of the tree is *turning* as defined by the turn statistics.
"""
function is_turning end

####
#### Pre-allocated buffers
####

"""
$(TYPEDEF)

# Fields

$(FIELDS)
"""
struct Buffers{Z,R,P}
    z₋::Z
    z₊::Z
    ẑ::Z
    ẑs::Vector{Z}
    p♯₋::P
    p♯₊::P
    p♯s::Vector{P}
    ρ::R
    ρs::Vector{R}
end

function Buffers(trajectory, max_depth::Integer)
    @argcheck 0 < max_depth ≤ SUPPORTED_MAX_DEPTH
    _vec(f) = [f(trajectory) for _ in 1:(max_depth + 1)]
    Buffers(empty_z(trajectory), empty_z(trajectory), empty_z(trajectory), _vec(empty_z),
            empty_p♯(trajectory), empty_p♯(trajectory), _vec(empty_p♯),
            empty_ρ(trajectory), _vec(empty_ρ))
end

"""
$(SIGNATURES)

Initialize `buffers` from the point `z`, for sampling a trajectory.
"""
function initialize_buffers!(buffers, trajectory, z)
    @unpack z₋, z₊, ẑ, p♯₋, p♯₊, ρ = buffers
    copy!(z₋, z)
    copy!(z₊, z)
    copy!(ẑ, z)
    get_p♯!(p♯₋, trajectory, z)
    copy!(p♯₊, p♯₋)
    get_ρ!(ρ, trajectory, z)
    nothing
end

mutable struct Visited{T <: AbstractFloat}
    "Number of visited nodes (other than the initial one)"
    leapfrog_steps::Int
    "Log of sum ``min(1, exp(Δ))`` for all visited nodes (other than the initial one)."
    log∑a::T
end

Visited(::Type{T}) where {T <: AbstractFloat} = Visited(0, zero(T))

Visited() = Visited(Float64)

"""
$(SIGNATURES)

Initialize the visited leaf statistics `visited`, for sampling a trajectory.
"""
function initialize_visited!(visited::Visited)
    visited.leapfrog_steps = 0
    visited.log∑a = 0
    nothing
end

"""
$(SIGNATURES)

Add `Δ` from a visited node.
"""
function add_Δ!(visited::Visited, Δ)
    visited.leapfrog_steps += 1
    visited.log∑a = logaddexp(visited.log∑a, min(0, Δ))
    nothing
end

"""
$(SIGNATURES)

Build a tree of the given `depth` on `trajectory`, adjacent to `z₋` or `z₊` in buffers, in
the direction specified by `is_forward`.

`π₀` is the log probability at the initial point and is used to calculate the difference `Δ`
for each phasepoint.

# Return values

1. When `Δ < min_Δ`, anywhere in the tree, a phasepoint is considered *divergent* and `-Inf`
is returned.

2. When a turning subtree is encountered, `NaN` is returned.

3. Otherwise, the returned value is the *weight* ``log(∑ exp(Δ))`` for the whole tree that
was built.

# Buffers and invariants

Acceptance information on all visited notes is accumulated in `visited`, regardless of
divergence and turning.

Buffers `buffer_index:(buffer_index + depth)` may be overwritten, and assumed to be
available for this purpose.

When the return value is *finite*,

1. `buffers.ẑs[buffer_index]` contains the proposal,

2. `buffers.ρs[buffer_index]` contains the sum of the `ρ`s from all nodes,

3. `buffers.p♯s[buffer_index]` contain the `p♯` for the endpoint of the tree (farthest from
the initial edge).
"""
function build_adjacent!(rng, trajectory, min_Δ, π₀, depth::Int, is_forward::Bool,
                         visited::Visited, buffers::Buffers, buffer_index::Int)
    @unpack z₋, z₊, ẑs, p♯s, ρs = buffers
    if depth == 0
        z = is_forward ? z₊ : z₋
        move!(z, trajectory, is_forward)
        Δ = log_probability(trajectory, z) - π₀
        # always update acceptance statistics (even for divergent)
        add_Δ!(visited, Δ)
        # when divergent, return -∞
        Δ < min_Δ && return oftype(Δ, -Inf)
        # propose this leaf for the trivial subtree
        copy!(ẑs[buffer_index], z)
        # obtain turn statistics for leaf
        get_p♯!(p♯s[buffer_index], trajectory, z)
        get_ρ!(ρs[buffer_index], trajectory, z)
        # return Δ
        Δ
    else
        w₁ = build_adjacent!(rng, trajectory, min_Δ, π₀, depth - 1, is_forward,
                             visited, buffers, buffer_index)
        isfinite(w₁) || return w₁
        w₂ = build_adjacent!(rng, trajectory, min_Δ, π₀, depth - 1, is_forward,
                             visited, buffers, buffer_index + 1)
        isfinite(w₂) || return w₂
        w = logaddexp(w₁, w₂)
        accept_Δ(rng, w₂ - w) && copy!(ẑs[buffer_index], ẑs[buffer_index + 1])
        add_ρ!(ρs[buffer_index], trajectory, ρs[buffer_index + 1])
        if is_turning(trajectory, ρs[buffer_index],
                      p♯s[buffer_index], p♯s[buffer_index + 1])
            oftype(w, NaN)
        else
            copy!(p♯s[buffer_index], p♯s[buffer_index + 1])
            w
        end
    end
end

function transition!(rng, trajectory, min_Δ, max_depth, direction_flags, z, visited, buffers)
    @unpack z₋, z₊, ẑ, ẑs, p♯₋, p♯₊, p♯s, ρ, ρs = buffers
    # initialization
    depth = 0
    initialize_visited!(visited)
    initialize_buffers!(buffers, trajectory, z)
    w = 0.0
    π₀ = log_probability(trajectory, z)
    # doubling
    while depth < max_depth
        is_forward = Bool(direction_flags & true)
        w′ = build_adjacent!(rng, trajectory, min_Δ, π₀, depth, is_forward,
                             visited, buffers, 1)
        isfinite(w′) || break   # adjacent tree is divergent or turning
        accept_Δ(rng, w′ - w) && copy!(ẑ, ẑs[1]) # accelerated
        add_ρ!(ρ, trajectory, ρs[1])
        copy!(is_forward ? p♯₊ : p♯₋, p♯s[1])
        is_turning(trajectory, ρ, p♯₋, p♯₊) && break
        direction_flags >>= 1
        depth += 1
    end
    depth
end

end
