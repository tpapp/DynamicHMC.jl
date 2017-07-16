export
    find_reasonable_logϵ,
    adapt,
    DualAveragingParameters, DualAveragingAdaptation, adapting_logϵ,
    FixedStepSize, fixed_logϵ

const MAXITER_BRACKET = 50
const MAXITER_BISECTION = 50

"""
    bracket_zero(f, x, Δ, C; maxiter)

Find `x₁`, `x₂′` that bracket `f(x) = 0`. `f` should be monotone, use
`Δ > 0` for increasing and `Δ < 0` decreasing `f`.

Return `x₁, x₂′, f(x₁), f(x₂)′`. `x₁` and `x₂′ are not necessarily
ordered.

Algorithm: start at the given `x`, adjust by `Δ` — for increasing `f`,
use `Δ > 0`. At each step, multiply `Δ` by `C`. Stop and throw an
error after `maxiter` iterations.
"""
function bracket_zero(f, x, Δ, C; maxiter = MAXITER_BRACKET)
    @argcheck C > 1
    @argcheck Δ ≠ 0
    fx = f(x)
    s = sign(fx)
    for _ in 1:maxiter
        x′ = x - s*Δ            # note: in the unlikely case s=0 ...
        fx′ = f(x′)
        if s*fx′ ≤ 0            # ... should still work
            return x, fx, x′, fx′
        else
            if abs(fx′) > abs(fx)
                warn("Residual increased, function may not be monotone.")
            end
            Δ *= C
            x = x′
            fx = fx′
        end
    end
    error("Reached maximum number of iterations without crossing 0.")
end

"""
    find_zero(f, a, b, tol; fa, fb, maxiter)

Use bisection to find ``x ∈ [a,b]`` such that `|f(x)| < tol`. When `f`
is costly, specify `fa` and `fb`.

When does not converge within `maxiter` iterations, throw an error.
"""
function find_zero(f, a, b, tol; fa=f(a), fb=f(b), maxiter = MAXITER_BISECTION)
    @argcheck fa*fb ≤ 0 "Initial values don't bracket the root."
    @argcheck tol > 0
    for _ in 1:maxiter
        x = middle(a,b)
        fx = f(x)
        if abs(fx) ≤ tol
            return x
        elseif fx*fa > 0
            a = x
            fa = fx
        else
            b = x
            fb = fx
        end
    end
    error("Reached maximum number of iterations.")
end

function bracket_find_zero(f, x, Δ, C, tol;
                           maxiter_bracket = MAXITER_BRACKET,
                           maxiter_bisection = MAXITER_BISECTION)
    a, fa, b, fb = bracket_zero(f, x, Δ, C; maxiter = maxiter_bracket)
    find_zero(f, a, b, tol; fa=fa, fb = fb, maxiter = maxiter_bisection)
end

"""
    find_reasonable_logϵ(H, z; tol, a, ϵ, maxiter)

Let

``A(ϵ) = exp(logdensity(H, leapfrog(H, z, ϵ)) - logdensity(H, z))``,

denote the ratio of densities between a point `z` and another point
after one leapfrog step with stepsize `ϵ`.

Returns an `ϵ` such that `|log(A(ϵ)) - log(a)| ≤ tol`. Uses iterative
bracketing (with gently expanding steps) and rootfinding.

Starts at `ϵ`, uses `maxiter` iterations for the bracketing and the
rootfinding, respectively.
"""
function find_reasonable_logϵ(H, z; tol = 0.15, a = 0.75, ϵ = 1.0,
                              maxiter_bracket = MAXITER_BRACKET,
                              maxiter_bisection = MAXITER_BISECTION)
    target = logdensity(H, z) + log(a)
    function residual(logϵ)
        z′ = leapfrog(H, z, exp(logϵ))
        logdensity(H, z′) - target
    end
    bracket_find_zero(residual, log(ϵ), log(0.5), 1.1, tol;
                      maxiter_bracket = MAXITER_BRACKET,
                      maxiter_bisection = MAXITER_BISECTION)
end

"""
Parameters for the dual averaging algorithm of Gelman and Hoffman
(2014, Algorithm 6).

To get reasonable defaults, initialize with
`DualAveragingParameters(logϵ₀)`.
"""
struct DualAveragingParameters{T}
    μ::T
    δ::T
    γ::T
    κ::T
    t₀::Int
end

DualAveragingParameters(logϵ₀; δ = 0.65, γ = 0.05, κ = 0.75, t₀ = 10) =
    DualAveragingParameters(promote(log(10)+logϵ₀, δ, γ, κ)..., t₀)

"Current state of adaptation for `ϵ`. Use
`DualAverageingAdaptation(logϵ₀)` to get an initial value."
struct DualAveragingAdaptation{T <: AbstractFloat}
    m::Int
    H̄::T
    logϵ::T
    logϵ̄::T
end

DualAveragingAdaptation(logϵ₀) =
    DualAveragingAdaptation(0, zero(logϵ₀), logϵ₀, zero(logϵ₀))

function adapting_logϵ(logϵ; args...)
    DualAveragingParameters(logϵ; args...), DualAveragingAdaptation(logϵ)
end

"""
    A′ = adapt(parameters, A, a)

Update the adaptation `A` of log stepsize `logϵ` with acceptance rate
`a`, using the dual averaging algorithm of Gelman and Hoffman (2014,
Algorithm 6). Return the new adaptation.
"""
function adapt(parameters::DualAveragingParameters, A::DualAveragingAdaptation, a)
    @unpack μ, δ, γ, κ, t₀ = parameters
    @unpack m, H̄, logϵ, logϵ̄ = A
    m += 1
    H̄ += (δ - a - H̄) / (m+t₀)
    logϵ = μ - √m/γ*H̄
    logϵ̄ += m^(-κ)*(logϵ - logϵ̄)
    DualAveragingAdaptation(m, H̄, logϵ, logϵ̄)
end

struct FixedStepSize{T}
    logϵ::T
end

adapt(::Any, A::FixedStepSize, a) = A

fixed_logϵ(logϵ) = nothing, FixedStepSize(logϵ)
