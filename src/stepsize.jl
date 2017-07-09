"""
    bracket_zero(f, x, Δ, C; maxiter)

Find `x₁`, `x₂′` that bracket `f(x) = 0`. Return `x₁, x₂′, f(x₁),
f(x₂)′`. `x₁` and `x₂′ are not necessarily ordered.

Algorithm: start at the given `x`, adjust by `Δ` — for increasing `f`,
use `Δ > 0`. At each step, multiply `Δ` by `C`. Stop after `maxiter`
iterations.
"""
function bracket_zero(f, x, Δ, C; maxiter = 50)
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
"""
function find_zero(f, a, b, tol; fa=f(a), fb=f(b), maxiter = 50)
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
                           maxiter_bracket = 50,
                           maxiter_bisection = 50)
    a, fa, b, fb = bracket_zero(f, x, Δ, C; maxiter = maxiter_bracket)
    find_zero(f, a, b, tol; fa=fa, fb = fb, maxiter = maxiter_bisection)
end

"""
    find_reasonable_ϵ(H, z; ϵ, a₀, maxiter)

Let

``A(ϵ) = exp(logdensity(H, leapfrog(H, z, ϵ)) - logdensity(H, z))``,

denote the ratio of densities between a point `z` and another point
after one leapfrog step with stepsize `ϵ`.

Returns an `ϵ` such that `|log(A(ϵ)) - log(a)| ≤ tol`. Uses iterative
bracketing (with gently expanding steps) and rootfinding.
"""
function find_reasonable_ϵ(H, z; tol = 0.5, a = 0.5, ϵ = 1.0, maxiter = 200)
    target = logdensity(H, z) + log(a)
    function residual(logϵ)
        z′ = leapfrog(H, z, exp(logϵ))
        logdensity(H, z′) - target
    end
    exp(bracket_find_zero(residual, log(ϵ), log(0.5), 1.1, tol))
end
