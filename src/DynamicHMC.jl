"""
Notation follows Betancourt (2017), with some differences

## Density
- `q` is the parameter vector
- `π(q)` is the density we are sampling from
- V(q) = -log(π(q)) ≡ -ℓ(q), in practice we use ℓ

## Momentym
- p is the momentum
- K(p,q) = -log π(p|q) is minus log conditional density for the momentum
- in practice we use ω for the momentum
"""
module DynamicHMC

import Base: rand

export
    logdensity,
    loggradient,
    UnitNormal,
    UNITNORMAL,
    minuslogH,
    leapfrog

"""
Unit normal momentum. Efficient when the distribution has been
decorrelated.

p|q ∼ MultivariateNormal(0,I) (independently of `q`)
"""
struct UnitNormal end

UNITNORMAL = UnitNormal()

logdensity(::UnitNormal, p, q) = -sum(abs2, p)/2

rand(::UnitNormal, q) = randn(length(q))

"Jacobian of the log density."
function loggradient end

function leapfrog(ℓ, ω::UnitNormal, q, p, ϵ)
    p₊ = p + ϵ/2 * loggradient(ℓ, q)
    q′ = q + ϵ * p₊
    p′ = p₊ + ϵ/2 * loggradient(ℓ, q′)
    q′, p′
end

minuslogH(ℓ, ω, q, p) = logdensity(ω, p, q) + logdensity(ℓ, q)

end # module
