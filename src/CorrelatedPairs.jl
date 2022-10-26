@doc raw"""
    Correlated Pair Distribution

"""
# module CorrelatedPairs

# export
#     CorrelatedPair,
#     _rand!

using Random, Distributions

struct CorrelatedPairSampler{T<:UnivariateDistribution} <: Sampleable{Multivariate,Continuous}

    # Base distribution
    base_dist::T

    # Stored values of sin and cos of θ
    sinθ::Real
    cosθ::Real

    # Inner constructor
    function CorrelatedPairSampler{T}(dist::T, ρ::Real) where {T<:UnivariateDistribution}
        θ = 0.5 * asin(ρ)
        return new{T}(dist, sin(θ), cos(θ))
    end

end

function CorrelatedPairSampler(dist::T, ρ::Real; check_args::Bool=true) where {T<:UnivariateDistribution}
    Distributions.@check_args(
        CorrelatedPairSampler,
        (ρ, -one(ρ) <= ρ <= one(ρ), "Correlation must be in the interval ρ ∈ [-1,1]."),
        (mean(dist) == zero(eltype(dist)), "Base distribution must be centered."),
        (var(dist) == one(eltype(dist)), "Base distribution must have unit variance."),
    )
    return CorrelatedPairSampler{T}(dist, ρ)
end

# function set_correlation!(corr_pair::CorrelatedPairSampler, ρ::Real)
#     θ = 0.5 * asin(ρ)
#     corr_pair.sinθ, corr_pair.cosθ = sin(θ), cos(θ)
# end

@inline Base.length(::CorrelatedPairSampler) = 2

function Distributions._rand!(rng::AbstractRNG, corr_pair::CorrelatedPairSampler, x::AbstractVector{<:Real})
    # Sample uncorrelated pair
    rand!(rng, corr_pair.base_dist, x)
    # Create correlated pair
    temp = x[1]
    x[1] = x[1] * corr_pair.sinθ + x[2] * corr_pair.cosθ
    x[2] = temp * corr_pair.cosθ + x[2] * corr_pair.sinθ
    return x
end

corr_pair = CorrelatedPairSampler(Normal(), 1.0)

# end
