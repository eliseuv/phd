@doc raw"""
    Correlated Pair Distribution

"""
module CorrelatedPairs

export
    CorrelatedPairSampler,
    CorrelatedTimeSeriesMatrixSampler

using Random, Distributions, LinearAlgebra

@doc raw"""
    CorrelatedPairSampler{T<:UnivariateDistribution} <: Sampleable{Multivariate,Continuous}

Multivariate sampler that generates pair of correlated random variables with a given correlation `ρ`
following a given centered and unit variance distributions.

# Fields:
- `base_dist::T`: base distribution from which to sample the original uncorrelated variables.
- `sinθ::Real` and `cosθ::Real`: stores auxiliary values needed for pair generation.
"""
struct CorrelatedPairSampler{T<:UnivariateDistribution} <: Sampleable{Multivariate,Continuous}

    # Base distribution
    base_dist::T

    # Stored values of sin and cos of θ
    sinθ::Real
    cosθ::Real

    # Inner constructor
    function CorrelatedPairSampler{T}(d::T, ρ::Real) where {T<:UnivariateDistribution}
        θ = 0.5 * asin(ρ)
        return new{T}(d, sin(θ), cos(θ))
    end

end

function CorrelatedPairSampler(d::T, ρ::Real; check_args::Bool=true) where {T<:UnivariateDistribution}
    Distributions.@check_args(
        CorrelatedPairSampler,
        (ρ, -one(ρ) <= ρ <= one(ρ), "Correlation must be in the interval ρ ∈ [-1,1]."),
        (mean(d) == zero(eltype(d)), "Base distribution must be centered."),
        (var(d) == one(eltype(d)), "Base distribution must have unit variance."),
    )
    return CorrelatedPairSampler{T}(d, ρ)
end

# function set_correlation!(corr_pair::CorrelatedPairSampler, ρ::Real)
#     θ = 0.5 * asin(ρ)
#     corr_pair.sinθ, corr_pair.cosθ = sin(θ), cos(θ)
# end

@doc raw"""
    Base.length(::CorrelatedPairSampler)

The vector length of the multivariate sampler `CorrelatedPairSampler`.

Since it generates a pair of random variables, its length is `2`.
"""
@inline Base.length(::CorrelatedPairSampler) = 2


@doc raw"""
    Distributions._rand!(rng::AbstractRNG, s::CorrelatedPairSampler, x::AbstractVector{<:Real})

Generates a sample inplace from the multivariate sampler `CorrealatedPairSampler`.
"""
function Distributions._rand!(rng::AbstractRNG, s::CorrelatedPairSampler, x::AbstractVector{T}) where {T<:Real}
    # Sample uncorrelated pair
    rand!(rng, s.base_dist, x)
    # Create correlated pair
    temp = x[1]
    x[1] = x[1] * s.sinθ + x[2] * s.cosθ
    x[2] = temp * s.cosθ + x[2] * s.sinθ
    return x
end

@doc raw"""
    CorrelatedTimeSeriesMatrixSampler <: Sampleable{Matrixvariate,Continuous}

Matrix-variate sampler that generates time series matrices consisting of pairs of time series whose entries are generated using a correlated pair generator.
"""
struct CorrelatedTimeSeriesMatrixSampler <: Sampleable{Matrixvariate,Continuous}

    # Correlated pair generator
    corr_pair::CorrelatedPairSampler
    # Number of time series pairs
    n_pairs::Integer
    # Length of each time series
    t_max::Integer

end

CorrelatedTimeSeriesMatrixSampler(dist::UnivariateDistribution, ρ::Real, n_pairs::Integer, t_max::Integer) =
    CorrelatedTimeSeriesMatrixSampler(CorrelatedPairSampler(dist, ρ), n_pairs, t_max)

@inline Base.size(s::CorrelatedTimeSeriesMatrixSampler) = (s.t_max, 2 * s.n_pairs)

function Distributions._rand!(rng::AbstractRNG, s::CorrelatedTimeSeriesMatrixSampler, x::DenseMatrix{T}) where {T<:Real}
    @views foreach(ts_pair -> rand!(rng, s.corr_pair, ts_pair'), x[:, 2*k-1:2*k] for k ∈ 1:s.n_pairs)
    return x
end

end
