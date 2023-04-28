@doc raw"""
    Statistics utils

"""
module Stats

export
    Histogram,
    CorrelatedPairSampler,
    CorrelatedTimeSeriesMatrixSampler

using Statistics, Random, Distributions, LinearAlgebra

struct Histogram{T<:Real}
    edges::AbstractRange{T}
    freqs::AbstractVector{UInt64}
end

@inline function Histogram(values::AbstractVector{T}, nbins::Integer) where {T<:Real}
    low, high = extrema(values)

    edges = range(low, high, length=nbins + 1)

    freqs = zeros(UInt64, nbins)
    bin_width = (high - low) / nbins
    for x ∈ values
        idx = min(floor(UInt64, (x - low) / bin_width) + 1, nbins)
        freqs[idx] += 1
    end

    Histogram{T}(edges, freqs)

end

@inline Statistics.mean(hist::Histogram) =
    let n = sum(hist.freqs)
        sum(e * (f / n) for (e, f) ∈ zip(hist.edges, hist.freqs))
    end


@inline function Statistics.var(hist::Histogram; corrected::Bool=true, mean=nothing)
    mean_used = if mean === nothing
        Statistics.mean(hist)
    elseif isa(mean, Real)
        mean
    else
        throw(ArgumentError("invalid value of mean, $(mean)::$(typeof(mean))"))
    end

    n = sum(hist.freqs)
    return sum(abs2(e - mean_used) * (f / (n - Int(corrected))) for (e, f) ∈ zip(hist.edges, hist.freqs))

end

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
    function CorrelatedPairSampler{T}(ρ::Real, dist::T=Normal()) where {T<:UnivariateDistribution}
        θ = 0.5 * asin(ρ)
        return new{T}(dist, sin(θ), cos(θ))
    end

end

function CorrelatedPairSampler(ρ::Real, dist::T=Normal(); check_args::Bool=true) where {T<:UnivariateDistribution}
    Distributions.@check_args(
        CorrelatedPairSampler,
        (ρ, -one(ρ) <= ρ <= one(ρ), "Correlation must be in the interval ρ ∈ [-1,1]."),
        (mean(dist) == zero(eltype(dist)), "Base distribution must be centered."),
        (var(dist) == one(eltype(dist)), "Base distribution must have unit variance."),
    )
    return CorrelatedPairSampler{T}(ρ, dist)
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
    # Length of each time series
    t_max::Integer
    # Number of time series pairs
    n_pairs::Integer

end

CorrelatedTimeSeriesMatrixSampler(ρ::Real, t_max::Integer, n_pairs::Integer, dist::UnivariateDistribution=Normal()) =
    CorrelatedTimeSeriesMatrixSampler(CorrelatedPairSampler(ρ, dist), t_max, n_pairs)

@inline Base.size(s::CorrelatedTimeSeriesMatrixSampler) = (s.t_max, 2 * s.n_pairs)

function Distributions._rand!(rng::AbstractRNG, s::CorrelatedTimeSeriesMatrixSampler, x::DenseMatrix{T}) where {T<:Real}
    @views foreach(ts_pair -> rand!(rng, s.corr_pair, ts_pair'), x[:, 2*k-1:2*k] for k ∈ 1:s.n_pairs)
    return x
end

end
