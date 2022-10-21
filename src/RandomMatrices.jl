module RandomMatrices

export
    correlated_pair, correlated_ts_matrix,
    normalize_ts_matrix, normalize_ts_matrix!,
    shuffle_cols!,
    cross_correlation_matrix

using Base: permutecols!!
using Random, Statistics, Distributions, LinearAlgebra

struct CorrelatedPair{S<:ValueSupport} <: Sampleable{Multivariate,S}

    ρ::Real
    base_dist::Distribution{Univariate,S}

    CorrelatedPair(ρ::Real, dist::Distribution{Univariate,S}) where {S<:ValueSupport} = new{S}(ρ, dist)

end

@inline Base.length(::CorrelatedPair) = 2

@inline function _rand!(rng::AbstractRNG, corr_pair::CorrelatedPair, x::AbstractVector{T}) where {T<:Real}
    # Uncorrelated pair
    rand!(rng, corr_pair.base_dist, x)
    # Create correlated pair
    θ = 0.5 * asin(corr_pair.ρ)
    x = [x[1] * sin(θ) + x[2] * cos(θ),
        x[1] * cos(θ) + x[2] * sin(θ)]
end

@doc raw"""
    correlated_pair(ρ::Real, dist::Distribution=Normal())

Generates a pair of correlated random variables.
"""
@inline function correlated_pair(ρ::Real, dist::Distribution=Normal())
    # Uncorrelated pair
    φ = rand(dist, 2)
    # Create correlated pair
    θ = 0.5 * asin(ρ)
    ϕ = (φ[1] * sin(θ) + φ[2] * cos(θ),
        φ[1] * cos(θ) + φ[2] * sin(θ))
    return ϕ
end

"""
    correlated_ts_matrix(ρ::Real, M::Integer=2, n_pairs::Integer=1, dist::Distribution=Normal())

Create a matrix consisting
"""
@inline correlated_ts_matrix(ρ::Real, t_max::Integer=2, n_pairs::Integer=1, dist::Distribution=Normal()) =
    hcat([vcat(map(ϕ -> [ϕ[1] ϕ[2]], correlated_pair(ρ, dist) for _ in 1:t_max)...) for _ in 1:n_pairs]...)

@doc raw"""
    normalize_ts_matrix(M_ts::AbstractMatrix)

Returns a new normalized version version of a time series matrix `M_ts` by normalizing each of its columns (time series samples).

Its `j`-th column (`mⱼ`) becomes:

    ``\frac{ m_j - ⟨m_j⟩ }{ √{ ⟨m_j^2⟩ - ⟨m_j⟩^2 } }``

# Arguments:
- `M_ts::AbstractMatrix`: `N×M` Matrix whose each of its `M` columns corresponds to a sample of a time series `Xₜ` of length `N`.
"""
@inline normalize_ts_matrix(M_ts::AbstractMatrix) = hcat(
    map(eachcol(M_ts)) do xⱼ
        xⱼ_avg = mean(xⱼ)
        (xⱼ .- xⱼ_avg) ./ stdm(xⱼ, xⱼ_avg, corrected=true)
    end...)


"""
    shuffle_cols!(M_ts::AbstractMatrix)

TBW
"""
@inline function shuffle_cols!(M_ts::AbstractMatrix)
    permutecols!!(M_ts, randperm(size(M_ts, 2)))
end

@doc raw"""
    normalize_ts_matrix!(M_ts::AbstractMatrix)

Normalizes *inplace* a given time series matrix `M_ts` by normalizing each of its columns (time series samples).

Its `j`-th column (`mⱼ`) becomes:

    ``\frac{ m_j - ⟨m_j⟩ }{ √{ ⟨m_j^2⟩ - ⟨m_j⟩^2 } }``

# Arguments:
- `M_ts::AbstractMatrix`: `N×M` Matrix whose each of its `M` columns corresponds to a sample of a time series `Xₜ` of length `N`.
"""
@inline function normalize_ts_matrix!(M_ts::AbstractMatrix)
    for xⱼ ∈ eachcol(M_ts)
        xⱼ_avg = mean(xⱼ)
        xⱼ .= (xⱼ .- xⱼ_avg) ./ stdm(xⱼ, xⱼ_avg, corrected=true)
    end
end

@doc raw"""
    cross_correlation_matrix(M_ts::AbstractMatrix)

Cross correlation matrix `G` of a given time series matrix `M_ts`.

    ``G = \frac{1}{N_{samples}} M_{ts}^T M_{ts}``

# Arguments:
- `M_ts::AbstractMatrix`: `N×M` Matrix whose each of its `M` columns corresponds to a sample of a time series `Xₜ` of length `N`.
"""
function cross_correlation_matrix(M_ts::AbstractMatrix)
    # Number of steps = number of rows
    n_steps = size(M_ts, 1)
    return (1 / n_steps) .* Symmetric(transpose(M_ts) * M_ts)
end

end
