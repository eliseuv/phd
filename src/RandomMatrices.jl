@doc raw"""
    Random Matrices


"""
module RandomMatrices

export
    normalize_ts_matrix!, normalize_ts_matrix,
    shuffle_cols!,
    cross_correlation_matrix

using Random, Statistics, Distributions, LinearAlgebra

"""
    shuffle_cols!([rng=GLOBAL_RNG,] M::AbstractMatrix)

Shuffles the columns of a given matrix `M`.
"""
@inline shuffle_cols!(rng::AbstractRNG, M::AbstractMatrix) =
    Base.permutecols!!(M, randperm(rng, size(M, 2)))

@inline shuffle_cols!(M::AbstractMatrix) =
    Base.permutecols!!(M, randperm(size(M, 2)))

# Normalize a given time series vector and store the result in another vector
function _normalize_ts!(x::AbstractVector, x′::AbstractVector)
    x̄ = mean(x)
    x′ .= (x .- x̄) ./ stdm(x, x̄, corrected=true)
end

# Normalize each column of a time series matrix and store the result in another matrix
@inline function _normalize_ts_matrix!(M::AbstractMatrix, M′::AbstractMatrix)
    for (xⱼ, x′ⱼ) ∈ zip(eachcol(M), eachcol(M′))
        _normalize_ts!(xⱼ, x′ⱼ)
    end
    return M′
end

@doc raw"""
    normalize_ts_matrix!(M::AbstractMatrix)

Normalizes *inplace* a given time series matrix `M` by normalizing each of its columns (time series samples).

Its `j`-th column (`mⱼ`) becomes:

    ``\frac{ m_j - ⟨m_j⟩ }{ √{ ⟨m_j^2⟩ - ⟨m_j⟩^2 } }``

# Arguments:
- `M::AbstractMatrix`: `N×M` Matrix whose each of its `M` columns corresponds to a sample of a time series `Xₜ` of length `N`.
"""
normalize_ts_matrix!(M::AbstractMatrix) = _normalize_ts_matrix!(M, M)

@doc raw"""
    normalize_ts_matrix(M::AbstractMatrix)

Returns a new normalized version version of a time series matrix `M_ts` by normalizing each of its columns (time series samples).

Its `j`-th column (`mⱼ`) becomes:

    ``\frac{ m_j - ⟨m_j⟩ }{ √{ ⟨m_j^2⟩ - ⟨m_j⟩^2 } }``

# Arguments:
- `M::AbstractMatrix`: `N×M` Matrix whose each of its `M` columns corresponds to a sample of a time series `Xₜ` of length `N`.
"""
normalize_ts_matrix(M::AbstractMatrix) = _normalize_ts_matrix!(M, similar(M))

@doc raw"""
    cross_correlation_matrix(M::AbstractMatrix)

Cross correlation matrix `G` of a given time series matrix `M_ts`.

    ``G = \frac{1}{N_{samples}} M_{ts}^T M_{ts}``

# Arguments:
- `M::AbstractMatrix`: `N×M` Matrix whose each of its `M` columns corresponds to a sample of a time series `Xₜ` of length `N`.
"""
@inline cross_correlation_matrix(M::AbstractMatrix) = Symmetric(M' * M) ./ size(M, 1)

end
