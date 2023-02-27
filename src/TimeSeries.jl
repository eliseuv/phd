@doc raw"""
    Time Series


"""
module TimeSeries

export
    normalize_ts, normalize_ts!,
    _normalize_ts_matrix!, normalize_ts_matrix, normalize_ts_matrix!,
    shuffle_cols!,
    triu_values,
    cross_correlation_matrix,
    cross_correlation_values, cross_correlation_values_norm

using Random, Combinatorics, Statistics, Distributions, LinearAlgebra

@doc raw"""
    _normalize_ts!(x′::AbstractVector, x::AbstractVector)

Normalize a given time series vector `x` and store the result in another vector `x′`.

    ``xᵢ′ = (xᵢ - x̄) / sₓ``
"""
@inline function _normalize_ts!(x′::AbstractVector, x::AbstractVector)
    x̄ = mean(x)
    x′ .= (x .- x̄) ./ stdm(x, x̄, corrected=true)
    return x′
end

@doc raw"""
    normalize_ts(x::AbstractVector)

Returns the normalized version of a given time series vector `x`.
"""
@inline normalize_ts(x::AbstractVector{T}) where {T} = _normalize_ts!(similar(x, promote_type(T, Float64)), x)

@inline normalize_ts!(x::AbstractVector) = _normalize_ts!(x, x)

# Normalize each column of a time series matrix and store the result in another matrix
@inline function _normalize_ts_matrix!(M′::AbstractMatrix, M::AbstractMatrix)
    for (x′ⱼ, xⱼ) ∈ zip(eachcol(M′), eachcol(M))
        _normalize_ts!(x′ⱼ, xⱼ)
    end
    return M′
end

@doc raw"""
    normalize_ts_matrix!(M::AbstractMatrix{<:AbstractFloat})

Normalizes *inplace* a given time series matrix `M` by normalizing each of its columns (time series samples).

Its `j`-th column (`xⱼ`) becomes:

    ``\frac{ x_j - ⟨x_j⟩ }{ √{ ⟨x_j^2⟩ - ⟨x_j⟩^2 } }``

Attention: This can only be done if the matrix `M` has floating point numbers as its entries.
Additionally, since the calculation is done in-place, the resulting matrix will have the same precision.

# Arguments:
- `M::AbstractMatrix{<:AbstractFloat}`: `N×M` Matrix whose each of its `M` columns corresponds to a sample of a time series `Xₜ` of length `N`.
"""
normalize_ts_matrix!(M::AbstractMatrix{<:AbstractFloat}) = _normalize_ts_matrix!(M, M)

@doc raw"""
    normalize_ts_matrix(M::AbstractMatrix)

Returns a new normalized version version of a time series matrix `M_ts` by normalizing each of its columns (time series samples).

Its `j`-th column (`mⱼ`) becomes:

    ``\frac{ m_j - ⟨m_j⟩ }{ √{ ⟨m_j^2⟩ - ⟨m_j⟩^2 } }``

# Arguments:
- `M::AbstractMatrix`: `N×M` Matrix whose each of its `M` columns corresponds to a sample of a time series `Xₜ` of length `N`.
"""
normalize_ts_matrix(M::AbstractMatrix) = _normalize_ts_matrix!(similar(M, Float64), M)

"""
    shuffle_cols!([rng=GLOBAL_RNG,] M::AbstractMatrix)

Shuffles the columns of a given matrix `M`.
"""
@inline shuffle_cols!(rng::AbstractRNG, M::AbstractMatrix) =
    Base.permutecols!!(M, randperm(rng, size(M, 2)))

@inline shuffle_cols!(M::AbstractMatrix) =
    Base.permutecols!!(M, randperm(size(M, 2)))

@doc raw"""
    cross_correlation_matrix(M::AbstractMatrix)

Cross correlation matrix `G` of a given time series matrix `M_ts`.

    ``G = \frac{1}{N_{samples}} M_{ts}^T M_{ts}``

# Arguments:
- `M::AbstractMatrix`: `N×M` Matrix whose each of its `M` columns corresponds to a sample of a time series `Xₜ` of length `N`.
"""
@inline cross_correlation_matrix(M::AbstractMatrix) = Symmetric(M' * M) ./ size(M, 1)

@inline function triu_values(M::AbstractMatrix)
    @assert size(M, 1) == size(M, 2) "Error: Matrix must be square!"
    n = size(M, 1)
    values = similar(M, ((n + 1) * n) ÷ 2)
    start_idx = 1
    for k ∈ 1:n
        @views column = M[begin:k, k]
        last_idx = start_idx - 1 + length(column)
        @views values[start_idx:last_idx] = column[:]
        start_idx = last_idx + 1
    end
    return values
end

@inline function cross_correlation_values(M::AbstractMatrix{<:Number})
    (t_max, n_series) = size(M)
    n_vals = ((n_series - 1) * n_series) ÷ 2
    corr_vec = Vector{Float64}(undef, n_vals)
    @inbounds @views for (k, (i, j)) ∈ enumerate(combinations(1:n_series, 2))
        x̄ᵢ = mean(M[:, i])
        x̄ⱼ = mean(M[:, j])
        corr_vec[k] = (((M[:, i] ⋅ M[:, j]) / t_max) - x̄ᵢ * x̄ⱼ) / sqrt(varm(M[:, i], x̄ᵢ) * varm(M[:, j], x̄ⱼ))
    end
    return corr_vec
end

@inline function cross_correlation_values_norm(M::AbstractMatrix{<:Number})
    (t_max, n_series) = size(M)
    n_vals = ((n_series - 1) * n_series) ÷ 2
    corr_vals = Vector{Float64}(undef, n_vals)
    @views for (k, (i, j)) ∈ enumerate(combinations(1:n_series, 2))
        corr_vals[k] = (M[:, i] ⋅ M[:, j]) / t_max
    end
    return corr_vals
end

end
