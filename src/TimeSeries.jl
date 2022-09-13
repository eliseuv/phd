module TimeSeries

export normalize_ts_matrix, normalize_ts_matrix!,
    cross_correlation_matrix

using LinearAlgebra, Statistics

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
        (xⱼ .- xⱼ_avg) ./ stdm(xⱼ, xⱼ_avg)
    end...)

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
        xⱼ .= (xⱼ .- xⱼ_avg) ./ stdm(xⱼ, xⱼ_avg)
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
