module Matrices

export cross_correlation_matrix

using LinearAlgebra, Statistics

@doc raw"""
    normalize_M_ts(M_ts::AbstractMatrix)

Normalizes a given time series matrix `M_ts` by normalizing each of its columns (time series samples).

Its `j`-th column (`mⱼ`) becomes:

    ``\frac{ m_j - ⟨m_j⟩ }{ √{ ⟨m_j^2⟩ - ⟨m_j⟩^2 } }``

# Arguments:
- `M_ts::AbstractMatrix`: `N×M` Matrix whose each of its `M` columns corresponds to a sample of a time series `Xₜ` of length `N`.
"""
@inline normalize_ts_matrix(M_ts::AbstractMatrix) = hcat(
    @views map(eachcol(M_ts)) do mⱼ
        (mⱼ .- mean(mⱼ)) ./ std(mⱼ)
    end...)

@inline function normalize_ts_matrix!(M_ts::AbstractMatrix)
    @views for mⱼ ∈ eachcol(M_ts)
        mⱼ .= (mⱼ .- mean(mⱼ)) ./ std(mⱼ)
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
    return (1 / n_steps) * transpose(M_ts) * M_ts
end

end
