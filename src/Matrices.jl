module Matrices

export cross_correlation_matrix

using LinearAlgebra

"""
    cross_correlation_matrix(ts_matrix::AbstractMatrix)

Cross correlation matrix.

# Arguments:
- `ts_matrix::AbstractMatrix`: `N×M` Matrix whose each of its `N` rows corresponds to a sample of a time series `Xₜ` of length `M`.

# Returns:
- `cov_matrix::AbstractMatrix`: `Cᵢⱼ`
"""
function cross_correlation_matrix(ts_matrix::AbstractMatrix)
    (_, ncol) = size(ts_matrix)
    return (1 / ncol) * ts_matrix * transpose(ts_matrix)
end

end
