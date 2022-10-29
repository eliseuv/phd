using Statistics, Random, BenchmarkTools

BenchmarkTools.DEFAULT_PARAMETERS.overhead = BenchmarkTools.estimate_overhead()
BenchmarkTools.DEFAULT_PARAMETERS.samples = 100000
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 10

@inline function _normalize_ts!(x::AbstractVector, x′::AbstractVector)
    x̄ = mean(x)
    x′ .= (x .- x̄) ./ stdm(x, x̄, corrected=true)
end

@inline function _normalize_ts_matrix!(M::AbstractMatrix, M′::AbstractMatrix)
    for (xⱼ, x′ⱼ) ∈ zip(eachcol(M), eachcol(M′))
        _normalize_ts!(xⱼ, x′ⱼ)
    end
    return M′
end

function normalize_ts_matrix_inline!(M::AbstractMatrix)
    for xⱼ ∈ eachcol(M)
        x̄ⱼ = mean(xⱼ)
        xⱼ .= (xⱼ .- x̄ⱼ) ./ stdm(xⱼ, x̄ⱼ, corrected=true)
    end
end

function normalize_ts_matrix_eachcol!(M::AbstractMatrix)
    for xⱼ ∈ eachcol(M)
        _normalize_ts!(xⱼ, xⱼ)
    end
end

normalize_ts_matrix_call!(M::AbstractMatrix) = _normalize_ts_matrix!(M, M)

function normalize_ts_matrix_inline(M::AbstractMatrix)
    M′ = similar(M)
    for (xⱼ, x′ⱼ) ∈ zip(eachcol(M), eachcol(M′))
        x̄ⱼ = mean(xⱼ)
        x′ⱼ .= (xⱼ .- x̄ⱼ) ./ stdm(xⱼ, x̄ⱼ, corrected=true)
    end
end

function normalize_ts_matrix_eachcol(M::AbstractMatrix)
    M′ = similar(M)
    for (xⱼ, x′ⱼ) ∈ zip(eachcol(M), eachcol(M′))
        _normalize_ts!(xⱼ, x′ⱼ)
    end
end

normalize_ts_matrix_call(M::AbstractMatrix) = _normalize_ts_matrix!(M, similar(M))

M_size = (1000, 1000)

println("NTSM inplace inline")
@btime normalize_ts_matrix_inline!($(rand(M_size...)))

println("NTSM inplace eachcol")
@btime normalize_ts_matrix_eachcol!($(rand(M_size...)))

println("NTSM inplace call")
@btime normalize_ts_matrix_call!($(rand(M_size...)))

println("NTSM inline")
@btime normalize_ts_matrix_inline($(rand(M_size...)))

println("NTSM eachcol")
@btime normalize_ts_matrix_eachcol($(rand(M_size...)))

println("NTSM call")
@btime normalize_ts_matrix_call($(rand(M_size...)))
