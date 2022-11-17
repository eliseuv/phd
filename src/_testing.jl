using Random

@inline logistic_map(r::Real, xₙ::Real) = r * xₙ * (1 - xₙ)

@inline function _logistic_map!(r::Real, x::Vector{<:Real})
    for i ∈ 1:length(x)-1
        x[i+1] = logistic_map(r, x[i])
    end
    return x
end

@inline function logistic_map(r::Real, x₀::T, n::Integer) where {T<:Real}
    x = Vector{T}(undef, n)
    x[begin] = x₀
    return _logistic_map!(r, x)
end

@inline function logistic_map_matrix(r::Real, x₀::Real, (m, n)::Pair{<:Integer})
    x = logistic_map(r, x₀, m * n)
    shuffle!(x)
    return reshape(x, m, n)
end

@inline logistic_map_matrix(r::Real, dims::Pair{<:Integer}) = logistic_map_matrix(r, rand(), dims)
