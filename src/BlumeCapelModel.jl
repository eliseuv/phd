module BlumeCapelModel

export SpinOneState,
    BlumeCapel, BlumeCapelConcrete, BlumeCapelSquareLattice,
    set_state!, randomize_state!,
    magnet_total, magnet,
    energy,
    nearest_neighbors, magnet_total_local,
    heatbath_and_measure_total_magnet!

using EnumX, Random, StatsBase

include("Metaprogramming.jl")
include("Geometry.jl")

using .Metaprogramming

@enumx SpinOneState::Int8 begin
    zero = 0
    down = -1
    up = +1
end

@inline Base.convert(::Type{T}, σ::SpinOneState.T) where {T<:Number} = T(Integer(σ))

@inline Base.promote_rule(T::Type, ::Type{SpinOneState.T}) = T

# Arithmetic with numbers and Spin States
for op in (:*, :/, :+, :-)
    @eval begin
        @inline Base.$op(x::Number, σ::SpinOneState.T) = $op(promote(x, σ)...)
        @inline Base.$op(σ::SpinOneState.T, y::Number) = $op(promote(σ, y)...)
    end
end

@inline Base.:*(σ₁::SpinOneState.T, σ₂::SpinOneState.T) = Integer(σ₁) * Integer(σ₂)

function Base.show(io::IO, ::MIME"text/plain", σ::SpinOneState.T)
    spin_char = σ == up ? '↑' : '↓'
    print(io, spin_char)
end

abstract type BlumeCapelConcrete{N} <: AbstractArray{SpinOneState.T,N} end

@inline Base.length(bc::BlumeCapelConcrete) = length(bc.state)

@inline Base.size(bc::BlumeCapelConcrete) = size(bc.state)
@inline Base.size(bc::BlumeCapelConcrete, dim) = size(bc.state, dim)

@inline Base.IndexStyle(::Type{<:BlumeCapelConcrete{N}}) where {N} = IndexStyle(Array{SpinOneState.T,N})

@inline Base.getindex(bc::BlumeCapelConcrete, inds...) = getindex(bc.state, inds...)
@inline Base.setindex!(bc::BlumeCapelConcrete, σ, inds...) = setindex!(bc.state, σ, inds...)

@inline Base.firstindex(bc::BlumeCapelConcrete) = firstindex(bc.state)
@inline Base.lastindex(bc::BlumeCapelConcrete) = lastindex(bc.state)

@inline function set_state!(bc::BlumeCapelConcrete, σ₀::SpinOneState.T)
    fill!(bc, σ₀)
end

@inline function randomize_state!(bc::BlumeCapelConcrete)
    rand!(bc, instances(SpinOneState.T))
end

@inline magnet_total(bc::BlumeCapelConcrete) = @inbounds sum(Integer, bc.state)

@inline magnet(bc::BlumeCapelConcrete) = magnet_total(bc) / length(bc)

@inline magnet_total_local(bc::BlumeCapelConcrete{N}, i::Union{Integer,CartesianIndex{N}}) where {N} = @inbounds sum(Integer, bc[nn] for nn ∈ nearest_neighbors(bc, i))

mutable struct BlumeCapelSquareLattice{N} <: BlumeCapelConcrete{N}

    state::Array{SpinOneState.T,N}

    BlumeCapelSquareLattice(size::NTuple{N,Integer}, σ₀::SpinOneState.T) where {N} = new{N}(fill(σ₀, size))
    BlumeCapelSquareLattice(size::NTuple{N,Integer}, ::Val{:rand}) where {N} = new{N}(rand(instances(SpinOneState.T), size))

    BlumeCapelSquareLattice(::Val{N}, L::Integer, σ₀::SpinOneState.T) where {N} = BlumeCapelSquareLattice(ntuple(_ -> L, Val(N)), σ₀)
    BlumeCapelSquareLattice(::Val{N}, L::Integer, ::Val{:rand}) where {N} = BlumeCapelSquareLattice(ntuple(_ -> L, Val(N)), Val(:rand))
end

function energy(bc::BlumeCapelSquareLattice{N}) where {N}
    # Interaction energy
    H = zero(Int64)
    # Loop on dimensions
    @inbounds for d ∈ 1:N
        # Bulk
        front_bulk = selectdim(bc, d, 1:(size(bc, d)-1))
        back_bulk = selectdim(bc, d, 2:size(bc, d))
        H -= sum(Integer, front_bulk .* back_bulk)
        # Periodic boundaries
        last_slice = selectdim(bc, d, size(bc, d))
        first_slice = selectdim(bc, d, 1)
        H -= sum(Integer, last_slice .* first_slice)
    end
    return H
end
@inline energy(bc::BlumeCapelSquareLattice, h::Real) = energy(bc) - h * magnet_total(bc)

@inline nearest_neighbors(bc::BlumeCapelSquareLattice{N}, idx::CartesianIndex{N}) where {N} = @inbounds Geometry.square_lattice_nearest_neighbors_flat(bc, idx)

@inline magnet_total_local(bc::BlumeCapelSquareLattice{N}, idx::CartesianIndex{N}) where {N} = @inbounds Geometry.square_lattice_nearest_neighbors_sum(bc, idx)
@inline magnet_total_local(bc::BlumeCapelSquareLattice, idx::Integer) = magnet_total_local(bc, CartesianIndices(bc)[idx])

function Base.show(io::IO, ::MIME"text/plain", bc::BlumeCapelSquareLattice{N}) where {N}
    # Get output from printing state
    io_temp = IOBuffer()
    show(IOContext(io_temp, :limit => true), "text/plain", Integer.(bc.state))
    str = String(take!(io_temp))
    # Use symbols instead of numbers
    str = replace(str, "-1" => " ↓", "1" => "↑", "0" => "-")
    # Fix horizontal spacing
    str = replace(str, "  " => " ")
    str = replace(str, "⋮" => "  ⋮", "⋱" => "   ⋱", " …  " => " … ")
    # Output final result
    print(io, str)
end

BlumeCapel = Union{BlumeCapelConcrete}

function heatbath_and_measure_total_magnet!(bc::BlumeCapel, β::Real, n_steps::Integer)
    # Magnetization vector
    M_t = Vector{Int64}(undef, n_steps + 1)
    # Initial magnetization
    M_t[1] = magnet_total(bc)
    # Sampling loop
    @inbounds for t ∈ 1:n_steps
        # Site loop
        @inbounds for i ∈ rand(eachindex(bc), length(bc))
            # Calculate weights
            h_local = magnet_total_local(bc, i)
            weights = ProbabilityWeights(map(x -> exp(β * h_local * Integer(x)), [instances(SpinOneState.T)...]))
            # Update state
            bc[i] = sample([instances(SpinOneState.T)...], weights)
        end
        # Update total magnetization vector
        M_t[t+1] = magnet_total(bc)
    end

    return M_t
end

end
