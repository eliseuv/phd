module BlumeCapelModel

export SpinState,
    BlumeCapel, BlumeCapelConcrete, BlumeCapelSquareLattice,
    set_state!, randomize_state!,
    magnet_total, magnet,
    energy,
    nearest_neighbors, nearest_neighbors_sum,
    heatbath_and_measure_total_magnet!

using Random, StatsBase

include("Metaprogramming.jl")
include("Geometry.jl")

using .Metaprogramming

SpinState = Int8

abstract type BlumeCapelConcrete{N} <: AbstractArray{SpinState,N} end

@inline Base.length(bc::BlumeCapelConcrete) = length(bc.state)

@inline Base.size(bc::BlumeCapelConcrete) = size(bc.state)
@inline Base.size(bc::BlumeCapelConcrete, dim) = size(bc.state, dim)

@inline Base.IndexStyle(::Type{<:BlumeCapelConcrete{N}}) where {N} = IndexStyle(Array{SpinState,N})

@inline Base.getindex(bc::BlumeCapelConcrete, inds...) = getindex(bc.state, inds...)
@inline Base.setindex!(bc::BlumeCapelConcrete, σ, inds...) = setindex!(bc.state, σ, inds...)

@inline Base.firstindex(bc::BlumeCapelConcrete) = firstindex(bc.state)
@inline Base.lastindex(bc::BlumeCapelConcrete) = lastindex(bc.state)

@inline function set_state!(bc::BlumeCapelConcrete, σ₀::SpinState)
    fill!(bc, σ₀)
end

@inline function randomize_state!(bc::BlumeCapelConcrete)
    rand!(bc, SpinState[-1, 0, +1])
end

@inline magnet_total(bc::BlumeCapelConcrete) = @inbounds sum(bc.state)
@inline magnet(bc::BlumeCapelConcrete) = magnet_total(bc) / length(bc)

@inline nearest_neighbors_sum(bc::BlumeCapelConcrete{N}, i::Union{Integer,CartesianIndex{N}}) where {N} = @inbounds sum(bc[nn] for nn ∈ nearest_neighbors(bc, i))

mutable struct BlumeCapelSquareLattice{N} <: BlumeCapelConcrete{N}

    state::Array{SpinState,N}

    BlumeCapelSquareLattice(size::NTuple{N,Integer}, σ₀::SpinState) where {N} = new{N}(fill(σ₀, size))
    BlumeCapelSquareLattice(size::NTuple{N,Integer}, ::Val{:rand}) where {N} = new{N}(rand(SpinState[-1, 0, +1], size))

    BlumeCapelSquareLattice(::Val{N}, L::Integer, σ₀::SpinState) where {N} = BlumeCapelSquareLattice(ntuple(_ -> L, Val(N)), σ₀)
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
        H -= sum(front_bulk .* back_bulk)
        # Periodic boundaries
        last_slice = selectdim(bc, d, size(bc, d))
        first_slice = selectdim(bc, d, 1)
        H -= sum(last_slice .* first_slice)
    end
    return H
end
@inline energy(bc::BlumeCapelSquareLattice, h::Real) = energy(bc) - h * magnet_total(bc)

@inline nearest_neighbors(bc::BlumeCapelSquareLattice{N}, idx::CartesianIndex{N}) where {N} = @inbounds Geometry.square_lattice_nearest_neighbors_flat(bc, idx)

@inline nearest_neighbors_sum(bc::BlumeCapelSquareLattice{N}, idx::CartesianIndex{N}) where {N} = @inbounds Geometry.square_lattice_nearest_neighbors_sum(bc, idx)
@inline nearest_neighbors_sum(bc::BlumeCapelSquareLattice, idx::Integer) = nearest_neighbors_sum(bc, CartesianIndices(bc)[idx])

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
            h_local = nearest_neighbors_sum(bc, i)
            weights = ProbabilityWeights(map(x -> exp(β * h_local * x), [-1, 0, +1]))
            # Update state
            bc[i] = sample(Int8[-1, 0, +1], weights)
        end
        # Update total magnetization vector
        M_t[t+1] = magnet_total(bc)
    end

    return M_t
end

end
