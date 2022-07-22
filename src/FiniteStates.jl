@doc raw"""
    Finite States

Finite states for systems.
"""
module FiniteStates

export
    # Abstract site state
    AbstractSiteState, instance_count,
    # Abstract finite state
    AbstractFiniteState,
    state_count, state_concentration,
    set_state!, randomize_state!,
    nearest_neighbors, nearest_neighbors_sum,
    # Mean field finite state
    MeanFieldFiniteState, clear, split_indices, site_counts_from_split_indices,
    # Concrete finite state
    ConcreteFiniteState,
    container, similar_container,
    # Square lattice finite state
    SquareLatticeFiniteState,
    # Simple graph finite state
    SimpleGraphFiniteState

using Random, Graphs

include("Lattices.jl")

"""
    AbstractSiteState

Supertype for single site state.
For now it is assumed to be an `Enum`.
"""
AbstractSiteState = Enum

"""
    instance_count(::Type{T}) where {T<:AbstractSiteState}

Total number of possible instances associated with a single site state.
"""
@inline instance_count(::Type{T}) where {T<:AbstractSiteState} = length(instances(T))

"""
    convert(::Type{T}, σ::AbstractSiteState) where {T<:Number}

Use the integer representation of `σ::AbstractSiteState` in order to convert it to a numerical type `T<:Number`.
"""
@inline Base.convert(::Type{T}, σ::AbstractSiteState) where {T<:Number} = T(Integer(σ))

"""
    promote_rule(T::Type, ::Type{SingleSpinState})

"""
@inline Base.promote_rule(T::Type, S::Type{<:AbstractSiteState}) = promote_rule(T, Base.Enums.basetype(S))

"""
    AbstractFiniteState{T<:AbstractSiteState,N},N} <: AbstractArray{T,N}

Supertype for all finite states.
"""
abstract type AbstractFiniteState{T<:AbstractSiteState,N} <: AbstractArray{T,N} end

@doc raw"""
    state_concentration(fs::AbstractFiniteState)

Concentration of each site state in given finite state `fs`.

``cᵢ = Nᵢ/N``

See also: [`state_count`](@ref).
"""
@inline state_concentration(fs::AbstractFiniteState) = state_count(fs) ./ length(fs)

@doc raw"""
    MeanFieldFiniteState{T} <: AbstractFiniteState{T,1}

State with no concept of space and locality.
Every site interacts equally with every other site.

Since in the mean field model there is no concept of space and locality,
we represent the state of the system simply by total number of sites in each state.

The `state` member is therefore of type `Dict{T,UInt64}`,
storing for each state instance the total number of sites in this state.

An `AbstractVector` interface for the `MeanFieldFiniteState` type can be implemented
if we assume that the all site states are stored in a sorted vector.

Assume the possible site states are `σ₁`, `σ₂`, etc.

    state = (σ₁, σ₁, …, σ₁, σ₂, σ₂, …, σ₂, σ₃, …)
                        ↑ k₁           ↑ k₂
            |----- N₁ ----||----- N₂ ----|
            |---------------- N ----------------|

The values `k₁`, `k₂`, …, `kₘ₋₁` are called split indices and can be used to access the `i`-th site.

``kᵢ = ∑_{j≤i} Nⱼ ∀ i = 1, 2, …, m``

Note that we always have `kₘ = N`.

For a `fs::MeanFieldFiniteState` we can access the `i`-th site `sᵢ = fs[i]`:
    sᵢ = σⱼ if kⱼ₋₁ < i ≤ kⱼ
"""
mutable struct MeanFieldFiniteState{T} <: AbstractFiniteState{T,1}

    "State of the system"
    state::Dict{T,Unsigned}


    @doc raw"""
        MeanFieldFiniteState(N::Integer, σ₀::T) where {T}

    Construct mean field finite state `N` sites, all in a given initial state `σ₀` for all of them.
    """
    function MeanFieldFiniteState(N::U, σ₀::T) where {T,U<:Unsigned}
        state = Dict(instances(T) .=> zero(U))
        state[σ₀] = N
        return new{T}(state)
    end

    @doc raw"""
        MeanFieldFiniteState(N::Integer, ::Val{:rand})

    Construct an spin state with mean field interaction with `N` spins in a random initial state.
    """
    function MeanFieldFiniteState{T}(N::U, ::Val{:rand}) where {T,U<:Unsigned}
        split_indices = [sort(rand(0:N, instance_count(T) - 1))..., N]
        site_counts = site_counts_from_split_indices(split_indices)
        return new{T}(Dict(instances(T) .=> site_counts))
    end

end

"""
    clear(fs::MeanFieldFiniteState{T}) where {T}

Clears the state of the mean field finites state `fs` by setting the site count to all states to zero.
"""
@inline function clear(fs::MeanFieldFiniteState{T}) where {T}
    fs.state = Dict(instances(T) .=> zero(UInt64))
end

"""
    split_indices(fs::MeanFieldFiniteState{T}) where {T}

Get a tuple with the values of the split indices `kⱼ` for the mean field finite state `fs`.
"""
@inline split_indices(fs::MeanFieldFiniteState{T}) where {T} = @inbounds cumsum(fs.state[σᵢ] for σᵢ ∈ instances(T))

"""
    site_counts_from_split_indices(split_indices::Vector{Integer})

Get the number of spins in each state given the split indices `split_indices`.
"""
@inline site_counts_from_split_indices(split_indices::Vector{Unsigned}) = [split_indices[begin], diff(split_indices)]

"""
    getindex(fs::MeanFieldFiniteState{T}, σ::T) where {T}

Allow the site count for a given state `σ` to be accessed directly using the mean field finite state `fs`.
"""
@inline Base.getindex(fs::MeanFieldFiniteState{T}, σ::T) where {T} = fs.state[σ]

"""
    setindex!(fs::MeanFieldFiniteState{T}, Nᵢ::Unsigned, σᵢ::T) where {T}

Set the site count to `Nᵢ` for a given site `σᵢ` in the mean field finite state `fs`.
"""
@inline function Base.setindex!(fs::MeanFieldFiniteState{T}, Nᵢ::Unsigned, σᵢ::T) where {T}
    fs.state[σᵢ] = Nᵢ
end

@doc raw"""
    length(spins::MeanFieldFiniteState)

Get the total number of sites `N` in an mean field finite state `fs`.
"""
@inline Base.length(spins::MeanFieldFiniteState) = @inbounds sum(values(spins.state))

@doc raw"""
    IndexStyle(::Type{<:MeanFieldFiniteState})

Use only linear indices for the `AbstractVector{AbstractSiteState}` interface for the `MeanFieldFiniteState` type.
"""
@inline Base.IndexStyle(::Type{<:MeanFieldFiniteState}) = IndexLinear()

@doc raw"""
    getindex(fs::MeanFieldFiniteState{T}, i::Integer) where {T}

Get the state of the `i`-th site in the mean field finite state `fs`.
"""
@inline function Base.getindex(fs::MeanFieldFiniteState{T}, i::Integer) where {T}
    # Iterate on the possible state indices and return if smaller than a given split index
    for (σᵢ, split_index) ∈ zip(instances(T)[1:end-1], split_indices(fs))
        if i < split_index
            return σᵢ
        end
    end
    # If `i` is not smaller than any split indices, return the last state value
    return instances(T)[end]
end

"""
    setindex!(fs::MeanFieldFiniteState{T}, σ_new::T, i::Integer) where {T}

Set the state of the `i`-th site to `σᵢ′` in the mean field finite state `fs`.
"""
@inline function Base.setindex!(fs::MeanFieldFiniteState{T}, σᵢ′::T, i::Integer) where {T}
    σᵢ = fs[i]
    fs[σᵢ] -= 1
    fs[σᵢ′] += 1
end

@doc raw"""
    firstindex(::MeanFieldFiniteState)

Index of the first site in the `AbstractVector{AbstractSiteState}` interface of `MeanFieldFiniteState` is `1`.
"""
@inline Base.firstindex(::MeanFieldFiniteState) = 1

@doc raw"""
    lastindex(fs::MeanFieldFiniteState)

Index of the last site in the `AbstractVector{AbstractSiteState}` interface of `MeanFieldFiniteState` is equal the total number of sites `N`.
"""
@inline Base.lastindex(fs::MeanFieldFiniteState) = length(fs)

"""
    sum(f::Function=identity, fs::MeanFieldFiniteState)

Get the sum of the states of all sites in the mean field finite state `fs` with the function `f` applied to each.
"""
@inline Base.sum(f::Function, fs::MeanFieldFiniteState) =
    @inbounds sum(fs.state) do (σᵢ, Nᵢ)
        Nᵢ * f(Integer(σᵢ))
    end
@inline Base.sum(fs::MeanFieldFiniteState) =
    @inbounds sum(fs.state) do (σᵢ, Nᵢ)
        Nᵢ * Integer(σᵢ)
    end

"""
    state_count(fs::MeanFieldFiniteState)

Get the state count for the mean field finite state `fs`.
"""
@inline state_count(fs::MeanFieldFiniteState) = tuple(values(fs.state)...)

"""
    set_state!(fs::MeanFieldFiniteState{T}, σ₀::T) where {T}

Set the state of all sites to `σ₀` in a mean field finite state `fs`.
"""
function set_state!(fs::MeanFieldFiniteState{T}, σ₀::T) where {T}
    N = length(fs)
    # Set all values in the state count to zero
    clear(fs)
    # Set the selected state site count to `N`
    fs[σ₀] = N
end

"""
    randomize_state!(fs::MeanFieldFiniteState{T}) where {T}

Randomize the state of a mean field finite state `fs`.
"""
function randomize_state!(fs::MeanFieldFiniteState{T}) where {T}
    N = length(fs)
    split_indices = [sort(rand(0:N, instance_count(T) - 1))..., N]
    site_counts = site_counts_from_split_indices(split_indices)
    fs.state = Dict(instances(T) .=> site_counts)
end

"""
    nearest_neighbors(fs::MeanFieldFiniteState)

Get iterator over all pairs of nearest neighbors for the mean field finite system `fs`.
That is, all possible pairs of sites.
"""
@inline nearest_neighbors(fs::MeanFieldFiniteState) = Iterators.Stateful((i => j) for i ∈ 2:length(fs) for j ∈ 1:(i-1))

"""
    nearest_neighbors(fs::MeanFieldFiniteState, i::Integer)

Get vector with the indices of the nearest neighobrs sites of the `i`-th site in the mean files spin state `fs`.
That is, all sites except for `i`.
"""
@inline nearest_neighbors(fs::MeanFieldFiniteState, i::Integer) = Iterators.Stateful([1:(i-1)..., (i+1):length(fs)...])

"""
    nearest_neighbors_sum(fs::MeanFieldFiniteState, i::Integer)

Get sum of the nearest neighbors of site `i` in the mean field finite state `fs`.
"""
@inline nearest_neighbors_sum(fs::MeanFieldFiniteState, i::Integer) = sum(fs) - Integer(fs[i])

@doc raw"""
    ConcreteFiniteState <: AbstractSpinState

Supertype for all finite state that have a concrete representation of its state in memory
in the form of a concrete array member `container::Array{AbstractSiteState}`.

The whole indexing interface of the `container` can be passed to the `ConcreteFiniteState` object itself.
"""
abstract type ConcreteFiniteState{T,N} <: AbstractFiniteState{T,N} end

"""
    container(fs::ConcreteFiniteState)

Returns container with state representation stored in memory.
"""
@inline container(fs::ConcreteFiniteState) = fs.container

"""
    length(fs::ConcreteFiniteState)

Total number of sites of a concrete finite state `fs`.
"""
@inline Base.length(fs::ConcreteFiniteState) = length(container(fs))

"""
    size(fs::ConcreteFiniteState)

Size of the concrete finite state `fs`.
"""
@inline Base.size(fs::ConcreteFiniteState) = size(container(fs))

"""
    IndexStyle(::Type{<:ConcreteFiniteState})

Use same indexing style used to index the state array.
"""
# @inline Base.IndexStyle(::Type{<:ConcreteFiniteState{T,N}}) where {T<:AbstractSiteState,N} = IndexStyle(Array{T,N})
@inline Base.IndexStyle(::Type{<:ConcreteFiniteState{T,N}}) where {T,N} = IndexCartesian()
@inline Base.IndexStyle(::Type{<:ConcreteFiniteState{T,1}}) where {T} = IndexLinear()

"""
    getindex(fs::ConcreteFiniteState, inds...)

Index the concrete finite state itself to access its container.
"""
@inline Base.getindex(fs::ConcreteFiniteState, inds...) = getindex(container(fs), inds...)

"""
    setindex!(fs::ConcreteFiniteState, σ, inds...)

Index the concrete finite state itself to access its container.
"""
@inline Base.setindex!(fs::ConcreteFiniteState, σ, inds...) = setindex!(container(fs), σ, inds...)

"""
    firstindex(fs::ConcreteFiniteState)

Get the index of the first site in the concrete finite state `fs`.
"""
@inline Base.firstindex(fs::ConcreteFiniteState) = firstindex(container(fs))

"""
    lastindex(spins::ConcreteFiniteState)

Get the index of the last site in the concrete finite state `fs`.
"""
@inline Base.lastindex(fs::ConcreteFiniteState) = lastindex(container(fs))

"""
    similar_container(fs::ConcreteFiniteState)

Return a new uninitialized instance of the the container used by the concrete finite state `fs`.
"""
@inline similar_container(fs::ConcreteFiniteState{T,N}) where {T,N} = similar(Array{T,N}, axes(fs.container))

"""
    state_count(fs::ConcreteFiniteState)

Get the state count for the concrete finite state `fs`.
"""
@inline state_count(fs::ConcreteFiniteState{T}) where {T} = (count(==(σ), container(fs)) for σ in instances(T))

"""
    set_state!(fs::ConcreteFiniteState{T}, σ₀::T) where {T}

Set the state of all sites of a concrete finite state `fs` to a given state `σ₀`.
"""
@inline function set_state!(fs::ConcreteFiniteState{T}, σ₀::T) where {T}
    fill!(container(fs), σ₀)
end

@inline function set_state!(fs::ConcreteFiniteState{T,N}, container::Array{T,N}) where {T,N}
    @assert size(container) == size(fs) "Container size mismatch."
    fs.container = container
end

"""
    randomize_state!(spins::ConcreteFiniteState{T}) where {T}

Set the state of all sites of a concrete finite state `fs` to a random state `σ ∈ AbstractSiteState`.
"""
@inline function randomize_state!(spins::ConcreteFiniteState{T}) where {T}
    rand!(container(spins), instances(T))
end

"""
    sum(f::Function=identity, fs::ConcreteFiniteState)

Get the sum of the states of all sites in the concrete finite state `fs` with the function `f` applied to each.
"""
@inline Base.sum(f::Function, fs::ConcreteFiniteState) = @inbounds sum(sᵢ -> f(Integer(sᵢ)), container(fs))
@inline Base.sum(fs::ConcreteFiniteState) = @inbounds sum(Integer, container(fs))

"""
    nearest_neighbors_sum(fs::ConcreteFiniteState{T,N}, i::Union{Integer,CartesianIndex{N}}) where {T,N}

Sum of the nearest neighbors of the `i`-th site for a concrete finite state `fs`.

Every subtype of `ConcreteFiniteState` must provide its own implementation of `nearest_neighbors(fs::ConcreteFiniteState{T,N}, i::Union{Integer,CartesianIndex{N}})::AbstractVector{IndexStyle(ConcreteFiniteState)}`
"""
@inline nearest_neighbors_sum(fs::ConcreteFiniteState{T,N}, i::Union{Integer,CartesianIndex{N}}) where {T,N} = @inbounds sum(Integer, fs[nn] for nn ∈ nearest_neighbors(fs, i))

"""
    SquareLatticeFiniteState{T,N} <: ConcreteFiniteState{T,N}

Finite state on a `N`-dimensional square lattice.
"""
mutable struct SquareLatticeFiniteState{T,N} <: ConcreteFiniteState{T,N}

    "Multidimensional array with system state"
    container::Array{T,N}

    """
        SquareLatticeFiniteState(state::Array{T,N}) where {T,N}

    Construct a new finite state in a `N`-dimensional square lattice using the the array `container`.
    """
    SquareLatticeFiniteState(container::Array{T,N}) where {T,N} = new{T,N}(container)

    """
        SquareLatticeFiniteState(size::NTuple{N,Integer}, σ₀::T) where {T,N}

    Construct a new finite state in a `N`-dimensional square lattice of dimensions provided by `size`,
    with all sites with same initial state `σ₀`.
    """
    SquareLatticeFiniteState(size::NTuple{N,Integer}, σ₀::T) where {T,N} = new{T,N}(fill(σ₀, size))

    """
        SquareLatticeFiniteState(size::NTuple{N,Integer}, ::Val{:rand}) where {T,N}

    Construct a new finite state in a `N`-dimensional square lattice of dimensions provided by `size`,
    with sites in random states.
    """
    SquareLatticeFiniteState(size::NTuple{N,Integer}, ::Type{T}, ::Val{:rand}) where {T,N} = new{T,N}(rand(instances(T), size))

    """
        SquareLatticeFiniteState(::Val{N}, L::Integer, σ₀::T) where {T,N}

    Construct a `N`-dimensional square finite state of side length `L` and a given initial state `σ₀`.
    """
    SquareLatticeFiniteState(::Val{N}, L::Integer, σ₀::T) where {T,N} = SquareLatticeFiniteState(ntuple(_ -> L, Val(N)), σ₀)

    """
        SquareLatticeFiniteState(::Val{N}, L::Integer, ::Val{:rand}) where {N}

    Construct a `N`-dimensional square finite state of side length `L` and random initial state.
    """
    SquareLatticeFiniteState(::Val{N}, L::Integer, ::Type{T}, ::Val{:rand}) where {T,N} = SquareLatticeFiniteState(ntuple(_ -> L, Val(N)), T, Val(:rand))

end

"""
    IndexStyle(::Type{<:SquareLatticeFiniteState})

Prefer cartesian indices for multidimensional square lattice finite states.
"""
# Base.IndexStyle(::Type{<:SquareLatticeFiniteState}) = IndexCartesian()

"""
    nearest_neighbors(fs::SquareLatticeFiniteState{T,N}, idx::CartesianIndex{N}) where {T,N}

Gets a vector containing the indices of the nearest neighbors to the `i`-site in the square lattice finite state `fs`.
"""
@inline nearest_neighbors(fs::SquareLatticeFiniteState{T,N}, i::CartesianIndex{N}) where {T,N} = @inbounds Lattices.square_lattice_nearest_neighbors_flat(fs.container, i)

"""
    nearest_neighbors_sum(fs::SquareLatticeFiniteState{T,N}, i::CartesianIndex{N}) where {T,N}

Sum of the nearest neighbors of the `i`-th site for a multidimensional square lattice finites state `fs`.
"""
@inline nearest_neighbors_sum(fs::SquareLatticeFiniteState{T,N}, i::CartesianIndex{N}) where {T,N} = @inbounds Lattices.square_lattice_nearest_neighbors_sum(fs.container, i)
# @inline nearest_neighbors_sum(spins::SquareLatticeFiniteState{T,N}, i::Integer) where {T,N} = nearest_neighbors_sum(spins, CartesianIndices(spins)[i])

"""
    SimpleGraphFiniteState{T} <: ConcreteFiniteState{T,1}

Finite state on a simple graph.
"""
mutable struct SimpleGraphFiniteState{T} <: ConcreteFiniteState{T,1}

    "Graph structure of the system"
    graph::SimpleGraph

    "State at each node"
    container::Vector{T}

    """
        SimpleGraphFiniteState(graph::Graph, σ₀::T) where {T}

    Construct a new finite state with graph structure `graph` with all sites with same initial state `σ₀`.
    """
    SimpleGraphFiniteState(graph::SimpleGraph, σ₀::T) where {T} = new{T}(graph, fill(σ₀, nv(graph)))

    """
        IsingGraph(g::Graph, ::Val{:rand})

    Construct a new finite state with graph structure `graph` and random initial states at each site.
    """
    SimpleGraphFiniteState(graph::SimpleGraph, ::Val{:rand}) where {T} = new{T}(graph, rand(instances(T), nv(graph)))

end

"""
    nearest_neighbors(fs::SimpleGraphFiniteState)

Get list of all nearest neighbors pairs in the simple graph finite state `fs`.
"""
@inline nearest_neighbors(fs::SimpleGraphFiniteState) = edges(fs.graph)

"""
    nearest_neighbors(fs::SimpleGraphFiniteState, i::Integer)

Get the indices of the nearest neighbors of `i`-th site in the simple graph finite state `fs`.
"""
@inline nearest_neighbors(fs::SimpleGraphFiniteState, i::Integer) = neighbors(fs.graph, i)

end
