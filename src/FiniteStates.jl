@doc raw"""
    Finite States


"""
module FiniteStates

export AbstractFiniteState

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
    AbstracFiniteState{T<:AbstractSiteState,N},N} <: AbstractArray{T,N}

Supertype for all finite states.
"""
abstract type AbstractFiniteState{T<:AbstractSiteState,N} <: AbstractArray{T,N} end

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

For a `mfss::MeanFieldFiniteState` we can access the `i`-th site `sᵢ = mfss[i]`:
    sᵢ = σⱼ if kⱼ₋₁ < i ≤ kⱼ
"""
mutable struct MeanFieldFiniteState{T} <: AbstractFiniteState{T,1}

    "State of the system"
    state::Dict{T,Unsigned}


    @doc raw"""
        MeanFieldFiniteState(N::Integer, σ₀::T) where {T}

    Construct mean field finite state `N` sites, all in a given initial state `σ₀` for all of them.
    """
    function MeanFieldFiniteState(N::U, σ₀::T) where {T<:AbstractSiteState,U<:Unsigned}
        state = Dict(instances(T) .=> zero(U))
        state[σ₀] = N
        return new{T}(state)
    end

    @doc raw"""
        MeanFieldFiniteState(N::Integer, ::Val{:rand})

    Construct an spin state with mean field interaction with `N` spins in a random initial state.
    """
    function MeanFieldFiniteState{T}(N::U, ::Val{:rand}) where {T<:AbstractSiteState,U<:Unsigned}
        split_indices = [sort(rand(0:N, instance_count(T) - 1))..., N]
        site_counts = site_counts_from_split_indices(split_indices)
        return new{T}(Dict(instances(T) .=> site_counts))
    end

end

"""
    split_indices(mfss::MeanFieldFiniteState{T}) where {T}

Get a tuple with the values of the split indices `kⱼ` for the mean field finite state `mfss`.
"""
@inline split_indices(mfss::MeanFieldFiniteState{T}) where {T} = cumsum(mfss.state[σᵢ] for σᵢ ∈ instances(T))

"""
    site_counts_from_split_indices(split_indices::Vector{Integer})

Get the number of spins in each state given the split indices `split_indices`.
"""
@inline site_counts_from_split_indices(split_indices::Vector{Unsigned}) = [split_indices[begin], diff(split_indices)]

"""
    clear(mfss::MeanFieldFiniteState{T,U}) where {T<:AbstractSiteState,U<:Unsigned}

Clears the state of the mean field finites state `mfss` by setting the site count to all states to zero.
"""
@inline function clear(mfss::MeanFieldFiniteState{T}) where {T<:AbstractSiteState}
    mfss.state = Dict(instances(T) .=> zero(UInt64))
end

"""
    getindex(mfss::MeanFieldFiniteState{T}, σ::T) where {T<:AbstractSiteState}

Allow the site count for a given state `σ` to be accessed directly using the mean field finite state `mfss`.
"""
@inline Base.getindex(mfss::MeanFieldFiniteState{T}, σ::T) where {T<:AbstractSiteState} = mfss.state[σ]

"""
    setindex!(mfss::MeanFieldFiniteState{T}, Nᵢ::Unsigned, σᵢ::T) where {T<:AbstractSiteState}

Set the site count to `Nᵢ` for a given site `σᵢ` in the mean field finite state `mfss`.
"""
@inline function Base.setindex!(mfss::MeanFieldFiniteState{T}, Nᵢ::Unsigned, σᵢ::T) where {T<:AbstractSiteState}
    mfss.state[σᵢ] = Nᵢ
end

@doc raw"""
    length(spins::MeanFieldFiniteState)

Get the total number of sites `N` in an mean field finite state `mfss`.
"""
@inline Base.length(spins::MeanFieldFiniteState) = sum(values(spins.state))

@doc raw"""
    IndexStyle(::Type{<:MeanFieldFiniteState})

Use only linear indices for the `AbstractVector{AbstractSiteState}` interface for the `MeanFieldFiniteState` type.
"""
@inline Base.IndexStyle(::Type{<:MeanFieldFiniteState}) = IndexLinear()

@doc raw"""
    getindex(mfss::MeanFieldFiniteState{T}, i::Integer) where {T<:SingleSpinState}

Get the state of the `i`-th site in the mean field finite state `mfss`.
"""
@inline function Base.getindex(mfss::MeanFieldFiniteState{T}, i::Integer) where {T<:AbstractSiteState}
    # Iterate on the possible state indices and return if smaller than a given split index
    for (σᵢ, split_index) ∈ zip(instances(T)[1:end-1], split_indices(mfss))
        if i < split_index
            return σᵢ
        end
    end
    # If `i` is not smaller than any split indices, return the last state value
    return instances(T)[end]
end

"""
    setindex!(mfss::MeanFieldFiniteState{T}, σ_new::T, i::Integer) where {T<:AbstractSiteState}

Set the state of the `i`-th site to `σᵢ′` in the mean field finite state `mfss`.
"""
@inline function Base.setindex!(mfss::MeanFieldFiniteState{T}, σᵢ′::T, i::Integer) where {T<:AbstractSiteState}
    σᵢ = mfss[i]
    mfss[σᵢ] -= 1
    mfss[σᵢ′] += 1
end

@doc raw"""
    firstindex(::MeanFieldFiniteState)

Index of the first site in the `AbstractVector{AbstractSiteState}` interface of `MeanFieldFiniteState` is `1`.
"""
@inline Base.firstindex(::MeanFieldFiniteState) = 1

@doc raw"""
    lastindex(mfss::MeanFieldFiniteState)

Index of the last site in the `AbstractVector{AbstractSiteState}` interface of `MeanFieldFiniteState` is equal the total number of sites `N`.
"""
@inline Base.lastindex(mfss::MeanFieldFiniteState) = length(mfss)

"""
    set_state!(mfss::MeanFieldFiniteState{T}, σ₀::T) where {T<:AbstractSiteState}

Set the state of all sites to `σ₀` in a mean field finite state `mfss`.
"""
function set_state!(mfss::MeanFieldFiniteState{T}, σ₀::T) where {T<:AbstractSiteState}
    N = length(mfss)
    # Set all values in the state count to zero
    clear(mfss)
    # Set the selected state site count to `N`
    mfss[σ₀] = N
end

"""
    randomize_state!(mfss::MeanFieldFiniteState{T}) where {T<:AbstractSiteState}

Randomize the state of a mean field finite state `mfss`.
"""
function randomize_state!(mfss::MeanFieldFiniteState{T}) where {T<:AbstractSiteState}
    N = length(mfss)
    split_indices = [sort(rand(0:N, instance_count(T) - 1))..., N]
    site_counts = site_counts_from_split_indices(split_indices)
    mfss.state = Dict(instances(T) .=> site_counts)
end

"""
    nearest_neighbors(mfss::MeanFieldFiniteState)

Get iterator over all pairs of nearest neighbors for the mean field finite system `mfss`.
That is, all possible pairs of sites.
"""
@inline nearest_neighbors(mfss::MeanFieldFiniteState) = Iterators.Stateful((i => j) for i ∈ 2:length(mfss) for j ∈ 1:(i-1))

"""
    nearest_neighbors(mfss::MeanFieldFiniteState, i::Integer)

Get vector with the indices of the nearest neighobrs sites of the `i`-th site in the mean files spin state `mfss`.
That is, all sites except for `i`.
"""
@inline nearest_neighbors(mfss::MeanFieldFiniteState, i::Integer) = Iterators.Stateful([1:(i-1)..., (i+1):length(mfss)...])

"""
    sum(f::Function=identity, mfss::MeanFieldFiniteState)

Get the sum of the states of all sites in the mean field finite state `mfss` with the function `f` applied to each.
"""
@inline Base.sum(f::Function, mfss::MeanFieldFiniteState) =
    sum(mfss.state) do (σᵢ, Nᵢ)
        Nᵢ * f(Integer(σᵢ))
    end
@inline Base.sum(mfss::MeanFieldFiniteState) =
    sum(mfss.state) do (σᵢ, Nᵢ)
        Nᵢ * Integer(σᵢ)
    end

"""
    nearest_neighbors_sum(mfss::MeanFieldFiniteState, i::Integer)

Get sum of the nearest neighbors of site `i` in the mean field finite state `mfss`.
"""
@inline nearest_neighbors_sum(mfss::MeanFieldFiniteState, i::Integer) = sum(mfss) - Integer(mfss[i])

@doc raw"""
    ConcreteFiniteState <: AbstractSpinState

Supertype for all finite state that have a concrete representation of its state in memory
in the form of a concrete array member `state::Array{AbstractSiteState}`.

The whole indexing interface of the `state` can be passed to the `ConcreteFiniteState` object itself.
"""
abstract type ConcreteFiniteState{T,N} <: AbstractFiniteState{T,N} end

"""
    state(cfs::ConcreteFiniteState)

Returns the state representation stored in memory
"""
@inline state(cfs::ConcreteFiniteState) = cfs.state

"""
    length(cfs::ConcreteFiniteState)

Total number of sites of a concrete finite state `cfs`.
"""
@inline Base.length(cfs::ConcreteFiniteState) = length(state(cfs))

"""
    size(cfs::ConcreteFiniteState)

Size of the concrete finite state `cfs`.
"""
@inline Base.size(cfs::ConcreteFiniteState) = size(state(cfs))

"""
    IndexStyle(::Type{<:ConcreteFiniteState{T,N}}) where {T<:AbstractSiteState,N}

Use same indexing style used to index the state array.
"""
@inline Base.IndexStyle(::Type{<:ConcreteFiniteState{T,N}}) where {T<:AbstractSiteState,N} = IndexStyle(Array{T,N})
# @inline Base.IndexStyle(::Type{<:ConcreteFiniteState{T,N}}) where {T,N} = IndexCartesian()
# @inline Base.IndexStyle(::Type{<:ConcreteFiniteState{T,1}}) where {T} = IndexLinear()

"""
    getindex(cfs::ConcreteFiniteState, inds...)

Index the concrete finite state itself to access its state.
"""
@inline Base.getindex(cfs::ConcreteFiniteState, inds...) = getindex(state(cfs), inds...)

"""
    setindex!(cfs::ConcreteFiniteState, σ, inds...)

Index the concrete finite state itself to access its state.
"""
@inline Base.setindex!(cfs::ConcreteFiniteState, σ, inds...) = setindex!(state(cfs), σ, inds...)

"""
    firstindex(cfs::ConcreteFiniteState)

Get the index of the first site in the concrete finite state `cfs`.
"""
@inline Base.firstindex(cfs::ConcreteFiniteState) = firstindex(state(cfs))

"""
    lastindex(spins::ConcreteFiniteState)

Get the index of the last site in the concrete finite state `cfs`.
"""
@inline Base.lastindex(cfs::ConcreteFiniteState) = lastindex(state(cfs))

"""
    set_state!(cfs::ConcreteFiniteState{T}, σ₀::T) where {T<:AbstractSiteState}

Set the state of all sites of a concrete finite state `cfs` to a given state `σ₀`.
"""
@inline function set_state!(cfs::ConcreteFiniteState{T}, σ₀::T) where {T<:AbstractSiteState}
    fill!(state(cfs), σ₀)
end

"""
    randomize_state!(spins::ConcreteFiniteState{T}) where {T<:AbstractSiteState}

Set the state of all sites of a concrete finite state `cfs` to a random state `σ ∈ AbstractSiteState`.
"""
@inline function randomize_state!(spins::ConcreteFiniteState{T}) where {T<:AbstractSiteState}
    rand!(state(spins), instances(T))
end

"""
    nearest_neighbors_sum(cfs::ConcreteFiniteState{T,N}, i::Union{Integer,CartesianIndex{N}}) where {T,N}

Sum of the nearest neighbors of the `i`-th site for a concrete finite state `cfs`.

Every subtype of `ConcreteFiniteState` must provide its own implementation of `nearest_neighbors(cfs::ConcreteFiniteState{T,N}, i::Union{Integer,CartesianIndex{N}})::AbstractVector{IndexStyle(ConcreteFiniteState)}`
"""
@inline nearest_neighbors_sum(cfs::ConcreteFiniteState{T,N}, i::Union{Integer,CartesianIndex{N}}) where {T,N} = @inbounds sum(Integer, cfs[nn] for nn ∈ nearest_neighbors(cfs, i))

"""
    SquareLatticeFiniteState{T,N} <: ConcreteFiniteState{T,N}

Finite state on a `N`-dimensional square lattice.
"""
mutable struct SquareLatticeFiniteState{T,N} <: ConcreteFiniteState{T,N}

    "Multidimensional array with system state"
    state::Array{T,N}

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
    nearest_neighbors(sqlatfs::SquareLatticeFiniteState{T,N}, idx::CartesianIndex{N}) where {T,N}

Gets a vector containing the indices of the nearest neighbors to the `i`-site in the square lattice finite state `sqlatfs`.
"""
@inline nearest_neighbors(sqlatfs::SquareLatticeFiniteState{T,N}, i::CartesianIndex{N}) where {T,N} = @inbounds Lattices.square_lattice_nearest_neighbors_flat(sqlatfs.state, i)

"""
    nearest_neighbors_sum(sqlatfs::SquareLatticeFiniteState{T,N}, i::CartesianIndex{N}) where {T,N}

Sum of the nearest neighbors of the `i`-th site for a multidimensional square lattice finites state `sqlatfs`.
"""
@inline nearest_neighbors_sum(spins::SquareLatticeFiniteState{T,N}, i::CartesianIndex{N}) where {T,N} = @inbounds Lattices.square_lattice_nearest_neighbors_sum(spins.state, i)
# @inline nearest_neighbors_sum(spins::SquareLatticeFiniteState{T,N}, i::Integer) where {T,N} = nearest_neighbors_sum(spins, CartesianIndices(spins)[i])

"""
    SimpleGraphFiniteState{T} <: ConcreteFiniteState{T,1}

Finite state on a simple graph.
"""
mutable struct SimpleGraphFiniteState{T} <: ConcreteFiniteState{T,1}

    "Graph structure of the system"
    graph::SimpleGraph

    "State at each node"
    state::Vector{T}

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
    nearest_neighbors(sgraphfs::SimpleGraphFiniteState)

Get list of all nearest neighbors pairs in the simple graph finite state `sgraphfs`.
"""
@inline nearest_neighbors(sgraphfs::SimpleGraphFiniteState) = edges(sgraphfs.graph)

"""
    nearest_neighbors(sgraphfs::SimpleGraphFiniteState, i::Integer)

Get the indices of the nearest neighbors of `i`-th site in the simple graph finite state `sgraphfs`.
"""
@inline nearest_neighbors(sgraphfs::SimpleGraphFiniteState, i::Integer) = neighbors(sgraphfs.graph, i)

end
