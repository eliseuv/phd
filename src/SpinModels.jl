@doc raw"""
    Spin Models

SpinState{<:SingleSpinState,N}

AbstractSpinModels{<:SpinState}
"""
module SpinModels

export SingleSpinState, SpinHalfState, SpinOneState,
    # Properties of single spin states
    state_count, new_rand_state,
    # Spin states
    AbstractSpinState,
    # Measurements in spin states
    magnet_total, magnet_squared_total, magnet_function_total, magnet,
    # Measurements on spins states that depend on locality
    nearest_neighbors_sum,
    energy_interaction,
    # Symmetries explored
    flip!,
    # Implementations of spin states
    MeanFieldSpinState, ConcreteSpinState, SquareLatticeSpinState, GraphSpinState,
    # Setting spin states
    set_state!, randomize_state!,
    # Locality in spin states
    nearest_neighbors,
    # Spin models
    AbstractSpinModel,
    # General properties of spin models
    state_type, single_spin_type, single_spin_values, spins,
    heatbath_weights,
    # General dynamics on spin models
    metropolis_measure!, heatbath_measure!,
    # Families of spin models
    AbstractIsingModel,
    # Implementations of spin models
    IsingModel, BlumeCapelModel,
    # Properties of spin models
    energy, energy_local, minus_energy_local, energy_diff,
    # Critical temperatures
    CriticalTemperature, critical_temperature

using Random, EnumX, Combinatorics, StatsBase, Distributions, Graphs

include("Geometry.jl")

"""
    SpinHalfState::Int8 <: SingleSpinState

Enumeration of possible spin `1/2` values.
"""
@enumx SpinHalfState::Int8 begin
    down = -1
    up = +1
end

"""
    SpinOneState::Int8 <: SingleSpinState

Enumeration of possible spin `1` values.
"""
@enumx SpinOneState::Int8 begin
    down = -1
    zero = 0
    up = +1
end

"""
    SingleSpinState

Supertype for all spin states.

They are usually enums, but even if they are not enums,
all single spin states must provide a method `instances(<:SingleSpinState)` that returns a tuple with all possible single spin states.
"""
SingleSpinState = Union{SpinOneState.T,SpinHalfState.T}

"""
    state_count(::T) where {T<:SingleSpinState}

Get the total number of possible states for a single spin state.
"""
@inline state_count(::T) where {T<:SingleSpinState} = length(instances(T))

"""
    new_rand_state(σ::T) where {T<:SingleSpinState}

Select a new random single spin state `σ′ ∈ SingleSpinState` different from `σ`.
"""
@inline new_rand_state(σ::T) where {T<:SingleSpinState} = rand(filter(!=(σ), instances(T)))

"""
    new_rand_state(σ::SpinHalfState.T)

Returns the complementary of the single spin state `σ`.
"""
@inline new_rand_state(σ::SpinHalfState.T) = SpinHalfState.T(-Integer(σ))

"""
    convert(::Type{T}, σ::SingleSpinState) where {T<:Number}

Use the integer representation of `σ::SingleSpinState` in order to convert it to a numerical type `T<:Number`.
"""
@inline Base.convert(::Type{T}, σ::SingleSpinState) where {T<:Number} = T(Integer(σ))

"""
    promote_rule(T::Type, ::Type{SingleSpinState})

Always try to promote the `SingleSpinState` to whatever the other type is.
"""
@inline Base.promote_rule(T::Type, ::Type{SingleSpinState}) = T

# Arithmetic with numbers and spin states
for op in (:*, :/, :+, :-)
    @eval begin
        @inline Base.$op(x::Number, σ::SingleSpinState) = $op(promote(x, σ)...)
        @inline Base.$op(σ::SingleSpinState, y::Number) = $op(promote(σ, y)...)
    end
end

"""
    *(σ₁::SingleSpinState, σ₂::SingleSpinState)

Multiplication of spin states.
"""
@inline Base.:*(σ₁::SingleSpinState, σ₂::SingleSpinState) = Integer(σ₁) * Integer(σ₂)

"""
    show(io::IO, ::MIME"text/plain", σ::SpinHalfState)

Text representation of `SpinHalfState`.
"""
function Base.show(io::IO, ::MIME"text/plain", σ::SpinHalfState.T)
    spin_char = σ == up ? '↑' : '↓'
    print(io, spin_char)
end

"""
    show(io::IO, ::MIME"text/plain", σ::SpinOneState)

Text representation of `SpinOneState`.
"""
function Base.show(io::IO, ::MIME"text/plain", σ::SpinOneState.T)
    spin_char = σ == up ? '↑' : σ == down ? '↓' : '-'
    print(io, spin_char)
end

"""
    AbstractSpinState{T<:SingleSpinState,N} <: AbstractArray{T,N}

Supertype for all spin states.

This type represents whole state of a spin system.

Since all sites are indexable, they all inherit from `AbstractArray{SingleSpinState}`.
"""
abstract type AbstractSpinState{T<:SingleSpinState,N} <: AbstractArray{T,N} end

"""
    magnet_total(spins::AbstractSpinState)

Total magnetization of concrete spin system `spins`.
"""
@inline magnet_total(spins::AbstractSpinState) = @inbounds sum(Integer, spins)

"""
    magnet_squared_total(spins::AbstractSpinState)

Sum of the squares of all spin states in the concrete spin system `spins`.
"""
@inline magnet_squared_total(spins::AbstractSpinState) = @inbounds sum(sᵢ -> Integer(sᵢ)^2, spins)

@doc raw"""
    magnet_function_total(f::Function, spins::AbstractSpinState)

``∑ᵢ f(sᵢ)``
"""
@inline magnet_function_total(f::Function, spins::AbstractSpinState) = @inbounds sum(sᵢ -> f(Integer(sᵢ)), spins)

@doc raw"""
    magnet(spins::AbstractSpinState)

Magnetization of the spin system `spins`.

``m = M / N = (1/N) ∑ᵢ sᵢ``
"""
@inline magnet(spins::AbstractSpinState) = magnet_total(spins) / length(spins)

"""
    nearest_neighbors_sum(spins::AbstractSpinState, i)

Sum of the spin values of the nearest neighbors of the `i`-th site in the spin state `spins`.
"""
@inline nearest_neighbors_sum(spins::AbstractSpinState, i) = @inbounds sum(Integer, spins[nn] for nn ∈ nearest_neighbors(spins, i))

@doc raw"""
    energy_interaction(spins::AbstractSpinState)

Interaction energy for a spin system `spins`.

``H_{int} = - ∑_⟨i,j⟩ sᵢ sⱼ``
"""
@inline energy_interaction(spins::AbstractSpinState) = @inbounds -sum(Integer(spins[i]) * Integer(spins[j]) for (i, j) ∈ nearest_neighbors(spins))

"""
    flip!(spins::AbstractSpinState{SpinHalfState.T}, i)

Flips the `i`-th spin in the spin-`1/2` state `spins`.
"""
@inline function flip!(spins::AbstractSpinState{SpinHalfState.T}, i)
    @inbounds spins[i] = SpinHalfState.T(-Integer(spins[i]))
end

@doc raw"""
    MeanFieldSpinState{T} <: AbstractSpinState{T,1}

Spin system with mean field interaction:
Every spin interacts equally with every other spin.

Since in the mean field model there is no concept of space and locality,
we represent the state of the system simply by total number of spins in each state.

The `state` member is therefore of type `Dict{<:SingleSpinState,Int64}`.

An `AbstractVector{SingleSpinState}` interface for the `MeanFieldSpinState` type can be implemented
if we assume that the all spin states are stored in a sorted vector.

Assume the possible spin values are `σ₁`, `σ₂`, etc.

    state = (σ₁, σ₁, …, σ₁, σ₂, σ₂, …, σ₂, …)
                            ↑ kᵢ
            |----- N₁ ----||----- N₂ ----|
            |-------------- N ---------------|

We have `N₁` sites at state `σ₁` (all sites below index index `k₁ = N₁ + 1`),
then we have `N₂` sites at state `σ₂` (all sites from index `k₁` and below index `k₂ = N₁ + N₂ + 1 = k₁ + N₂`),
and so on until the last possible single spin state `σₘ`.

The values `k₁`, `k₂`, …, `kₘ₋₁` are called split indices and can be used to access the `i`-th spin site.
Note that we always have `kₘ = N`.

For a `spins::MeanFieldSpinState` we can access the `i`-th spin `sᵢ = spins[i]`:
    sᵢ = σⱼ if kⱼ₋₁ ≤ i < kⱼ
"""
mutable struct MeanFieldSpinState{T} <: AbstractSpinState{T,1}

    # State of the system
    state::Dict{T,Int64}


    @doc raw"""
        MeanFieldSpinState(N::Integer, σ₀::T) where {T<:SingleSpinState}

    Construct spin state with mean field interaction with `N` spins, all in a given initial state `σ₀`.
    """
    function MeanFieldSpinState(N::Integer, σ₀::T) where {T<:SingleSpinState}
        state = Dict(instances(T) .=> zero(Int64))
        state[σ₀] = N
        return new{T}(state)
    end

    @doc raw"""
        MeanFieldSpinState(N::Integer, ::Val{:rand})

    Construct an spin state with mean field interaction with `N` spins in a random initial state.
    """
    function MeanFieldSpinState{T}(N::Integer, ::Val{:rand}) where {T<:SingleSpinState}
        split_indices = sort(rand(0:N, state_count(T) - 1))
        spin_counts = spin_counts_from_split_indices(split_indices, N)
        return new{T}(Dict(instances(T) .=> spin_counts))
    end

end

@doc raw"""
    magnet_total(spins::MeanFieldSpinState)

Total magnetization of mean files spin state `spins`.

``M = ∑ᵢ sᵢ``
"""
@inline magnet_total(spins::MeanFieldSpinState) = @inbounds sum(Nᵢ * Integer(σᵢ) for (σᵢ, Nᵢ) ∈ spins.state)

@doc raw"""
    magnet_squared_total(spins::MeanFieldSpinState)

Total magnetization squared of mean files spin state `spins`.

``∑ᵢ sᵢ²``
"""
@inline magnet_squared_total(spins::MeanFieldSpinState) = @inbounds sum(Nᵢ * Integer(σᵢ)^2 for (σᵢ, Nᵢ) ∈ spins.state)

@doc raw"""
    magnet_function_total(f::Function, spins::MeanFieldSpinState)

``∑ᵢ f(sᵢ)``
"""
@inline magnet_function_total(f::Function, spins::MeanFieldSpinState) = @inbounds sum(Nᵢ * f(Integer(σᵢ)) for (σᵢ, Nᵢ) ∈ spins.state)


"""
    split_indices(spins::MeanFieldSpinState{T}) where {T<:SingleSpinState}

Get a tuple with the values of the split indices `kⱼ` for the spin state with mean field interaction `spins`.
"""
@inline split_indices(spins::MeanFieldSpinState{T}) where {T<:SingleSpinState} = cumsum(spins.state[σᵢ] for σᵢ ∈ instances(T)[1:end-1])

"""
    spin_counts_from_split_indices(split_indices::Vector{Integer}, N::Integer)

Get the number of spins in each state given the split indices `split_indices` and a total number of spins `N`.
"""
@inline spin_counts_from_split_indices(split_indices::Vector{Integer}, N::Integer) = [split_indices[begin], diff([split_indices..., N])]

@doc raw"""
    length(spins::MeanFieldSpinState)

Total number of spins (`N`) in an spin system with mean field interaction `spins`
is simply the sum of the number of spins in each state.
"""
@inline Base.length(spins::MeanFieldSpinState) = sum(spins.state)

@doc raw"""
    IndexStyle(::MeanFieldSpinState)

Use only linear indices for the `AbstractVector{SingleSpinState}` interface for the `MeanFieldSpinState` type.
"""
@inline Base.IndexStyle(::Type{<:MeanFieldSpinState}) = IndexLinear()

@doc raw"""
    getindex(spins::MeanFieldSpinState{T}, i::Integer) where {T<:SingleSpinState}

Get the state of the `i`-th spin in the spin state with mean field interaction `spins`.
"""
@inline function Base.getindex(spins::MeanFieldSpinState{T}, i::Integer) where {T<:SingleSpinState}
    # Iterate on the possible spin indices and return if smaller than a given split index
    for (σᵢ, split_index) ∈ zip(instances(T), split_indices(spins))
        if i < split_index
            return σᵢ
        end
    end
    # If `i` is not smaller than any split indices, return the last spin value
    return instances(T)[end]
end

"""
    setindex!(spins::MeanFieldSpinState{T}, σ_new::T, i::Integer) where {T<:SingleSpinState}

Set the state of the `i`-th spin site to `σ′` in the spin state with mean field interaction `spins`.
"""
@inline function Base.setindex!(spins::MeanFieldSpinState{T}, σ′::T, i::Integer) where {T<:SingleSpinState}
    σ = spins[i]
    spins.state[σ] -= 1
    spins.state[σ′] += 1
end

@doc raw"""
    firstindex(spins::MeanFieldSpinState)

Index of the first spin site in the `AbstractVector{SingleSpinState}` interface of `MeanFieldSpinState` is `1`.
"""
@inline Base.firstindex(spins::MeanFieldSpinState) = 1

@doc raw"""
    lastindex(spins::MeanFieldSpinState)

Index of the last spin site in the `AbstractVector{SingleSpinState}` interface of `MeanFieldSpinState` is equal the total number of sites `N`.
"""
@inline Base.lastindex(spins::MeanFieldSpinState) = length(spins)

"""
    set_state!(spins::MeanFieldSpinState{T}, σ₀::T) where {T<:SingleSpinState}

Set the state of all spins to `σ₀` in a mean field spin state `spins`.
"""
function set_state!(spins::MeanFieldSpinState{T}, σ₀::T) where {T<:SingleSpinState}
    # Set all values in the state count to zero
    spins.state = Dict(instances(T) .=> zero(Int64))
    # Set the selected state count to `N`
    N = length(spins)
    spins.state[σ₀] = N
end

"""
    randomize_state!(spins::MeanFieldSpinState{T}) where {T<:SingleSpinState}

Randomize the state of a mean field spin state `spins`.
"""
function randomize_state!(spins::MeanFieldSpinState{T}) where {T<:SingleSpinState}
    N = length(spins)
    split_indices = sort(rand(0:N, state_count(T) - 1))
    spin_counts = spin_counts_from_split_indices(split_indices, N)
    spins.state = Dict(instances(T) .=> spin_counts)
end

"""
    randomize_state!(spins::MeanFieldSpinState{SpinHalfState.T}, p::Real=0.5)

Set the state of each site of an spin-`1/2` state with mean field interaction `ising` to a random state `σ ∈ {↑, ↓}` with a probability `p` of being `↑`.
"""
@inline function randomize_state!(spins::MeanFieldSpinState{SpinHalfState.T}, p::Real)
    N = length(spins)
    dist = Binomial(N, p)
    N₊ = rand(dist)
    N₋ = N - N₊
    spins.state = Dict(SpinHalfState.up => N₊,
        SpinHalfState.down => N₋)
end
@inline function randomize_state!(spins::MeanFieldSpinState{SpinHalfState.T})
    N = length(spins)
    N₊ = rand(0:N)
    N₋ = N - N₊
    spins.state = Dict(SpinHalfState.up => N₊,
        SpinHalfState.down => N₋)
end


"""
    nearest_neighbors(spins::MeanFieldSpinState)

Get iterator over all pairs of nearest neoghbors for the mean field spin system `spins`.
"""
@inline nearest_neighbors(spins::MeanFieldSpinState) = Iterators.Stateful((i => j) for i ∈ 2:length(spins) for j ∈ 1:(i-1))

"""
    nearest_neighbors(spins::MeanFieldSpinState, i::Integer)

Get vector with the indices of the nearest neighobrs sites of the `i`-th site in the mean files spin state `spins`.
That is, all sites except for `i`.
"""
@inline nearest_neighbors(spins::MeanFieldSpinState, i::Integer) = Iterators.Stateful([1:(i-1)..., (i+1):length(spins)...])

"""
    nearest_neighbors_sum(spins::MeanFieldSpinState, i::Integer)

Get sum of the nearest neighbors spins of site `i` in the mean field spin state `spins`.
"""
@inline nearest_neighbors_sum(spins::MeanFieldSpinState, i::Integer) = magnet_total(spins) - Integer(spins[i])

"""
    energy_interaction(spins::MeanFieldSpinState)

Get the interaction energy of the mean field spin state `spins`.
"""
function energy_interaction(spins::MeanFieldSpinState{T}) where {T<:SingleSpinState}
    S_equal = sum(instances(T)) do σₖ
        Nₖ = spins.state[σₖ]
        return ((Nₖ * (Nₖ - 1)) ÷ 2) * Integer(σₖ)^2
    end
    S_diff = sum(combinations(instances(T), 2)) do (σₖ, σₗ)
        Nₖ = spins.state[σₖ]
        Nₗ = spins.state[σₗ]
        return Nₖ * Nₗ * Integer(σₖ) * Integer(σₗ)
    end
    return S_equal + S_diff
end

@doc raw"""
    flip!(spins::MeanFieldSpinState{SpinHalfState.T}, i::Integer)

Flip the state of the `i`-th spin in the spin-`1/2` state with mean field interaction `spins`.
"""
@inline function flip!(spins::MeanFieldSpinState{SpinHalfState.T}, i::Integer)
    sᵢ = Integer(spins[i])
    spins.state[SpinHalfState.up] -= sᵢ
    spins.state[SpinHalfState.down] += sᵢ
end

@doc raw"""
    energy_interaction(spins::MeanFieldSpinState{SpinHalfState.T})

Interaction energy of the spin-`1/2` spin state with mean field interaction `spins`.

``H_{int} = - ∑_⟨i,j⟩ sᵢsⱼ = (N - M^2) / 2``
"""
@inline energy_interaction(spins::MeanFieldSpinState{SpinHalfState.T}) = (length(spins) - magnet_total(spins)^2) ÷ 2

@doc raw"""
    ConcreteSpinState <: AbstractSpinState

Supertype for all spin models that have a concrete representation of its state in memory
in the form of a concrete array member `state::Array{SingleSpinState}`.

The whole indexing interface of the `state` can be passed to the `ConcreteSpinState` object itself.
"""
abstract type ConcreteSpinState{T,N} <: AbstractSpinState{T,N} end

"""
    state(spins::ConcreteSpinState)

Returns the state of the spin system stored in memory
"""
@inline state(spins::ConcreteSpinState) = spins.state

"""
    length(spins::ConcreteSpinState)

Total number of sites of an spin system `spins`.
"""
@inline Base.length(spins::ConcreteSpinState) = length(state(spins))

"""
    size(spins::ConcreteSpinState)

Size of the state of an spin system `spins`.
"""
@inline Base.size(spins::ConcreteSpinState) = size(state(spins))

"""
    IndexStyle(::Type{<:ConcreteSpinState{T,N}}) where {T<:SingleSpinState,N}

Use same indexing style used to index the state array.
"""
@inline Base.IndexStyle(::Type{<:ConcreteSpinState{T,N}}) where {T<:SingleSpinState,N} = IndexStyle(Array{T,N})
# @inline Base.IndexStyle(::Type{<:ConcreteSpinState{T,N}}) where {T,N} = IndexCartesian()
# @inline Base.IndexStyle(::Type{<:ConcreteSpinState{T,1}}) where {T} = IndexLinear()

"""
    getindex(spins::ConcreteSpinState, inds...)

Index the spin system itself to access its state.
"""
@inline Base.getindex(spins::ConcreteSpinState, inds...) = getindex(state(spins), inds...)

"""
    setindex!(spins::ConcreteSpinState, σ, inds...)

Set the state of a given spin at site `i` to `σ` in the spin system `spins`.
"""
@inline Base.setindex!(spins::ConcreteSpinState, σ, inds...) = setindex!(state(spins), σ, inds...)

"""
    firstindex(spins::ConcreteSpinState)

Get the index of the first spin in the system.
"""
@inline Base.firstindex(spins::ConcreteSpinState) = firstindex(state(spins))

"""
    lastindex(spins::ConcreteSpinState)

Get the index of the last spin in the system.
"""
@inline Base.lastindex(spins::ConcreteSpinState) = lastindex(state(spins))

"""
    set_state!(spins::ConcreteSpinState{T}, σ₀::T) where {T<:SingleSpinState}

Set the state of all sites of an Spinmodel system `spins` to a given site state `σ₀`.
"""
@inline function set_state!(spins::ConcreteSpinState{T}, σ₀::T) where {T<:SingleSpinState}
    fill!(state(spins), σ₀)
end

"""
    randomize_state!(spins::ConcreteSpinState{T}) where {T<:SingleSpinState}

Set the state of each site of an spin system `spins` to a random state `σ₀ ∈ SingleSpinState`.
"""
@inline function randomize_state!(spins::ConcreteSpinState{T}) where {T<:SingleSpinState}
    rand!(state(spins), instances(T))
end

"""
    nearest_neighbors_sum(spins::ConcreteSpinState{T,N}, i::Union{Integer,CartesianIndex{N}}) where {T,N}

Sum of the nearest neighbors of the `i`-th site for a concrete spin model `spins`.

Every `spins::ConcreteSpinState` subtype must provide its own implementation of `nearest_neighbors(spins::ConcreteSpinState{T,N}, i::Union{Integer,CartesianIndex{N}})::AbstractVector{IndexStyle(ConcreteSpinState)}`
"""
@inline nearest_neighbors_sum(spins::ConcreteSpinState{T,N}, i::Union{Integer,CartesianIndex{N}}) where {T,N} = @inbounds sum(Integer, spins[nn] for nn ∈ nearest_neighbors(spins, i))

"""
    SquareLatticeSpinState{T,N} <: ConcreteSpinState{T,N}

Spin models on a `N`-dimensional square lattice.
"""
mutable struct SquareLatticeSpinState{T,N} <: ConcreteSpinState{T,N}

    "Multidimensional array with system state"
    state::Array{T,N}

    """
        SquareLatticeSpinState(size::NTuple{N,Integer}, σ₀::T) where {T,N}

    Construct a new spin state in a multidimensional square lattice of dimensions provided by `size`,
    with nearest neighbor interaction and with all spins with same initial state `σ₀`.
    """
    SquareLatticeSpinState(size::NTuple{N,Integer}, σ₀::T) where {T,N} = new{T,N}(fill(σ₀, size))


    """
        SquareLatticeSpinState(size::NTuple{N,Integer}, ::Val{:rand}) where {T,N}

    Construct a new spin system in a multidimensional square lattice of dimensions provided by `size`,
    with nearest neighbor interaction and with spins in random states.
    """
    SquareLatticeSpinState(size::NTuple{N,Integer}, ::Type{T}, ::Val{:rand}) where {T,N} = new{T,N}(rand(instances(T), size))

    """
        SquareLatticeSpinState(::Val{N}, L::Integer, σ₀::T) where {T,N}

    Construct a `dim`-dimensional square spin system of side length `L` and a given initial state `σ₀`.
    """
    SquareLatticeSpinState(::Val{N}, L::Integer, σ₀::T) where {T,N} = SquareLatticeSpinState(ntuple(_ -> L, Val(N)), σ₀)

    """
        SquareLatticeSpinState(::Val{N}, L::Integer, ::Val{:rand}) where {N}

    Construct a `dim`-dimensional square spin system of side length `L` and random initial state.
    """
    SquareLatticeSpinState(::Val{N}, L::Integer, ::Type{T}, ::Val{:rand}) where {T,N} = SquareLatticeSpinState(ntuple(_ -> L, Val(N)), T, Val(:rand))

end

"""
    IndexStyle(::Type{<:SquareLatticeSpinState})

Prefer cartesian indices for multidimensional square lattice spin states.
"""
Base.IndexStyle(::Type{<:SquareLatticeSpinState}) = IndexCartesian()

"""
    nearest_neighbors(spins::SquareLatticeSpinState{T,N}, idx::CartesianIndex{N}) where {T,N}

Gets a vector containing the site that are nearest neighbors to a given site `i` in the square lattice spin system `spins`.
"""
@inline nearest_neighbors(spins::SquareLatticeSpinState{T,N}, i::CartesianIndex{N}) where {T,N} = @inbounds Geometry.square_lattice_nearest_neighbors_flat(spins.state, i)

"""
    nearest_neighbors_sum(spins::SquareLatticeSpinState{T,N}, i::CartesianIndex{N}) where {T,N}

Sum of the nearest neighbors of the `i`-th site for a concrete spin model `spins`, optimized for square lattice systems.
"""
@inline nearest_neighbors_sum(spins::SquareLatticeSpinState{T,N}, i::CartesianIndex{N}) where {T,N} = @inbounds Geometry.square_lattice_nearest_neighbors_sum(spins.state, i)
# @inline nearest_neighbors_sum(spins::SquareLatticeSpinState{T,N}, i::Integer) where {T,N} = nearest_neighbors_sum(spins, CartesianIndices(spins)[i])

@doc raw"""
    energy_interaction(spins::SquareLatticeSpinState{N}) where {N}

Interaction energy for a square lattice spin model `spins`.

``H_{int} = - \sum_⟨i,j⟩ σᵢ σⱼ``
"""
function energy_interaction(spins::SquareLatticeSpinState{T,N}) where {T,N}
    # Varaible to accumulate
    H = zero(Int64)
    # Loop on dimensions
    @inbounds for d ∈ 1:N
        # Bulk
        front_bulk = selectdim(state(spins), d, 1:(size(spins, d)-1))
        back_bulk = selectdim(state(spins), d, 2:size(spins, d))
        H -= sum(Integer, front_bulk .* back_bulk)
        # Periodic boundaries
        last_slice = selectdim(state(spins), d, size(spins, d))
        first_slice = selectdim(state(spins), d, 1)
        H -= sum(Integer, last_slice .* first_slice)
    end
    return H
end

mutable struct GraphSpinState{T} <: ConcreteSpinState{T,1}

    "Graph structure of the system"
    graph::SimpleGraph

    "State at each node"
    state::Vector{T}

    """
        GraphSpinState(graph::Graph, σ₀::T) where {T}

    Construct a new spin system with graph structure `graph` with all spins with same initial state `σ₀`.
    """
    GraphSpinState(graph::SimpleGraph, σ₀::T) where {T} = new{T}(graph, fill(σ₀, nv(graph)))

    """
        IsingGraph(g::Graph, ::Val{:rand})

    Construct a new Ising system with graph structure `g` and random initial states at each node.
    """
    GraphSpinState(graph::SimpleGraph, ::Val{:rand}) where {T} = new{T}(graph, rand(instances(T), nv(graph)))

end

"""
    nearest_neighbors(spins::GraphSpinState)

Get list of all nearest neighbors pairs in the spin state `spins` over a graph.
"""
@inline nearest_neighbors(spins::GraphSpinState) = edges(spins.graph)

"""
    nearest_neighbors(spins::GraphSpinState, i::Integer)

Get the indices of the nearest neighbors of `i`-th site in the spin state on a graph `spins`.
"""
@inline nearest_neighbors(spins::GraphSpinState, i::Integer) = neighbors(spins.graph, i)

"""
    energy_interaction(spins::GraphSpinState)

Get the interaction energy for a spin state on a graph `spins`.
"""
@inline energy_interaction(spins::GraphSpinState) = @inbounds -sum(spins[src(e)] * spins[dst(e)] for e ∈ edges(spins.graph))

"""
    AbstractSpinModel

Supertype for all spin models.
"""
abstract type AbstractSpinModel{T,N} <: AbstractArray{T,N} end

"""
    single_spin_type(::AbstractSpinModel)

Get the type of the single spin state of a given spin model.
"""
@inline single_spin_type(::AbstractSpinModel{T}) where {T} = T

"""
    single_spin_values(::AbstractSpinModel)

Get a tuple with the possible instances of single spin state.
"""
@inline single_spin_values(::AbstractSpinModel{T}) where {T} = instances(T)

"""
    spins(spinmodel::AbstractSpinModel)

Get the spin state associated with a given spin model.
"""
@inline spins(spinmodel::AbstractSpinModel) = spinmodel.spins

"""
    state_type(::AbstractSpinModel)

Get the type of the spin state of a given spin model.
"""
# @inline state_type(spinmodel::AbstractSpinModel) = typeof(spins(spinmodel))

"""
    length(spinmodel::AbstractSpinModel)

Total number of sites of an spin system `spinmodel`.
"""
@inline Base.length(spinmodel::AbstractSpinModel) = length(spins(spinmodel))

"""
    size(spinmodel::AbstractSpinModel)

Size of the spins of an spin system `spinmodel`.
"""
@inline Base.size(spinmodel::AbstractSpinModel) = size(spins(spinmodel))

"""
    IndexStyle(::Type{<:AbstractSpinModel{T,N}}) where {T,N}

Use linear indices for spin models in one dimension, and cartesian indices otherwise.
"""
@inline Base.IndexStyle(::Type{<:AbstractSpinModel{T,N}}) where {T,N} = IndexCartesian()
@inline Base.IndexStyle(::Type{<:AbstractSpinModel{T,1}}) where {T} = IndexLinear()

"""
    getindex(spinmodel::AbstractSpinModel, inds...)

Index the spin system itself to access its spins.
"""
@inline Base.getindex(spinmodel::AbstractSpinModel, inds...) = getindex(spins(spinmodel), inds...)

"""
    setindex!(spinmodel::AbstractSpinModel, σ, inds...)

Set the spins of a given spin at site `i` to `σ` in the spin system `spinmodel`.
"""
@inline Base.setindex!(spinmodel::AbstractSpinModel, σ, inds...) = setindex!(spins(spinmodel), σ, inds...)

"""
    firstindex(spinmodel::AbstractSpinModel)

Get the index of the first spin in the system.
"""
@inline Base.firstindex(spinmodel::AbstractSpinModel) = firstindex(spins(spinmodel))

"""
    lastindex(spinmodel::AbstractSpinModel)

Get the index of the last spin in the system.
"""
@inline Base.lastindex(spinmodel::AbstractSpinModel) = lastindex(spins(spinmodel))

# Allow spin state measurements to be done directly on the spin model
for func in (:magnet_total, :magnet_squared_total, :magnet_function_total, :magnet, :energy_interaction)
    @eval begin
        @inline $func(spinmodel::AbstractSpinModel) = $func(spins(spinmodel))
    end
end

"""
    set_state!(spinmodel::AbstractSpinModel{<:AbstractSpinState{S}}, σ₀::S) where {S}

Set the state of all spins in a given spin model `spinmodel` to a given single spin state `σ₀`.
"""
@inline set_state!(spinmodel::AbstractSpinModel{T}, σ₀::T) where {T} = set_state!(spins(spinmodel), σ₀)

"""
    randomize_state!(spinmodel::AbstractSpinModel)

Randomize the state of a given spin model `spinmodel`.
"""
@inline randomize_state!(spinmodel::AbstractSpinModel) = randomize_state!(spins(spinmodel))
@inline randomize_state!(spinmodel::AbstractSpinModel, args...) = randomize_state!(spins(spinmodel), args...)

# Pass locality functions from the spin model to the spin state
@inline nearest_neighbors(spinmodel::AbstractSpinModel) = nearest_neighbors(spins(spinmodel))
for func in (:nearest_neighbors, :nearest_neighbors_sum)
    @eval begin
        @inline $func(spinmodel::AbstractSpinModel, i) = $func(spins(spinmodel), i)
    end
end

"""
    metropolis_measure!(measurement::Function, spinmodel::AbstractSpinModel, β::Real, n_steps::Integer)

Metropolis sample the spin model `spinmodel` at temperature `β` for `n_steps`
and perform the measurement `measurement` on the spin model at the end of each step.

Note that a single sampling step is equivalent to `N` metropolis prescription steps,
where `N` is the total number of sites in the system.
"""
function metropolis_measure!(measurement::Function, spinmodel::T, β::Real, n_steps::Integer) where {T<:AbstractSpinModel}
    # Results vector
    ResultType = Base.return_types(measurement, (T,))[1]
    results = Vector{ResultType}(undef, n_steps + 1)
    # Initial measurement
    results[1] = measurement(spinmodel)
    # Sampling loop
    @inbounds for t ∈ 1:n_steps
        # Site loop
        @inbounds for i ∈ rand(eachindex(spinmodel), length(spinmodel))
            # Select random new state
            sᵢ′ = new_rand_state(spinmodel[i])
            # Get energy difference
            ΔH = energy_diff(spinmodel, i, sᵢ′)
            # Metropolis prescription
            if ΔH < 0 || exp(-β * ΔH) > rand()
                # Change spin
                spinmodel[i] = sᵢ′
            end
        end
        # Update results vector
        results[t+1] = measurement(spinmodel)
    end
    # Return measurement results
    return results
end

"""
    metropolis_measure!(measurement::Function, spinmodel::AbstractSpinModel{<:AbstractSpinState{SpinHalfState.T}}, β::Real, n_steps::Integer)

Metropolis sample the spin-`1/2` model `spinmodel` at temperature `β` for `n_steps`
and perform the measurement `measurement` on the spin model at the end of each step.

Note that a single sampling step is equivalent to `N` metropolis prescription steps,
where `N` is the total number of sites in the system.

This implementation takes advantage of symmetries in spin-`1/2` systems to simplify the algorithm.
"""
function metropolis_measure!(measurement::Function, spinmodel::T, β::Real, n_steps::Integer) where {T<:AbstractSpinModel{SpinHalfState.T}}
    # Results vector
    ResultType = Base.return_types(measurement, (T,))[1]
    results = Vector{ResultType}(undef, n_steps + 1)
    # Initial measurement
    results[1] = measurement(spinmodel)
    # Sampling loop
    @inbounds for t ∈ 1:n_steps
        # Site loop
        @inbounds for i ∈ rand(eachindex(spinmodel), length(spinmodel))
            # Get energy difference
            ΔH = energy_diff(spinmodel, i)
            # Metropolis prescription
            if ΔH < 0 || exp(-β * ΔH) > rand()
                # Flip spin
                flip!(spinmodel.spins, i)
            end
        end
        # Update results vector
        results[t+1] = measurement(spinmodel)
    end
    # Return measurement results
    return results
end

@doc raw"""
    heatbath_weights(spinmodel::AbstractSpinModel, i, β::Real)

Calculate the weights required by the heatbath sampling algorithm for the spin model `spinmodel` for the `i`-th site and at temperature `β`.

The weight associated with the single spin state `σ` at the `i`-th site at temperature `β` is:

``w(σ, i, β) = exp(-β hᵢ(σ))``

where `hᵢ(σ)` is the local energy associated with `i`-th site assuming that its state is `σ`.
"""
@inline heatbath_weights(spinmodel::AbstractSpinModel{T}, i, β::Real) where {T} =
    map(σ -> exp(β * minus_energy_local(spinmodel, i, σ)), [instances(T)...]) |> ProbabilityWeights

"""
    heatbath_measure!(measurement::Function, spinmodel::AbstractSpinModel, β::Real, n_steps::Integer)

Heatbath sample the spin model `spinmodel` at temperature `β` for `n_steps`
and perform the measurement `measurement` on the spin model at the end of each step.

Note that a single system sampling step is equivalent to `N` heatbath prescription steps,
where `N` is the total number of sites in the system.
"""
function heatbath_measure!(measurement::Function, spinmodel::T, β::Real, n_steps::Integer) where {S,T<:AbstractSpinModel{S}}
    # Results vector
    ResultType = Base.return_types(measurement, (T,))[1]
    results = Vector{ResultType}(undef, n_steps + 1)
    # Initial measurement
    results[1] = measurement(spinmodel)
    # Sampling loop
    @inbounds for t ∈ 1:n_steps
        # Site loop
        @inbounds for i ∈ rand(eachindex(spinmodel), length(spinmodel))
            weights = heatbath_weights(spinmodel, i, β)
            spinmodel[i] = sample([instances(S)...], weights)
        end
        # Update measurement vector
        results[t+1] = measurement(spinmodel)
    end
    # Return measurement results
    return results
end

@doc raw"""
    IsingModel{N} <: AbstractSpinModel{SpinHalfState.T,N}

The Ising model without external magnetic field.
"""
struct IsingModel{N} <: AbstractSpinModel{SpinHalfState.T,N}

    "State of the spins"
    spins::AbstractSpinState{SpinHalfState.T,N}

    """
        IsingModel(spins::AbstractSpinState{SpinHalfState.T,N}) where {N}

    Construct an Ising system without external magnetic field and with given initial spins state `spins`
    """
    IsingModel(spins::AbstractSpinState{SpinHalfState.T,N}) where {N} = new{N}(spins)
end

@doc raw"""
    energy(ising::IsingModel)

Total energy of an Ising system `ising`.

Given by the Hamiltonian:

``H = - ∑_⟨i,j⟩ sᵢsⱼ``

where `⟨i,j⟩` means that `i` and `j` are nearest neighbors.
"""
@inline energy(ising::IsingModel) = energy_interaction(ising)

@doc raw"""
    energy_local(ising::IsingModel, i)

Local energy of the `i`-th site in the Ising system `ising`.

``hᵢ = - sᵢ ∑ⱼ sⱼ``

where the sum is over the nearest neighbors `j` of `i`.
"""
@inline energy_local(ising::IsingModel, i) = -Integer(ising[i]) * nearest_neighbors_sum(ising, i)

@doc raw"""
    energy_local(ising::IsingModel, i, sᵢ)

Local energy of the `i`-th site assuming its state is `sᵢ` in the Ising system `ising`.

``hᵢ = - sᵢ ∑ⱼ sⱼ``

where the sum is over the nearest neighbors `j` of `i`.
"""
@inline energy_local(ising::IsingModel, i, sᵢ) = -Integer(sᵢ) * nearest_neighbors_sum(ising, i)

@doc raw"""
    energy_diff(ising::IsingModel, i)

Calculate the energy difference for an Ising system `ising` if the `i`-th spin were to be flipped.

``ΔHᵢ = 2 sᵢ ∑ⱼ sⱼ``

where the sum is over the nearest neighbors `j` of `i`.
"""
@inline energy_diff(ising::IsingModel, i) = 2 * Integer(ising[i]) * nearest_neighbors_sum(ising, i)

@doc raw"""
    energy_diff(ising::IsingModel, i, sᵢ′)

Calculate the energy difference for an Ising system `ising` if the `i`-th spin were to be changed to `sᵢ′`.

``ΔHᵢ = (sᵢ - sᵢ′) ∑ⱼ sⱼ``

where the sum is over the nearest neighbors `j` of `i`.
"""
@inline function energy_diff(ising::IsingModel, i, sᵢ′)
    sᵢ = ising[i]
    if sᵢ′ == sᵢ
        return 0
    else
        return (Integer(sᵢ) - Integer(sᵢ′)) * nearest_neighbors_sum(ising, i)
    end
end

@doc raw"""
    IsingModelExtField{N} <: AbstractSpinModel{SpinHalfState.T,N}

The Ising model with external magnetic field.
"""
struct IsingModelExtField{N} <: AbstractSpinModel{SpinHalfState.T,N}

    "State of the spins"
    spins::AbstractSpinState{SpinHalfState.T,N}

    "External magnetic field"
    h::Real

    """
        IsingModelExtField(spins::AbstractSpinState{SpinHalfState.T,N}, h::Real) where {N}

    Construct an Ising system with external magnetic field `h` and with given initial spins state `spins`
    """
    IsingModelExtField(spins::AbstractSpinState{SpinHalfState.T,N}, h::Real) where {N} = new{N}(spins, h)
end

@doc raw"""
    energy(ising::IsingModel)

Total energy of an Ising system `ising`.

Given by the Hamiltonian:

``H = - ∑_⟨i,j⟩ sᵢsⱼ - h ∑ᵢ sᵢ``

where `⟨i,j⟩` means that `i` and `j` are nearest neighbors.
"""
@inline energy(ising::IsingModelExtField) = energy_interaction(ising) - ising.h * magnet_total(ising)

@doc raw"""
    energy_local(ising::IsingModel, i)

Local energy of the `i`-th site in the Ising system `ising`.

``hᵢ = - sᵢ (∑ⱼ sⱼ + h)``

where the sum is over the nearest neighbors `j` of `i`.
"""
@inline energy_local(ising::IsingModelExtField, i) = -Integer(ising[i]) * (nearest_neighbors_sum(ising, i) + ising.h)

@doc raw"""
    energy_local(ising::IsingModel, i, sᵢ)

Local energy of the `i`-th site assuming its state is `sᵢ` in the Ising system `ising`.

``hᵢ = - sᵢ (∑ⱼ sⱼ + h)``

where the sum is over the nearest neighbors `j` of `i`.
"""
@inline energy_local(ising::IsingModelExtField, i, sᵢ) = -Integer(sᵢ) * (nearest_neighbors_sum(ising, i) + ising.h)


@doc raw"""
    energy_diff(ising::IsingModelExtField, i)

Calculate the energy difference for an Ising system `ising` if the `i`-th spin were to be flipped.

``ΔHᵢ = 2 sᵢ (∑ⱼ sⱼ + h)``

where the sum is over the nearest neighbors `j` of `i`.
"""
@inline energy_diff(ising::IsingModelExtField, i) = 2 * Integer(ising[i]) * (nearest_neighbors_sum(ising, i) + ising.h)

@doc raw"""
    energy_diff(ising::IsingModelExtField, i, sᵢ′)

Calculate the energy difference for an Ising system `ising` if the `i`-th spin were to be changed to `sᵢ′`.

``ΔHᵢ = (sᵢ - sᵢ′) ( ∑ⱼ sⱼ + h)``

where the sum is over the nearest neighbors `j` of `i`.
"""
@inline function energy_diff(ising::IsingModelExtField, i, sᵢ′)
    sᵢ = ising[i]
    if sᵢ′ == sᵢ
        return 0
    else
        return (Integer(sᵢ) - Integer(sᵢ′)) * (nearest_neighbors_sum(ising, i) + ising.h)
    end
end

@doc raw"""
    BlumeCapelModel{N} <: AbstractSpinModel{SpinOneState.T,N}

Blume-Capel model without external mangnetic field.

``H = - ∑_⟨i,j⟩ sᵢsⱼ + D ∑ᵢ sᵢ²``
"""
struct BlumeCapelModel{N} <: AbstractSpinModel{SpinOneState.T,N}

    "State of the spins"
    spins::AbstractSpinState{SpinOneState.T,N}

    "Parameter"
    D::Real

    """
        BlumeCapelModel(spins::AbstractSpinState{SpinOneState.T,N}, D::Real) where {N}

    Construct an Blume-Capel system without external magnetic field and with given initial spins state `spins` and parameter `D`.
    """
    BlumeCapelModel(spins::AbstractSpinState{SpinOneState.T,N}, D::Real) where {N} = new{N}(spins, D)

end

@doc raw"""
    energy(blumecapel::BlumeCapelModel)

Total energy of an Blume-Capel system `blumecapel`.

Given by the Hamiltonian:

``H = - ∑_⟨i,j⟩ sᵢsⱼ + D ∑ᵢ sᵢ²``

where `⟨i,j⟩` means that `i` and `j` are nearest neighbors.
"""
@inline energy(blumecapel::BlumeCapelModel) = energy_interaction(blumecapel) + blumecapel.D * magnet_squared_total(blumecapel)

@doc raw"""
    energy_local(blumecapel::BlumeCapelModel, i)

Local energy of the `i`-th site in the Blume-Capel system `blumecapel`.

``hᵢ = - sᵢ ∑ⱼ sⱼ + D sᵢ²``

where the sum is over the nearest neighbors `j` of `i`.
"""
@inline energy_local(blumecapel::BlumeCapelModel, i) =
    let sᵢ = Integer(blumecapel[i])
        blumecapel.D * sᵢ^2 - sᵢ * nearest_neighbors_sum(blumecapel, i)
    end

@doc raw"""
    energy_local(blumecapel::BlumeCapelModel, i, sᵢ)

Local energy of the `i`-th site assuming its state is `sᵢ` in the Blume-Capel system `blumecapel`.

``hᵢ(sᵢ) = - sᵢ ∑ⱼ sⱼ + D sᵢ²``

where the sum is over the nearest neighbors `j` of `i`.
"""
@inline energy_local(blumecapel::BlumeCapelModel, i, sᵢ) =
    let sᵢ = Integer(sᵢ)
        blumecapel.D * sᵢ^2 - sᵢ * nearest_neighbors_sum(blumecapel, i)
    end

@doc raw"""
    minus_energy_local(blumecapel::BlumeCapelModel, i, sᵢ)

*Minus* the value of the local energy of the `i`-th site assuming its state is `sᵢ` in the Blume-Capel system `blumecapel`.

``-hᵢ(sᵢ) = sᵢ ∑ⱼ sⱼ - D sᵢ²``

where the sum is over the nearest neighbors `j` of `i`.
"""
@inline minus_energy_local(blumecapel::BlumeCapelModel, i, sᵢ) =
    let sᵢ = Integer(sᵢ)
        sᵢ * nearest_neighbors_sum(blumecapel, i) - blumecapel.D * sᵢ^2
    end

@doc raw"""
    energy_diff(blumecapel::BlumeCapelModel, i, sᵢ′)

Calculate the energy difference for an Blume-Capel system `blumecapel` if the `i`-th spin were to be changed to `sᵢ′`.

``ΔHᵢ = (sᵢ - sᵢ′) ∑ⱼ sⱼ + D (sᵢ′² - sᵢ²)``

where the sum is over the nearest neighbors `j` of `i`.
"""
@inline energy_diff(blumecapel::BlumeCapelModel, i, sᵢ′) =
    let sᵢ = Integer(blumecapel[i]), sᵢ′ = Integer(sᵢ′)
        if sᵢ′ == sᵢ
            0
        else
            (sᵢ - sᵢ′) * nearest_neighbors_sum(blumecapel, i) + blumecapel.D * (sᵢ′^2 - sᵢ^2)
        end
    end

end
