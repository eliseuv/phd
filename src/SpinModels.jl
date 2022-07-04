@doc raw"""
    Spin Models

SpinState{<:SingleSpinState,N}

SpinModels{<:SpinState}
"""
module SpinModels

export SingleSpinState, SpinHalfState, SpinOneState,
    AbstractSpinState, MeanFieldSpinState, ConcreteSpinState, SquareLatticeSpinState,
    SpinModel, IsingModel, BlumeCapelModel,
    set_state!, randomize_state!,
    magnet, magnet_total, magnet_squared_total, magnet_function_total,
    nearest_neighbors, nearest_neighbors_sum,
    energy_interaction

using Random, EnumX, Combinatorics

include("Metaprogramming.jl")
include("Geometry.jl")

using .Metaprogramming

# Single Spin States

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

# Spin States

"""
    AbstractSpinState{T<:SingleSpinState,N} <: AbstractArray{T,N}

Supertype for all spin states.

This type represents whole state of a spin system.

Since all site are indexable, they all inherit from `AbstractArray{SingleSpinState}`.
"""
abstract type AbstractSpinState{T<:SingleSpinState,N} <: AbstractArray{T,N} end

@doc raw"""
    magnet(spins::AbstractSpinState)

Magnetization of the spin system `spins`.

``m = M / N = (1/N) ∑ᵢ sᵢ``
"""
@inline magnet(spins::AbstractSpinState) = magnet_total(spins) / length(spins)

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
abstract type MeanFieldSpinState{T} <: AbstractSpinState{T,1} end

"""
    state(spins::MeanFieldSpinState)

Get the tuple containig the number of sites in each single spin state.
"""
@inline state(spins::MeanFieldSpinState) = spins.state

"""
    split_indices(spins::MeanFieldSpinState{T}) where {T<:SingleSpinState}

Get a tuple with the values of the split indices `kⱼ` for the spin state with mean field interaction `spins`.
"""
@inline split_indices(spins::MeanFieldSpinState{T}) where {T<:SingleSpinState} = cumsum(state(spins)[σᵢ] for σᵢ ∈ instances(T)[1:end-1])

@doc raw"""
    length(spins::MeanFieldSpinState)

Total number of spins (`N`) in an spin system with mean field interaction `spins`
is simply the sum of the number of spins in each state.
"""
@inline Base.length(spins::MeanFieldSpinState) = sum(state(spins))

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
    state(spins)[σ] -= 1
    state(spins)[σ′] += 1
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
    for σ ∈ instances(T)
        state(spins)[σ] = 0
    end
    # Set the selected state count to `N`
    N = length(spins)
    state(spins)[σ₀] = N
end

"""
    randomize_state!(spins::MeanFieldSpinState{T}) where {T<:SingleSpinState}

Randomize the state of a mean field spin state `spins`.
"""
function randomize_state!(spins::MeanFieldSpinState{T}) where {T<:SingleSpinState}
    N = length(spins)
    splits = sort(rand(1:N, state_count(T) - 1))
    state = diff([0, splits..., N])
    for (σᵢ, Nᵢ) ∈ zip(instances(T), state)
        state(spins)[σᵢ] = Nᵢ
    end
end

@doc raw"""
    magnet_total(spins::MeanFieldSpinState)

Total magnetization of mean files spin state `spins`.

``M = ∑ᵢ sᵢ``
"""
@inline magnet_total(spins::MeanFieldSpinState) = @inbounds sum(Nᵢ * Integer(σᵢ) for (σᵢ, Nᵢ) ∈ state(spins))

@doc raw"""
    magnet_squared_total(spins::MeanFieldSpinState)

Total magnetization squared of mean files spin state `spins`.

``∑ᵢ sᵢ²``
"""
@inline magnet_squared_total(spins::MeanFieldSpinState) = @inbounds sum(Nᵢ * Integer(σᵢ)^2 for (σᵢ, Nᵢ) ∈ state(spins))

@doc raw"""
    magnet_function_total(spins::MeanFieldSpinState)

Total magnetization squared of mean files spin state `spins`.

``∑ᵢ f(sᵢ)``
"""
@inline magnet_function_total(f::Function, spins::MeanFieldSpinState) = @inbounds sum(Nᵢ * f(Integer(σᵢ)) for (σᵢ, Nᵢ) ∈ state(spins))

"""
    nearest_neighbors(spins::MeanFieldSpinState, i::Integer)

Get vector with the indices of the nearest neighobrs sites of the `i`-th site in the mean files spin state `spins`.
That is, all sites except for `i`.
"""
@inline nearest_neighbors(spins::MeanFieldSpinState, i::Integer) = [1:(i-1)..., (i+1):length(spins)...]

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
        Nₖ = state(spins)[σₖ]
        return ((Nₖ * (Nₖ - 1)) ÷ 2) * Integer(σₖ)^2
    end
    S_diff = sum(combinations(instances(T), 2)) do (σₖ, σₗ)
        Nₖ = state(spins)[σₖ]
        Nₗ = state(spins)[σₗ]
        return Nₖ * Nₗ * Integer(σₖ) * Integer(σₗ)
    end
    return S_equal + S_diff
end

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
@inline state(spins::ConcreteSpinState) = state(spins)

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
    magnet_total(spins::ConcreteSpinState)

Total magnetization of concrete spin system `spins`.
"""
@inline magnet_total(spins::ConcreteSpinState) = @inbounds sum(Integer, state(spins))

"""
    magnet_squared_total(spins::ConcreteSpinState)

Sum of the squares of all spin states in the concrete spin system `spins`.
"""
@inline magnet_squared_total(spins::ConcreteSpinState) = @inbounds sum(σ -> Integer(σ)^2, state(spins))

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

    state::Array{T,N}

    SquareLatticeSpinState(size::NTuple{N,Integer}, σ₀::T) where {T,N} = new{T,N}(fill(σ₀, size))
    SquareLatticeSpinState(size::NTuple{N,Integer}, ::Val{:rand}) where {T,N} = new{T,N}(rand(instances(T), size))

    SquareLatticeSpinState(::Val{N}, L::Integer, σ₀::T) where {T,N} = SquareLatticeSpinState(ntuple(_ -> L, Val(N)), σ₀)
    SquareLatticeSpinState(::Val{N}, L::Integer, ::Val{:rand}) where {N} = SquareLatticeSpinState(ntuple(_ -> L, Val(N)), Val(:rand))

end

"""
    nearest_neighbors(spins::SquareLatticeSpinState{T,N}, idx::CartesianIndex{N}) where {T,N}

Gets a vector containing the site that are nearest neighbors to a given site `i` in the square lattice spin system `spins`.
"""
@inline nearest_neighbors(spins::SquareLatticeSpinState{T,N}, i::CartesianIndex{N}) where {T,N} = @inbounds Geometry.square_lattice_nearest_neighbors_flat(state(spins), i)

"""
    nearest_neighbors_sum(spins::SquareLatticeSpinState{T,N}, i::CartesianIndex{N}) where {T,N}

Sum of the nearest neighbors of the `i`-th site for a concrete spin model `spins`, optimized for square lattice systems.
"""
@inline nearest_neighbors_sum(spins::SquareLatticeSpinState{T,N}, i::CartesianIndex{N}) where {T,N} = @inbounds Geometry.square_lattice_nearest_neighbors_sum(state(spins), i)
@inline nearest_neighbors_sum(spins::SquareLatticeSpinState{T,N}, i::Integer) where {T,N} = nearest_neighbors_sum(state(spins), CartesianIndices(spins)[i])

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

"""
    SpinModel

Supertype for all spin models.
"""
abstract type SpinModel{T<:AbstractSpinState} end

"""
    spin_state_type(::SpinModel{T}) where {T}

Get the type of the spin state of a given spin model.
"""
@inline spin_state_type(::SpinModel{T}) where {T} = T

"""
    spin_instances(::SpinModel{<:AbstractSpinState{S}}) where {S}

Get
"""
@inline spin_instances(::SpinModel{<:AbstractSpinState{S}}) where {S} = instances(S)

"""
    spins(spinmodel::SpinModel)

Get the spin state associated with a given spin model.
"""
@inline spins(spinmodel::SpinModel) = spinmodel.spins

# Pass measurements from the spin model to the spin state
for func in (:magnet_total, :magnet, :magnet_squared_total, :energy_interaction)
    @eval begin
        @inline $func(spinmodel::SpinModel) = $func(spins(spinmodel))
    end
end

function heatbath_and_measure!(measurement::Function, spinmodel::SpinModel, β::Real, n_steps::Integer)
    # Results vector
    ResultType = Base.return_types(measurement, (typeof(spinmodel),))[1]
    results = Vector{ResultType}(undef, n_steps + 1)
    # Initial measurement
    results[1] = measurement(spinmodel)
    # Sampling loop
    @inbounds for t ∈ 1:n_steps
        # Site loop
        @inbounds for i ∈ rand(eachindex(spins(spinmodel)), length(spins(spinmodel)))
            weights = heatbath_weights(spinmodel, β, i)
            spinmodel[i] = sample([spin_instances(spinmodel)...], weights)
        end
        # Update total magnetization vector
        results[t+1] = measurement(spinmodel)
    end

    return results
end


struct IsingModel{T} <: SpinModel{T}
    spins::T
end

struct BlumeCapelModel{T} <: SpinModel{T}

    spins::T
    D::Real

end

@inline heatbath_weights(bc::BlumeCapelModel, β, i) =
    map([spin_instances(bc)...]) do σ
        h_local = magnet_total_local(state(bc), i)
        s = Integer(σ)
        return exp(β * (h_local * s - bc.D * s^2))
    end |> ProbabilityWeights

end
