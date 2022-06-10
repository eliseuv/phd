@doc raw"""
    Brass Cellular Automaton

The Brass cellular automaton models the polarization of T-helper cells in response to parasitic infections as described by [Tome and Drugowich de Felicio, 1996](https://doi.org/10.1007/BF01857724).

The probabilities `p` and `r` are parameters of the model:

- `p`: Probability associated with antigen concentration
- `r`: Probability associated with mean lifetime of cells TH1 and TH2

The transition probabilities wᵢ(σ₁′|σᵢ,sᵢ) for a site to go from state σᵢ to σᵢ′ given that the sign of the sum of its nearest neighbors is sᵢ are:
    | wᵢ(0|σᵢ,sᵢ) = (1-p) δ(σᵢ,0) + r [δ(σᵢ,+1) + δ(σᵢ,-1)]
    | wᵢ(+1|σᵢ,sᵢ) = p δ(σᵢ,0) [δ(sᵢ,+1) + 0.5 δ(sᵢ,0)] + (1-r) δ(σᵢ,+1)
    | wᵢ(-1|σᵢ,sᵢ) = p δ(σᵢ,0) [δ(sᵢ,-1) + 0.5 δ(sᵢ,0)] + (1-r) δ(σᵢ,-1)

Assuming σᵢ = 0:
⟹   | wᵢ(0|σᵢ=0,sᵢ) = 1 - p
    | wᵢ(+1|σᵢ=0,sᵢ) = p [δ(sᵢ,+1) + 0.5 δ(sᵢ,0)]
    | wᵢ(-1|σᵢ=0,sᵢ) = p [δ(sᵢ,-1) + 0.5 δ(sᵢ,0)]

Assuming σᵢ = ±1:
⟹   | wᵢ(0|σᵢ=±1,sᵢ) = r
    | wᵢ(+1|σᵢ=±1,sᵢ) = (1-r) δ(σᵢ,+1)
    | wᵢ(-1|σᵢ=±1,sᵢ) = (1-r) δ(σᵢ,-1)

We define W₀ as the probability for the new state to be σᵢ′ = 0:
    W₀ = wᵢ(0|σᵢ,sᵢ)
and W₁ as the probability for the new state to be either σᵢ′ = 0 or σᵢ′ = +1:
    W₁ = W₀ + wᵢ(+1|σᵢ,sᵢ)

If σᵢ = 0:
    | W₀ = 1 - p
    | W₁ = W₀ + p [δ(sᵢ,+1) + 0.5 δ(sᵢ,0)]
Else (σᵢ = ±1):
    | W₀ = r
    | W₁ = W₀ + (1-r) δ(σᵢ,+1)

On counting states:

Given the total number of sites `N` and the total magnetization `M`:
    | N = N₀ + N₁ + N₂
    | M = N₁ - N₂

Therefore knowing the total number of site `N`, the magnetization `M` and the number of sites in the `+1` state (`N₁`), we can write N₀ and N₂ as follows:
⟹   | N₀ = N + M - 2N₁
    | N₂ = N₁ - M

"""
module BrassCellularAutomaton

export BrassState, TH0, TH1, TH2,
    BrassCA,
    BrassCAMeanField,
    BrassCAConcrete, BrassCASquareLattice, BrassCAGraph,
    set_state!, state_count,
    magnet_total,
    nearest_neighbors_sum,
    state_concentration,
    magnet_total_local, magnet, magnet_moment,
    advance!, advance_and_measure!,
    advance_parallel!, advance_parallel_and_measure!,
    advance_async!, advance_async_and_measure!

using Statistics, Random, Graphs

include("Metaprogramming.jl")
include("Geometry.jl")

using .Metaprogramming

"""
    BrassState::Int8

Enumeration of the possible states for each site of the Brass Cellular Automaton.
"""
@enum BrassState::Int8 begin
    TH0 = 0
    TH1 = +1
    TH2 = -1
end

"""
    convert(::Type{T}, σ::BrassState) where {T<:Number}

Use the integer representation of `σ::BrassState` in order to convert it to a numerical type `T<:Number`.
"""
@inline Base.convert(::Type{T}, σ::BrassState) where {T<:Number} = T(Integer(σ))

"""
    promote_rule(T::Type, ::Type{BrassState})

Always promote the `BrassState` to whatever the other type is.
"""
@inline Base.promote_rule(T::Type, ::Type{BrassState}) = T

# Arithmetic with numbers and Brass States
for op in (:*, :/, :+, :-)
    @eval begin
        @inline Base.$op(x::Number, σ::BrassState) = $op(promote(x, σ)...)
        @inline Base.$op(σ::BrassState, y::Number) = $op(promote(σ, y)...)
    end
end

"""
    *(σ₁::BrassState, σ₂::BrassState)

Since `({0, +1, -1}, *)` have a monoid structure, it is safe to define multiplication of Brass states.
"""
@inline Base.:*(σ₁::BrassState, σ₂::BrassState) = Integer(σ₁) * Integer(σ₂)

function Base.show(io::IO, ::MIME"text/plain", σ::BrassState)
    brass_str = σ == TH0 ? "TH0" : σ == TH1 ? "TH1" : "TH2"
    print(io, brass_str)
end

"""
    cumulative_transition_probabilities(σᵢ::BrassState, sᵢ::T, p::Float64, r::Float64) where {T<:Integer}

Calculate cumulative transition probabilities for a given site currently at state `σᵢ`
and whose sum of neighbors has sign `sᵢ`.

The probabilities `p` and `r` are parameters of the model.

# Returns:
- `(W₀, W₁)::NTuple{2, Float64}` where:
    - `W₀::Float64`: Probability for new state to be `σ′ = 0`
    - `W₁::Float64`: Probability for new state to be either `σ′ = 0` or `σ′ = +1`

See also [`new_site_state`](@ref).
"""
@inline function cumulative_transition_probabilities(σᵢ::BrassState, sᵢ::T, p::Float64, r::Float64) where {T<:Integer}
    if σᵢ == TH0
        W₀ = 1.0 - p
        W₁ = W₀ + (sᵢ == zero(T) ? (0.5 * p) : sᵢ == one(T) ? p : 0.0)
    else
        W₀ = r
        W₁ = W₀ + (σᵢ == TH1 ? (1.0 - r) : 0.0)
    end
    return (W₀, W₁)
end

"""
    new_site_state(σᵢ::BrassState, sᵢ::Integer, p::Float64, r::Float64)

Determines new state for a given site currently at state `σᵢ` and whose sign of the sum of neighboring site is `sᵢ`.

The probabilities `p` and `r` are parameters of the model.

This function uses the cumulative transition weights `(W₀,W₁)` calculated by [`cumulative_transition_probabilities`](@ref).

A random number `tirage` from an uniform distribution over `[0,1]` is generated and the new state `σᵢ′` is determined as follows:

-  0   < `tirage` < `W₀` ⟹ σᵢ′ = 0
- `W₀` < `tirage` < `W₁` ⟹ σᵢ′ = +1
- `W₁` < `tirage` <  1   ⟹ σᵢ′ = -1

# Returns:
- `σᵢ′::Integer`: New state for the site

See also [`cumulative_transition_probabilities`](@ref).
"""
@inline function new_site_state(σᵢ::BrassState, sᵢ::Integer, p::Float64, r::Float64)
    W₀, W₁ = cumulative_transition_probabilities(σᵢ, sᵢ, p, r)
    tirage = rand()
    σᵢ′ = tirage < W₀ ? TH0 : tirage < W₁ ? TH1 : TH2
    return σᵢ′
end

@doc raw"""
    BrassCAMeanField

Brass system with mean field interaction:
Every spin interacts equally with every other spin.

Since in the mean field model there is no concept of space and locality,
we represent the state of the system simply by total number of spin up and spin down sites.

An `AbstractVector{BrassState}` interface for the `BrassCAMeanField` type can be implemented
if we assume that the all spin states are stored in a sorted vector with ``N = N₊ + N₋`` elements:

    σ = (TH0, TH0, …, TH0, TH1, TH1, …, TH1, TH2, TH2, …, TH2)
        |------ N₀ ------||------ N₁ ------||------ N₂ ------|
        |------------------------- N ------------------------|

Therefore, for an `brass::BrassCAMeanField` we can access the `i`-th spin `σᵢ = brass[i]`:
If `i ≤ N₊` then `σᵢ = ↑` else (`N₊ < i ≤ N`) `σᵢ = ↓`.

# Fields:
- `state::NamedTuple{(:TH0, :TH1, :TH2),NTuple{3,Int64}}`: State of the system given by the number of sites in each state.
"""
mutable struct BrassCAMeanField <: AbstractVector{BrassState}

    "State of the system"
    state::NamedTuple{(:TH0, :TH1, :TH2),NTuple{3,Int64}}

    @doc raw"""
        BrassCAMeanField(; up::Int64, down::Int64)

    Construct an Brass system with mean field interaction with a given number of spins in each state.
    """
    BrassCAMeanField(; TH0::Integer = 0, TH1::Integer = 0, TH2::Integer = 0) = new((TH0 = TH0, TH1 = TH1, TH2 = TH2))

    @doc raw"""
        BrassCAMeanField(N::Integer, σ₀::BrassState)

    Construct an Brass system with mean field interaction with `N` spins, all in a given initial state `σ₀`.
    """
    function BrassCAMeanField(N::Integer, σ₀::BrassState)
        if σ₀ == TH0
            return BrassCAMeanField(TH0 = N)
        elseif σ₀ == TH1
            return BrassCAMeanField(TH1 = N)
        else
            return BrassCAMeanField(TH2 = N)
        end
    end

    @doc raw"""
        BrassCAMeanField(N::Integer, ::Val{:rand})

    Construct an Brass system with mean field interaction with `N` spins in a random initial state.
    """
    function BrassCAMeanField(N::Integer, ::Val{:rand})
        N₀, N′ = minmax(rand(1:N, 2)...)
        N₁ = N′ - N₀
        N₂ = N - N′
        return new((TH0 = N₀, TH1 = N₁, TH2 = N₂))
    end
end

"""
    show(io::IO, ::MIME"text/plain", ca::BrassCAMeanField)

Plain text representation of Bass cellular autamaton with mean field interaction.
"""
function Base.show(io::IO, ::MIME"text/plain", ca::BrassCAMeanField)
    print(io, ca.state)
end

@doc raw"""
    length(ca::BrassCAMeanField)

Total number of spins (`N`) in an Brass system with mean field interaction `ca`.
"""
Base.length(ca::BrassCAMeanField) = sum(ca.state)

Base.size(ca::BrassCAMeanField) = (length(ca),)

@doc raw"""
    IndexStyle(::BrassCAMeanField)

Use only linear indices for the `AbstractVector{BrassState}` interface for the `BrassCAMeanField` type.
"""
@inline Base.IndexStyle(::Type{<:BrassCAMeanField}) = IndexLinear()

@doc raw"""
    getindex(ca::BrassCAMeanField, i::Integer)

Get the state of the `i`-th spin in the Brass system with mean field interaction `ca`.
"""
@inline function Base.getindex(ca::BrassCAMeanField, i::Integer)
    N₀ = ca.state.TH0
    N′ = N₀ + ca.state.TH1
    σ = i <= N₀ ? TH0 : i <= N′ ? TH1 : TH2
    return σ
end

@doc raw"""
    setindex!(ca::BrassCAMeanField, σ::BrassState, i::Integer)

Set the state of the `i`-th spin to `σ` in the Brass system with mean field interaction `ca`.
"""
@inline function Base.setindex!(ca::BrassCAMeanField, σ::BrassState, i::Integer)
    N₀ = ca.state.TH0
    N₁ = ca.state.TH1
    N₂ = ca.state.TH2
    ca.state = if ca[i] == TH0
        if σ == TH1
            (TH0 = N₀ - 1, TH1 = N₁ + 1, TH2 = N₂)
        elseif σ == TH2
            (TH0 = N₀ - 1, TH1 = N₁, TH2 = N₂ + 1)
        end
    elseif ca[i] == TH1
        if σ == TH0
            (TH0 = N₀ + 1, TH1 = N₁ - 1, TH2 = N₂)
        elseif σ == TH2
            (TH0 = N₀, TH1 = N₁ - 1, TH2 = N₂ + 1)
        end
    else
        if σ == TH0
            (TH0 = N₀ + 1, TH1 = N₁, TH2 = N₂ - 1)
        elseif σ == TH1
            (TH0 = N₀, TH1 = N₁ + 1, TH2 = N₂ - 1)
        end
    end
end

@doc raw"""
    firstindex(brass::BrassCAMeanField)

The first spin in the `AbstractVector{BrassState}` interface of `BrassCAMeanField`.
"""
@inline Base.firstindex(ca::BrassCAMeanField) = 1

@doc raw"""
    lastindex(brass::BrassCAMeanField)

The last spin in the `AbstractVector{BrassState}` interface of `BrassCAMeanField`.
"""
@inline Base.lastindex(ca::BrassCAMeanField) = length(ca)

"""
    set_state!(ca::BrassCAMeanField, σ₀::BrassState)

Set the state of all sites of a Brass CA `ca` to a given site state `σ₀`.
"""
@inline function set_state!(ca::BrassCAMeanField, σ₀::BrassState)
    N = length(ca)
    ca.state = if σ₀ == TH0
        (TH0 = N, TH1 = 0, TH2 = 0)
    elseif σ₀ == TH1
        (TH0 = 0, TH1 = N, TH2 = 0)
    else
        (TH0 = 0, TH1 = 0, TH2 = N)
    end
end

"""
    set_state!(ca::BrassCAMeanField, ::Val{:rand})

Set the state of each site of a Brass CA `ca` to a random state `σ₀ ∈ {TH0, TH1, TH2}`.
"""
@inline function set_state!(ca::BrassCAMeanField, ::Val{:rand})
    N = length(ca)
    N₀, N′ = minmax(rand(1:N, 2)...)
    N₁ = N′ - N₀
    N₂ = N - N′
    ca.state = (TH0 = N₀, TH1 = N₁, TH2 = N₂)
end

@doc raw"""
    state_count(ca::BrassCAMeanField)

Number of each spin in the Brass cellular automaton `ca`.
"""
@inline state_count(ca::BrassCAMeanField) = (ca.state.TH0, ca.state.TH1, ca.state.TH2)

@doc raw"""
    magnet_total(brass::BrassCAMeanField)

Total magnetization of the Brass cellular autamaton with mean field interaction `ca`.

    ``M = N₁ - N₂``
"""
@inline magnet_total(ca::BrassCAMeanField) = ca.state.TH1 - ca.state.TH2

@doc raw"""
    nearest_neighbors_sum(ca::BrassCAMeanField, i::Integer)

Sum of the nearest neighbors of the `i`-th site of the Brass cellular autamaton with mean field interaction `ca`.
"""
@inline nearest_neighbors_sum(ca::BrassCAMeanField, i::Integer) = magnet_total(ca) - Integer(ca[i])

"""
    BrassCAConcrete{N} <: AbstractArray{BrassState,N}

Supertype for all Brass cellular automata that have a concrete representation of its state in memory
in the form of a concrete array member `state::Array{BrassState,N}`.

The whole indexing interface of the `state::Array{BrassState,N}` can be passed to the `::IsingConcrete{N}` object itself.
"""
abstract type BrassCAConcrete{N} <: AbstractArray{BrassState,N} end

"""
    length(ca::BrassCAConcrete)

Total number of sites of a Brass cellular automaton `ca`.
"""
@inline Base.length(ca::BrassCAConcrete) = length(ca.state)

"""
    size(ca::BrassCAConcrete)

Size of the state of a Brass cellular automaton `ca`.
"""
@inline Base.size(ca::BrassCAConcrete) = size(ca.state)

"""
    size(ca::BrassCAConcrete, dim::Integer)

Size of the state of Brass cellular automaton `ca` along a given dimension `dim`.
"""
@inline Base.size(ca::BrassCAConcrete, dim::Integer) = size(ca.state, dim)

"""
    IndexStyle(::Type{<:BrassCAConcrete{N}}) where {N}

Use same indexing style used to index the state array.
"""
@inline Base.IndexStyle(::Type{<:BrassCAConcrete{N}}) where {N} = IndexStyle(Array{BrassState,N})

"""
    getindex(ca::BrassCAConcrete{N}, inds::Union{Integer,CartesianIndex{N}}) where {N}

Index the Ising system itself to access its state.
"""
@inline Base.getindex(ca::BrassCAConcrete, inds...) = getindex(ca.state, inds...)

"""
    setindex!(ising::BrassCAConcrete{N}, σ::SpinState, inds::Union{Integer,CartesianIndex{N}}) where {N}

Set the state of a given spin at site `i` to `σ` in the Ising system `ising`.
"""
@inline Base.setindex!(ca::BrassCAConcrete, σ, inds...) = setindex!(ca.state, σ, inds...)

"""
    firstindex(ising::BrassCAConcrete)

Get the index of the first spin in the system.
"""
@inline Base.firstindex(ca::BrassCAConcrete) = firstindex(ca.state)

"""
    lastindex(ising::BrassCAConcrete)

Get the index of the last spin in the system.
"""
@inline Base.lastindex(ca::BrassCAConcrete) = lastindex(ca.state)

"""
    set_state!(ca::BrassCAConcrete, σ₀::BrassState)

Set the state of all sites of a Brass CA `ca` to a given site state `σ₀`.
"""
@inline function set_state!(ca::BrassCAConcrete, σ₀::BrassState)
    fill!(ca, σ₀)
end

"""
    set_state!(ca::BrassCAConcrete, ::Val{:rand})

Set the state of each site of a Brass CA `ca` to a random state `σ₀ ∈ {TH0, TH1, TH2}`.
"""
@inline function set_state!(ca::BrassCAConcrete, ::Val{:rand})
    rand!(ca, instances(BrassState))
end

"""
    state_count(ca::BrassCAConcrete)

Count each type cell on given Brass CA `ca`.

# Returns:
- `(N₀, N₁, N₂)::NTuple{3,Integer}`, where:
    - `N₀`: TH0 cell count
    - `N₁`: TH1 cell count
    - `N₂`: TH2 cell count

See also: [`state_concentration`](@ref).
"""
@inline function state_count(state::Array{BrassState})
    # Total number of sites
    N = length(state)
    # Total magnetization
    M = magnet_total(state)
    # Calculate N₁
    N₁ = count(==(TH1), state)
    # Calculate remaining values
    N₀ = N + M - 2 * N₁
    N₂ = N₁ - M
    # Return tuple
    return (N₀, N₁, N₂)
end
@inline state_count(ca::BrassCAConcrete) = state_count(ca.state)

@doc raw"""
    magnet_total(ca::BrassCAConcrete)

Total magnetization of a Brass CA `ca`.

The total magnetization is defined as the sum of all site states:

``M = ∑ᵢ σᵢ``

See also: [`magnet`](@ref), [`magnet_moment`](@ref).
"""
@inline magnet_total(state::Array{BrassState}) = @inbounds sum(Integer, state)
@inline magnet_total(ca::BrassCAConcrete) = magnet_total(ca.state)

@inline magnet(state::Array{BrassState}) = magnet_total(state) / length(state)

"""
     step!(ca::BrassCAConcrete{N}, state::Array{BrassState,N}, state′::Array{BrassState,N}, p::Float64, r::Float64) where {N}

Single step of the Brass CA concrete.

# Arguments:
- `ca`: Brass CA
- `σ`: Current state of the CA
- `σ′`: Array to store resulting state of the CA
- `p` and `r`: Probabilities of the model
"""
@inline function step!(ca::BrassCAConcrete{N}, state::Array{BrassState,N}, state′::Array{BrassState,N}, p::Float64, r::Float64) where {N}
    # Iterate over every site
    for i in eachindex(ca)
        σᵢ = state[i]
        # Get sign of the sum of nearest neighbors
        sᵢ = sign(nearest_neighbors_sum(ca, state, i))
        # Transition to new site state
        state′[i] = new_site_state(σᵢ, sᵢ, p, r)
    end
end

"""
    step_parallel!(ca::BrassCAConcrete{N}, state::Array{BrassState,N}, state′::Array{BrassState,N}, p::Float64, r::Float64) where {N}

Single step of the Brass CA.

The sites are updated in parallel.

# Arguments:
- `ca`: Brass CA
- `σ`: Current state of the CA
- `σ′`: Array to store resulting state of the CA
- `p` and `r`: Probabilities of the model
"""
@inline function step_parallel!(ca::BrassCAConcrete{N}, state::Array{BrassState,N}, state′::Array{BrassState,N}, p::Float64, r::Float64) where {N}
    # Iterate over every site
    @inbounds Threads.@threads for i in CartesianIndices(ca)
        σᵢ = state[i]
        # Get sign of the sum of nearest neighbors
        sᵢ = sign(nearest_neighbors_sum(ca, state, i))
        # Transition to new site state
        state′[i] = new_site_state(σᵢ, sᵢ, p, r)
    end
end

"""
    step_async!(ca::BrassCAConcrete{N}, i::CartesianIndex{N}, p::Float64, r::Float64) where {N}

Single *asynchronous* step of the Brass CA, updating a given site.

# Arguments:
- `ca`: Brass CA
- `i`: site to be updated
- `p` and `r`: Probabilities of the model
"""
@inline function step_async!(ca::BrassCAConcrete{N}, i::CartesianIndex{N}, p::Float64, r::Float64) where {N}
    σᵢ = ca[i]
    # Get sign of the sum of nearest neighbors
    sᵢ = sign(nearest_neighbors_sum(ca, ca.state, i))
    # Transition to new site state
    ca[i] = new_site_state(σᵢ, sᵢ, p, r)
end

"""
    step_async!(ca::BrassCAConcrete, p::Float64, r::Float64)

Single *asynchronous* step of the Brass CA, updating a random site.

# Arguments:
- `ca`: Brass CA
- `p` and `r`: Probabilities of the model
"""
@inline function step_async!(ca::BrassCAConcrete, p::Float64, r::Float64)
    i = rand(eachindex(ca))
    step_async!(ca, i, p, r)
end

"""
    advance!(ca::BrassCAConcrete, p::Float64, r::Float64, n_steps::Integer = 1)

Advance the state of a Brass CA `ca` *synchronously* by `n_steps` time steps.

The probabilities `p` and `r` are parameters of the model.

Each specific type of Brass CA `BrassCAConcreteSpecific` must provide its own implementation of the `step!(ca::BrassCAConcreteSpecific, state::Array{StateType}, state′::Array{StateType}, p::Float64, r::Float64)` method.

See also [`step!`](@ref), [`advance_and_measure!`](@ref), [`advance_parallel!`](@ref), [`advance_async!`](@ref).
"""
function advance!(ca::BrassCAConcrete, p::Float64, r::Float64, n_steps::Integer = 1)
    @assert n_steps > 0 "Number of steps must be positive."
    # Auxiliar state
    state′ = similar(ca.state)
    # Time steps iteration
    @inbounds for _ in 1:floor(Int, n_steps / 2)
        step!(ca, ca.state, state′, p, r)
        step!(ca, state′, ca.state, p, r)
    end
    if isodd(n_steps)
        step!(ca, ca.state, state′, p, r)
        ca.state = state′
    end
end

"""
    advance_parallel!(ca::BrassCAConcrete, p::Float64, r::Float64, n_steps::Integer = 1)

Advance the state of a Brass CA `ca` *synchronously* by `n_steps` time steps.

For each time step the sites of the CA are updated in parallel.

The probabilities `p` and `r` are parameters of the model.

Each specific type of Brass CA `BrassCAConcreteSpecific` must provide its own implementation of the `step_parallel!(ca::BrassCAConcreteSpecific, state::Array{StateType}, state′::Array{StateType}, p::Float64, r::Float64)` method.

See also [`step_parallel!`](@ref), [`advance_parallel_and_measure!`](@ref), [`advance!`](@ref), [`advance_async!`](@ref).
"""
function advance_parallel!(ca::BrassCAConcrete, p::Float64, r::Float64, n_steps::Integer = 1)
    @assert n_steps > 0 "Number of steps must be positive."
    # Auxiliar state
    state′ = similar(ca.state)
    # Time steps iteration
    @inbounds for _ in 1:floor(Int, n_steps / 2)
        step_parallel!(ca, ca.state, state′, p, r)
        step_parallel!(ca, state′, ca.state, p, r)
    end
    if isodd(n_steps)
        step_parallel!(ca, ca.state, state′, p, r)
        ca.state = state′
    end
end

"""
    advance_and_measure!(measurement::Function, ca::BrassCAConcrete, p::Float64, r::Float64, n_steps::Integer = 1)

Advance the state of a Brass CA `ca` *synchronously* by `n_steps` time steps and performs a measurement given by the function `measurement` after wach time step.

The function `measurement` must take as its sole argument the state of CA.

    measurement::(Array -> ResultType)

Each specific type of Brass CA `BrassCAConcreteSpecific` must provide its own implementation of the `step!(ca::BrassCAConcreteSpecific, state::Array{StateType}, state′::Array{StateType}, p::Float64, r::Float64)` method.

# Returns:
- `results::Vector{ResultType}`: Vector containing the mearurements results

See also [`step!`](@ref), [`advance!`](@ref), [`advance_parallel_and_measure!`](@ref), [`advance_async_and_measure!`](@ref).
"""
function advance_and_measure!(measurement::Function, ca::BrassCAConcrete{N}, p::Float64, r::Float64, n_steps::Integer = 1) where {N}
    @assert n_steps > 0 "Number of steps must be positive."
    # Measurement results
    ResultType = Base.return_types(measurement, (Array{BrassState,N},))[1]
    results = Array{ResultType}(undef, n_steps + 1)
    # First measurement
    results[1] = measurement(ca.state)
    # Auxiliar state
    state′ = similar(ca.state)
    # Time steps iteration
    @inbounds for t in 1:floor(Int, n_steps / 2)
        step!(ca, ca.state, state′, p, r)
        results[2*t] = measurement(state′)
        step!(ca, state′, ca.state, p, r)
        results[2*t+1] = measurement(ca.state)
    end
    if isodd(n_steps)
        step!(ca, ca.state, state′, p, r)
        ca.state = state′
        results[end] = measurement(ca.state)
    end
    return results
end

"""
    advance_parallel_and_measure!(measurement::Function, ca::BrassCAConcrete, p::Float64, r::Float64, n_steps::Integer = 1)

Advance the state of a Brass CA `ca` *synchronously* by `n_steps` time steps and performs a measurement given by the function `measurement` after wach time step.

For each time step the sites of the CA are updated in parallel.

The function `measurement` must take as its sole argument the state of CA.

    measurement::(Array -> ResultType)

Each specific type of Brass CA `BrassCAConcreteSpecific` must provide its own implementation of the `step_parallel!(ca::BrassCAConcreteSpecific, state::Array{StateType}, state′::Array{StateType}, p::Float64, r::Float64)` method.

# Returns:
- `results::Vector{ResultType}`: Vector containing the mearurements results

See also [`step_parallel!`](@ref), [`advance_parallel!`](@ref), [`advance_and_measure!`](@ref), [`advance_async_and_measure!`](@ref).
"""
function advance_parallel_and_measure!(measurement::Function, ca::BrassCAConcrete{N}, p::Float64, r::Float64, n_steps::Integer = 1) where {N}
    @assert n_steps > 0 "Number of steps must be positive."
    # Measurement results
    ResultType = Base.return_types(measurement, (Array{BrassState,N},))[1]
    results = Array{ResultType}(undef, n_steps + 1)
    # First measurement
    results[1] = measurement(ca.state)
    # Auxiliar state
    state′ = similar(ca.state)
    # Time steps iteration
    @inbounds for t in 1:floor(Int, n_steps / 2)
        step_parallel!(ca, ca.state, state′, p, r)
        results[2*t] = measurement(state′)
        step_parallel!(ca, state′, ca.state, p, r)
        results[2*t+1] = measurement(ca.state)
    end
    if isodd(n_steps)
        step_parallel!(ca, ca.state, state′, p, r)
        ca.state = state′
        results[end] = measurement(ca.state)
    end
    return results
end

"""
    advance_async!(ca::BrassCAConcrete, p::Float64, r::Float64, n_steps::Integer = 1)

Advance the state of a Brass CA `ca` *asynchronously* by `n_steps` time steps.

The probabilities `p` and `r` are parameters of the model.

Each specific type of Brass CA `BrassCAConcreteSpecific` must provide its own implementation of the `step_async!(ca::BrassCAConcreteSpecific, p::Float64, r::Float64)` method.

See also [`step_async!`](@ref), [`advance!`](@ref), [`advance_parallel!`](@ref), [`advance_async_and_measure!`](@ref).
"""
function advance_async!(ca::BrassCAConcrete, p::Float64, r::Float64, n_steps::Integer = 1)
    @assert n_steps > 0 "Number of steps must be positive."
    # Time steps iteration
    @inbounds for _ in 1:n_steps
        step_async!(ca, p, r)
    end
end

"""
    advance_async_and_measure!(measurement::Function, ca::BrassCAConcrete, p::Float64, r::Float64, n_steps::Integer = 1)

Advance the state of a Brass CA `ca` *asynchronously* by `n_steps` time steps and performs a measurement given by the function `measurement` after wach time step.

The function `measurement` must take as its sole argument the state of CA.

    measurement::(Array -> ResultType)

Each specific type of Brass CA `BrassCAConcreteSpecific` must provide its own implementation of the `step_async!(ca::BrassCAConcreteSpecific, p::Float64, r::Float64)` method.

# Returns:
- `results::Vector{ResultType}`: Vector containing the mearurements results

See also [`step_async!`](@ref), [`advance_async!`](@ref), [`advance_and_measure!`](@ref), [`advance_parallel_and_measure!`](@ref).
"""
function advance_async_and_measure!(measurement::Function, ca::BrassCAConcrete{N}, p::Float64, r::Float64, n_steps::Integer = 1) where {N}
    @assert n_steps > 0 "Number of steps must be positive."
    # Measurement results
    ResultType = Base.return_types(measurement, (Array{BrassState,N},))[1]
    results = Array{ResultType}(undef, n_steps + 1)
    # First measurement
    results[1] = measurement(ca.state)
    # Time steps iteration
    @inbounds for t in 2:n_steps+1
        step_async!(ca, p, r)
        results[t] = measurement(ca.state)
    end
    return results
end

"""
    BrassCASquareLattice{N} <: BrassCAConcrete{N}

Brass cellular automaton on a periodic `N`-dimensional square lattice.

# Fields:
- `σ::Array{BrassState}`: State of the CA
"""
mutable struct BrassCASquareLattice{N} <: BrassCAConcrete{N}

    "State of the CA"
    state::Array{BrassState,N}

    @doc raw"""
        BrassCASquareLattice(state::Array{BrassState,N}) where {N}

    Costruct a Brass CA with a given initial state `state`.
    """
    BrassCASquareLattice(state::Array{BrassState,N}) where {N} = new{N}(state)

    @doc raw"""
        BrassCASquareLattice(size::NTuple{N,Integer}, σ₀::BrassState) where {N}

    Construct a Brass CA with dimensions `size` and random initial state `σ₀`.
    """
    BrassCASquareLattice(size::NTuple{N,Integer}, σ₀::BrassState) where {N} = new{N}(fill(σ₀, size))

    @doc raw"""
        BrassCASquareLattice(size::NTuple{N,Integer}, ::Val{:rand}) where {N}

    Construct a Brass CA with dimensions `size` and random initial state.
    """
    BrassCASquareLattice(size::NTuple{N,Integer}, ::Val{:rand}) where {N} = new{N}(rand(instances(BrassState), size))

    @doc raw"""
        BrassCASquareLattice(::Val{N}, L::Integer, σ₀::BrassState) where {N}

    Construct a `dim`-dimensional square Brass CA of side length `L` and a given initial state `σ₀`.
    """
    BrassCASquareLattice(::Val{N}, L::Integer, σ₀::BrassState) where {N} = BrassCASquareLattice(ntuple(_ -> L, Val(N)), σ₀)

    @doc raw"""
        BrassCASquareLattice(::Val{N}, L::Integer, ::Val{:rand}) where {N}

    Construct a `dim`-dimensional square Brass CA of side length `L` and random initial state.
    """
    BrassCASquareLattice(::Val{N}, L::Integer, ::Val{:rand}) where {N} = BrassCASquareLattice(ntuple(_ -> L, Val(N)), Val(:rand))

end

"""
    nearest_neighbors_sum(ca::BrassCASquareLattice{N}, state::Array{BrassState,N}, idx::CartesianIndex{N}) where {N}

Sum of the states of the nearest neighbors of a given site at `idx` of a Brass cellular automaton `ca` with state `state` on a periodic `N`-dimensional square lattice.

# Arguments:
- `state`: State of the Brass CA
- `idx`: Cartesian index of the chosen site
"""
@inline nearest_neighbors_sum(ca::BrassCASquareLattice{N}, state::Array{BrassState,N}, idx::CartesianIndex{N}) where {N} = @inbounds Geometry.square_lattice_nearest_neighbors_sum(state, idx)

"""
Brass CA on an abitrary graph `g` with states of each node stored in the vector `state`
"""
mutable struct BrassCAGraph <: BrassCAConcrete{1}

    "Graph structure"
    g::Graph

    "State at each node"
    state::Vector{BrassState}

    """
        BrassCAGraph(g::Graph, σ::BrassState)

    Construct a new Brass CA with graph structure `g` and a given initial state `σ` for all sites.
    """
    BrassCAGraph(g::Graph, σ::BrassState) = new(g, fill(σ, nv(g)))

    """
        BrassCAGraph(g::Graph, ::Val{:rand})

    Construct a new Brass CA with graph structure `g` and random initial states at each node.
    """
    BrassCAGraph(g::Graph, ::Val{:rand}) = new(g, rand(instances(BrassState), nv(g)))
end
"""
    nearest_neighbors_sum(state::Vector{BrassState,N}, idx::CartesianIndex{N}) where {N}

Sum of the states of the nearest neighbors of a given site at `idx` of a Brass cellular automaton with state `state` on a periodic `N`-dimensional square lattice.

# Arguments:
- `state`: State of the Brass CA
- `idx`: Cartesian index of the chosen site
"""
@inline nearest_neighbors_sum(ca::BrassCAGraph, state::Vector{BrassState}, i::Integer) = @inbounds sum(Integer, state[nn] for nn ∈ neighbors(ca.g, i))

"""
    BrassCA

Supertype for all Brass cellular automata.
"""
BrassCA = Union{BrassCAConcrete,BrassCAMeanField}

@doc raw"""
    state_concentration(ca::BassCA)

Concentration of each type cell on given Brass CA state `state`.

``cᵢ = Nᵢ/N``

# Returns:
- `(c₀,c₁,c₂)::NTuple{3,Float64}`, where:
    - `c₀`: TH0 cell concentration
    - `c₁`: TH1 cell concentration
    - `c₂`: TH2 cell concentration

See also: [`state_count`](@ref).
"""
@inline state_concentration(ca::BrassCA) = state_count(ca) ./ length(ca)

@doc raw"""
    magnet_total_local(brass::BrassCA, i::Integer, σᵢ′::BrassState)

Change in local magnetization of an Brass f the `i`-th were to be changed to `σᵢ′`.
"""
@inline magnet_total_local(ca::BrassCA, i::Integer, σᵢ′::BrassState) = Integer(σᵢ′) - Integer(ca[i])

@doc raw"""
    magnet(ca::BrassCA)

Magnetization per site of a Brass CA `ca`.

``m = M / N = ∑ᵢ σᵢ / N``

See also: [`magnet_total`](@ref), [`magnet_moment`](@ref).
"""
@inline magnet(ca::BrassCA) = magnet_total(ca) / length(ca)

@doc raw"""
    magnet_moment(ca::BrassCA, k::integer)

Calculates the k-th momentum of the magnetization of a brass CA `ca`.

``mᵏ = 1/nᵏ (∑ᵢ σᵢ)ᵏ``

See also: [`magnet`](@ref), [`magnet_total`](@ref).
"""
@inline magnet_moment(ca::BrassCA, k::Integer) = magnet(ca)^k

end
