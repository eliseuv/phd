@doc raw"""
    Cellular Automata


"""
module CellularAutomata

export
    # Abstract site state
    AbstractSiteState, instance_count,
    # Abstract finite state
    AbstractFiniteState,
    state_count, state_concentration,
    set_state!, randomize_state!,
    nearest_neighbors, nearest_neighbors_sum,
    # Mean field finite state
    MeanFieldFiniteState,
    # Concrete finite state
    ConcreteFiniteState,
    container, similar_container,
    # Square lattice finite state
    SquareLatticeFiniteState,
    # Simple graph finite state
    SimpleGraphFiniteState,
    # Abstract cellular automaton
    AbstractCellularAutomaton,
    state,
    step!,
    advance!, advance_measure!,
    async_advance!,
    # Brass states
    BrassState,
    # Brass cellular automaton
    BrassCellularAutomaton,
    # Measurements on Brass cellular automaton
    magnet_total, magnet


using EnumX, Random

include("FiniteStates.jl")

using .FiniteStates

"""
    AbstractCellularAutomaton{T<:AbstractFiniteState}

Supertype for all cellular automata
"""
abstract type AbstractCellularAutomaton{T<:AbstractFiniteState} end

@doc raw"""
    state(ca::AbstractCellularAutomaton)

Get the state of the cellular automaton `ca`.
"""
@inline state(ca::AbstractCellularAutomaton) = ca.state

"""
    length(ca::AbstractCellularAutomaton)

Total number of sites of an spin system `ca`.
"""
@inline Base.length(ca::AbstractCellularAutomaton) = length(state(ca))

"""
    size(ca::AbstractCellularAutomaton)

Size of the spins of an spin system `ca`.
"""
@inline Base.size(ca::AbstractCellularAutomaton) = size(state(ca))

"""
    IndexStyle(::Type{<:AbstractCellularAutomaton})

Use the same index style from the spin state.
"""
@inline Base.IndexStyle(::Type{<:AbstractCellularAutomaton{T}}) where {T} = IndexStyle(T)

"""
    getindex(ca::AbstractCellularAutomaton, inds...)

Index the spin system itself to access its spins.
"""
@inline Base.getindex(ca::AbstractCellularAutomaton, inds...) = getindex(state(ca), inds...)

"""
    setindex!(ca::AbstractCellularAutomaton, σ, inds...)

Set the spins of a given spin at site `i` to `σ` in the spin system `ca`.
"""
@inline Base.setindex!(ca::AbstractCellularAutomaton, σ, inds...) = setindex!(state(ca), σ, inds...)

"""
    firstindex(ca::AbstractCellularAutomaton)

Get the index of the first spin in the system.
"""
@inline Base.firstindex(ca::AbstractCellularAutomaton) = firstindex(state(ca))

"""
    lastindex(ca::AbstractCellularAutomaton)

Get the index of the last spin in the system.
"""
@inline Base.lastindex(ca::AbstractCellularAutomaton) = lastindex(state(ca))

@doc raw"""
    step!(ca::AbstractCellularAutomaton{<:ConcreteFiniteState{T,N}}, container′::Array{T,N}) where {T,N}

Advances a single step the cellular automaton `ca` with concrete finite state and store the new state in `container′`.

The sites are updated in parallel.
"""
function step!(ca::AbstractCellularAutomaton{<:ConcreteFiniteState{T,N}}, container′::Array{T,N}) where {T,N}
    @inbounds Threads.@threads for i in eachindex(ca.state)
        container′[i] = new_site_state(ca, i)
    end
end

@doc raw"""
    advance!(ca::AbstractCellularAutomaton, n_steps::Integer=1)

Advances the cellular automaton `ca` *synchronously* for `n_steps` steps.
"""
function advance!(ca::AbstractCellularAutomaton, n_steps::I=1) where {I<:Integer}
    @assert n_steps > 0 "Number of steps must be positive."
    # Auxiliary container
    container′ = similar_container(ca.state)
    # Time steps iteration
    @inbounds for _ in 1:n_steps
        # Calculate next CA state and store in aux container
        step!(ca, container′)
        # Swap states
        ca.state.container, container′ = container′, ca.state.container
    end
end

@doc raw"""
    advance_measure!(measurement::Function, ca::T, n_steps::Integer=1) where {T<:AbstractCellularAutomaton}

Advances the cellular automaton `ca` *synchronously* for `n_steps` steps and performs the measurement `measurement` on the cellular automaton after each step.
"""
function advance_measure!(measurement::Function, ca::T, n_steps::Integer=1) where {T<:AbstractCellularAutomaton}
    @assert n_steps > 0 "Number of steps must be positive."
    # Results vector
    ResultType = Base.return_types(measurement, (T,))[1]
    results = Vector{ResultType}(undef, n_steps + 1)
    # Initial measurement
    results[1] = measurement(ca)
    # Auxiliary container
    container′ = similar_container(ca.state)
    # Time steps iteration
    @inbounds for t in 2:(n_steps+1)
        # Calculate next CA state and store in aux container
        step!(ca, container′)
        # Swap states
        ca.state.container, container′ = container′, ca.state.container
        # Perform measurements
        results[t] = measurement(ca)
    end
    # Return measurement results
    return results
end

@doc raw""""
    async_advance!(ca::AbstractCellularAutomaton, n_steps::Integer=1)

Advance the state of the celluar automaton `ca` *asynchronously* by `n_steps` time steps.
"""
function async_advance!(ca::AbstractCellularAutomaton, n_steps::Integer=1)
    @assert n_steps > 0 "Number of steps must be positive."
    for _ in 1:n_steps
        @inbounds for i in rand(eachindex(state(ca)), length(ca))
            ca[i] = new_site_state(ca, i)
        end
    end
end

@doc raw"""
    BrassState::Int8

States of the Brass cellular automaton.
"""
@enumx BrassState::Int8 begin
    TH0 = 0
    TH1 = +1
    TH2 = -1
end

"""
    state_count(fs::ConcreteFiniteState{BrassState.T})

Count each type cell on given concrete finite state `fs`.

# Returns:
- `(N₀, N₁, N₂)::NTuple{3,Integer}`, where:
    - `N₀`: TH0 cell count
    - `N₁`: TH1 cell count
    - `N₂`: TH2 cell count

See also: [`state_concentration`](@ref).
"""
@inline function state_count(fs::ConcreteFiniteState{BrassState.T})
    # Total number of sites
    N = length(fs)
    # Total magnetization
    M = sum(fs)
    # Calculate N₁
    N₁ = count(==(TH1), fs)
    # Calculate remaining values
    N₀ = N + M - 2 * N₁
    N₂ = N₁ - M
    # Return tuple
    return (N₀, N₁, N₂)
end

function Base.show(io::IO, ::MIME"text/plain", σ::BrassState.T)
    brass_str = σ == BrassState.TH0 ? "TH0" : σ == BrassState.TH1 ? "TH1" : "TH2"
    print(io, brass_str)
end

@doc raw"""
    magnet_total(fs::ConcreteFiniteState{BrassState.T})

Total magnetization of a concrete finite state `fs` of Brass site states.

The total magnetization is defined as the sum of all site states:

``M = ∑ᵢ σᵢ``
"""
@inline magnet_total(fs::AbstractFiniteState{BrassState.T}) = sum(fs)

@doc raw"""
    magnet_total(fs::MeanFieldFiniteState{BrassState.T})

Total magnetization of a mean field finite state `fs` of Brass site states.
"""
@inline magnet_total(fs::MeanFieldFiniteState{BrassState.T}) = fs[BrassState.TH1] - fs[BrassState.TH2]

@doc raw"""
    magnet(fs::AbstractFiniteState)

Magnetization per site of a finite state `fs` of Brass site states.

``m = M / N = ∑ᵢ σᵢ / N``
"""
@inline magnet(fs::AbstractFiniteState{BrassState.T}) = magnet_total(fs) / length(fs)

@doc raw"""
    BrassCellularAutomaton{T<:AbstractFiniteState{BrassState.T}} <: AbstractCellularAutomaton{T}

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
struct BrassCellularAutomaton{T<:AbstractFiniteState{BrassState.T}} <: AbstractCellularAutomaton{T}

    "State of Brass CA"
    state::T

    "Parameters of the model"
    p::Real
    r::Real

    """
        BrassCellularAutomaton(state::T, p::Real, r::Real) where {T}

    Construct a Brass cellular automaton with given initial state `state` and parameters `p` and `r`.
    """
    BrassCellularAutomaton(state::T, p::Real, r::Real) where {T} = new{T}(state, p, r)

end

@doc raw"""
    cumulative_transition_probabilities(σᵢ::BrassState.T, sᵢ::T, p::Float64, r::Float64) where {T<:Integer}

Calculate cumulative transition probabilities for a given site currently at state `σᵢ`
and whose sum of neighbors has sign `sᵢ`.

The probabilities `p` and `r` are parameters of the model.

# Returns:
- `(W₀, W₁)::NTuple{2, Float64}` where:
    - `W₀::Float64`: Probability for new state to be `σ′ = 0`
    - `W₁::Float64`: Probability for new state to be either `σ′ = 0` or `σ′ = +1`

See also [`new_site_state`](@ref).
"""
@inline function cumulative_transition_probabilities(σᵢ::BrassState.T, sᵢ::T, p::Real, r::Real) where {T<:Integer}
    if σᵢ == BrassState.TH0
        W₀ = 1.0 - p
        W₁ = W₀ + (sᵢ == zero(T) ? (0.5 * p) : sᵢ == one(T) ? p : 0.0)
    else
        W₀ = r
        W₁ = W₀ + (σᵢ == BrassState.TH1 ? (1.0 - r) : 0.0)
    end
    return (W₀, W₁)
end

@doc raw"""
    new_site_state(ca::BrassCellularAutomaton{<:ConcreteFiniteState{T,N}}, i) where {T,N}

Determines new state of the `i`-th site for the Brass cellular automaton `ca`.

This function uses the cumulative transition weights `(W₀,W₁)` calculated by [`cumulative_transition_probabilities`](@ref).

A random number `tirage` from an uniform distribution over `[0,1]` is generated and the new state `σᵢ′` is determined as follows:

-  0   < `tirage` < `W₀` ⟹ σᵢ′ = 0
- `W₀` < `tirage` < `W₁` ⟹ σᵢ′ = +1
- `W₁` < `tirage` <  1   ⟹ σᵢ′ = -1

# Returns:
- `σᵢ′::Integer`: New state for the site

See also [`cumulative_transition_probabilities`](@ref).
"""
@inline function new_site_state(ca::BrassCellularAutomaton{<:ConcreteFiniteState{T,N}}, i) where {T,N}
    # Get cumulative transition weights
    W₀, W₁ = let σᵢ = ca[i], sᵢ = sign(nearest_neighbors_sum(ca.state, i))
        cumulative_transition_probabilities(σᵢ, sᵢ, ca.p, ca.r)
    end
    tirage = rand()
    σᵢ′ = tirage < W₀ ? BrassState.TH0 : tirage < W₁ ? BrassState.TH1 : BrassState.TH2
    return σᵢ′
end

@doc raw"""
    magnet_total(ca::BrassCellularAutomaton)

Total magnetization of a Brass cellular automaton `ca`.
"""
@inline magnet_total(ca::BrassCellularAutomaton) = magnet_total(state(ca))

@doc raw"""
    magnet(ca::BrassCellularAutomaton)

Magnetization per site of a Brass cellular automaton `ca`.

``m = M / N = ∑ᵢ σᵢ / N``
"""
@inline magnet(ca::BrassCellularAutomaton) = magnet(state(ca))

end
