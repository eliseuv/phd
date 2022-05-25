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

export BrassCA, BrassCAMeanField, BrassCASquareLattice, BrassCAGraph,
    set_state!,
    advance!, advance_and_measure!,
    advance_parallel!, advance_parallel_and_measure!,
    advance_async!, advance_async_and_measure!,
    magnet_total, magnet, state_count, state_concentration

using Statistics, Random, Graphs

"""
    BrassCA

Supertype for all Brass Cellular Automata.
"""
abstract type BrassCA end

"""
    size(ca::BrassCA)

Size of the state of a Brass cellular automaton `ca`.
"""
@inline Base.size(ca::BrassCA) = size(ca.σ)

"""
    size(ca::BrassCA, d::Integer)

Size of the state of Brass cellular automaton `ca` along a given dimension `d`.
"""
@inline Base.size(ca::BrassCA, d::Integer) = size(ca.σ, d)

"""
    length(ca::BrassCA)

Total number of sites of a Brass cellular automaton `ca`.
"""
@inline Base.length(ca::BrassCA) = length(ca.σ)

@doc raw"""
    magnet_total(σ::Array)

Total magnetization of a Brass CA with state `σ`.

The total magnetization is defined as the sum of all site states:

``M = ∑ᵢ σᵢ``

See also: [`magnet`](@ref), [`magnet_moment`](@ref).
"""
@inline magnet_total(σ::Array) = @inbounds sum(σ)

@doc raw"""
    magnet_total(ca::BrassCA)

Total magnetization of a Brass CA `ca`.

The total magnetization is defined as the sum of all site states:

``M = ∑ᵢ σᵢ``

See also: [`magnet`](@ref), [`magnet_moment`](@ref).
"""
@inline magnet_total(ca::BrassCA) = magnet_total(ca.σ)

@doc raw"""
    magnet(σ::Array)

Magnetization per site of a Brass CA with state `σ`.

``m = M / N = ∑ᵢ σᵢ / N``

See also: [`magnet_total`](@ref), [`magnet_moment`](@ref).
"""
@inline magnet(σ::Array) = magnet_total(σ) / length(σ)

@doc raw"""
    magnet(ca::BrassCA)

Magnetization per site of a Brass CA `ca`.

``m = M / N = ∑ᵢ σᵢ / N``

See also: [`magnet_total`](@ref), [`magnet_moment`](@ref).
"""
@inline magnet(ca::BrassCA) = magnet(ca.σ)

@doc raw"""
    magnet_moment(σ::array, k::integer)

Calculates the k-th momentum of the magnetization of a brass CA with state `σ`.

``mᵏ = 1/nᵏ (∑ᵢ σᵢ)ᵏ``

See also: [`magnet`](@ref), [`magnet_total`](@ref).
"""
@inline magnet_moment(σ::Array, k::Integer) = magnet(σ)^k

@doc raw"""
    magnet_moment(ca::BrassCA, k::integer)

Calculates the k-th momentum of the magnetization of a brass CA `ca`.

``mᵏ = 1/nᵏ (∑ᵢ σᵢ)ᵏ``

See also: [`magnet`](@ref), [`magnet_total`](@ref).
"""
@inline magnet_moment(ca::BrassCA, k::Integer) = magnet_moment(ca.σ, Val(k))

"""
    state_count(σ::Array)

Count each type cell on given Brass CA state `σ`.

# Returns:
- `(N₀, N₁, N₂)::NTuple{3,Integer}`, where:
    - `N₀`: TH0 cell count
    - `N₁`: TH1 cell count
    - `N₂`: TH2 cell count

See also: [`state_concentration`](@ref).
"""
@inline function state_count(σ::Array)
    # Calculate N₁
    N₁ = count(==(+1), σ)
    # Aux variables
    N = length(σ)
    M = sum(σ)
    # Calculate remaining values
    N₀ = N + M - 2 * N₁
    N₂ = N₁ - M
    # Return tuple
    return (N₀, N₁, N₂)
end

"""
    state_count(ca::BrassCA)

Count each type cell on given Brass CA `ca`.

# Returns:
- `(N₀, N₁, N₂)::NTuple{3,Integer}`, where:
    - `N₀`: TH0 cell count
    - `N₁`: TH1 cell count
    - `N₂`: TH2 cell count

See also: [`state_concentration`](@ref).
"""
@inline state_count(ca::BrassCA) = state_count(ca.σ)

@doc raw"""
    state_concentration(σ::Array)

Concentration of each type cell on given Brass CA state `σ`.

``cᵢ = Nᵢ/N``

# Returns:
- `(c₀,c₁,c₂)::NTuple{3,Float64}`, where:
    - `c₀`: TH0 cell concentration
    - `c₁`: TH1 cell concentration
    - `c₂`: TH2 cell concentration

See also: [`state_count`](@ref).
"""
@inline state_concentration(σ::Array) = state_count(σ) ./ length(σ)

@doc raw"""
    state_concentration(ca::BrassCA)

Concentration of each type cell on given Brass CA `ca`.

``cᵢ = Nᵢ/N``

# Returns:
- `(c₀,c₁,c₂)::NTuple{3,Float64}`, where:
    - `c₀`: TH0 cell concentration
    - `c₁`: TH1 cell concentration
    - `c₂`: TH2 cell concentration

See also: [`state_count`](@ref).
"""
@inline state_concentration(ca::BrassCA) = state_concentration(ca.σ)

"""
    set_state!(ca::BrassCA, σ₀::Int8)

Set the state of all sites of a Brass CA `ca` to a given site state `σ₀`.
"""
@inline set_state!(ca::BrassCA, σ₀::Int8) = fill!(ca.σ, σ₀)

"""
    set_state!(ca::BrassCA, ::Val{:rand})

Set the state of each site of a Brass CA `ca` to a random state `σ ∈ {-1, 0, +1}`.
"""
@inline set_state!(ca::BrassCA, ::Val{:rand}) = rand!(ca.σ, Int8.(-1:1))

"""
    cumulative_transition_probabilities(σᵢ::Integer, sᵢ::Integer, p::Float64, r::Float64)

Calculate cumulative transition probabilities for a given site currently at state `σᵢ`
and whose sum of neighbors has sign `sᵢ`.

The probabilities `p` and `r` are parameters of the model.

# Returns:
- `(W₀, W₁)::NTuple{2, Float64}` where:
    - `W₀::Float64`: Probability for new state to be `σ′ = 0`
    - `W₁::Float64`: Probability for new state to be either `σ′ = 0` or `σ′ = +1`

See also [`new_site_state`](@ref).
"""
@inline function cumulative_transition_probabilities(σᵢ::Integer, sᵢ::Integer, p::Float64, r::Float64)
    if σᵢ == 0
        W₀ = 1 - p
        W₁ = W₀ + (sᵢ == 0 ? (0.5 * p) : sᵢ == +1 ? p : 0.0)
    else
        W₀ = r
        W₁ = W₀ + (σᵢ == +1 ? (1.0 - r) : 0.0)
    end
    return (W₀, W₁)
end

"""
    new_site_state(σᵢ::Integer, sᵢ::Integer, p::Float64, r::Float64)

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
@inline function brass_ca_new_site_state(σᵢ::Integer, sᵢ::Integer, p::Float64, r::Float64)
    W₀, W₁ = cumulative_transition_probabilities(σᵢ, sᵢ, p, r)
    tirage = rand()
    σᵢ′ = tirage < W₀ ? 0 : tirage < W₁ ? +1 : -1
    return σᵢ′
end

"""
    advance!(ca::BrassCA, p::Float64, r::Float64, n_steps::Integer = 1)

Advance the state of a Brass CA `ca` *synchronously* by `n_steps` time steps.

The probabilities `p` and `r` are parameters of the model.

Each specific type of Brass CA `BrassCASpecific` must provide its own implementation of the `step!(ca::BrassCASpecific, σ::Array{Int8}, σ′::Array{Int8}, p::Float64, r::Float64)` method.

See also [`step!`](@ref), [`advance_and_measure!`](@ref), [`advance_parallel!`](@ref), [`advance_async!`](@ref).
"""
function advance!(ca::BrassCA, p::Float64, r::Float64, n_steps::Integer = 1)
    @assert n_steps > 0 "Number of steps must be positive."
    # Auxiliar state
    σ′ = similar(ca.σ)
    # Time steps iteration
    @inbounds for _ in 1:floor(Int, n_steps / 2)
        step!(ca, ca.σ, σ′, p, r)
        step!(ca, σ′, ca.σ, p, r)
    end
    if isodd(n_steps)
        step!(ca, ca.σ, σ′, p, r)
        ca.σ = σ′
    end
end

"""
    advance_and_measure!(measurement::Function, ca::BrassCA, p::Float64, r::Float64, n_steps::Integer = 1)

Advance the state of a Brass CA `ca` *synchronously* by `n_steps` time steps and performs a measurement given by the function `measurement` after wach time step.

The function `measurement` must take as its sole argument the state of CA.

    measurement::(Array -> ResultType)

Each specific type of Brass CA `BrassCASpecific` must provide its own implementation of the `step!(ca::BrassCASpecific, σ::Array{Int8}, σ′::Array{Int8}, p::Float64, r::Float64)` method.

# Returns:
- `results::Vector{ResultType}`: Vector containing the mearurements results

See also [`step!`](@ref), [`advance!`](@ref), [`advance_parallel_and_measure!`](@ref), [`advance_async_and_measure!`](@ref).
"""
function advance_and_measure!(measurement::Function, ca::BrassCA, p::Float64, r::Float64, n_steps::Integer = 1)
    @assert n_steps > 0 "Number of steps must be positive."
    # Measurement results
    ResultType = Base.return_types(measurement, (typeof(ca.σ),))[1]
    results = Array{ResultType}(undef, n_steps + 1)
    # First measurement
    results[1] = measurement(ca.σ)
    # Auxiliar state
    σ′ = similar(ca.σ)
    # Time steps iteration
    @inbounds for t in 1:floor(Int, n_steps / 2)
        step!(ca, ca.σ, σ′, p, r)
        results[2*t] = measurement(σ′)
        step!(ca, σ′, ca.σ, p, r)
        results[2*t+1] = measurement(ca.σ)
    end
    if isodd(n_steps)
        step!(ca, ca.σ, σ′, p, r)
        ca.σ = σ′
        results[end] = measurement(ca.σ)
    end
    return results
end

"""
    advance_parallel!(ca::BrassCA, p::Float64, r::Float64, n_steps::Integer = 1)

Advance the state of a Brass CA `ca` *synchronously* by `n_steps` time steps.

For each time step the sites of the CA are updated in parallel.

The probabilities `p` and `r` are parameters of the model.

Each specific type of Brass CA `BrassCASpecific` must provide its own implementation of the `step_parallel!(ca::BrassCASpecific, σ::Array{Int8}, σ′::Array{Int8}, p::Float64, r::Float64)` method.

See also [`step_parallel!`](@ref), [`advance_parallel_and_measure!`](@ref), [`advance!`](@ref), [`advance_async!`](@ref).
"""
function advance_parallel!(ca::BrassCA, p::Float64, r::Float64, n_steps::Integer = 1)
    @assert n_steps > 0 "Number of steps must be positive."
    # Auxiliar state
    σ′ = similar(ca.σ)
    # Time steps iteration
    @inbounds for _ in 1:floor(Int, n_steps / 2)
        step_parallel!(ca, ca.σ, σ′, p, r)
        step_parallel!(ca, σ′, ca.σ, p, r)
    end
    if isodd(n_steps)
        step_parallel!(ca, ca.σ, σ′, p, r)
        ca.σ = σ′
    end
end

"""
    advance_parallel_and_measure!(measurement::Function, ca::BrassCA, p::Float64, r::Float64, n_steps::Integer = 1)

Advance the state of a Brass CA `ca` *synchronously* by `n_steps` time steps and performs a measurement given by the function `measurement` after wach time step.

For each time step the sites of the CA are updated in parallel.

The function `measurement` must take as its sole argument the state of CA.

    measurement::(Array -> ResultType)

Each specific type of Brass CA `BrassCASpecific` must provide its own implementation of the `step_parallel!(ca::BrassCASpecific, σ::Array{Int8}, σ′::Array{Int8}, p::Float64, r::Float64)` method.

# Returns:
- `results::Vector{ResultType}`: Vector containing the mearurements results

See also [`step_parallel!`](@ref), [`advance_parallel!`](@ref), [`advance_and_measure!`](@ref), [`advance_async_and_measure!`](@ref).
"""
function advance_parallel_and_measure!(measurement::Function, ca::BrassCA, p::Float64, r::Float64, n_steps::Integer = 1)
    @assert n_steps > 0 "Number of steps must be positive."
    # Measurement results
    ResultType = Base.return_types(measurement, (typeof(ca.σ),))[1]
    results = Array{ResultType}(undef, n_steps + 1)
    # First measurement
    results[1] = measurement(ca.σ)
    # Auxiliar state
    σ′ = similar(ca.σ)
    # Time steps iteration
    @inbounds for t in 1:floor(Int, n_steps / 2)
        step_parallel!(ca, ca.σ, σ′, p, r)
        results[2*t] = measurement(σ′)
        step_parallel!(ca, σ′, ca.σ, p, r)
        results[2*t+1] = measurement(ca.σ)
    end
    if isodd(n_steps)
        step_parallel!(ca, ca.σ, σ′, p, r)
        ca.σ = σ′
        results[end] = measurement(ca.σ)
    end
    return results
end

"""
    advance_async!(ca::BrassCA, p::Float64, r::Float64, n_steps::Integer = 1)

Advance the state of a Brass CA `ca` *asynchronously* by `n_steps` time steps.

The probabilities `p` and `r` are parameters of the model.

Each specific type of Brass CA `BrassCASpecific` must provide its own implementation of the `step_async!(ca::BrassCASpecific, p::Float64, r::Float64)` method.

See also [`step_async!`](@ref), [`advance!`](@ref), [`advance_parallel!`](@ref), [`advance_async_and_measure!`](@ref).
"""
function advance_async!(ca::BrassCA, p::Float64, r::Float64, n_steps::Integer = 1)
    @assert n_steps > 0 "Number of steps must be positive."
    # Time steps iteration
    @inbounds for _ in 1:n_steps
        step_async!(ca, p, r)
    end
end

"""
    advance_async_and_measure!(measurement::Function, ca::BrassCA, p::Float64, r::Float64, n_steps::Integer = 1)

Advance the state of a Brass CA `ca` *asynchronously* by `n_steps` time steps and performs a measurement given by the function `measurement` after wach time step.

The function `measurement` must take as its sole argument the state of CA.

    measurement::(Array -> ResultType)

Each specific type of Brass CA `BrassCASpecific` must provide its own implementation of the `step_async!(ca::BrassCASpecific, p::Float64, r::Float64)` method.

# Returns:
- `results::Vector{ResultType}`: Vector containing the mearurements results

See also [`step_async!`](@ref), [`advance_async!`](@ref), [`advance_and_measure!`](@ref), [`advance_parallel_and_measure!`](@ref).
"""
function advance_async_and_measure!(measurement::Function, ca::BrassCA, p::Float64, r::Float64, n_steps::Integer = 1)
    @assert n_steps > 0 "Number of steps must be positive."
    # Measurement results
    ResultType = Base.return_types(measurement, (typeof(ca.σ),))[1]
    results = Array{ResultType}(undef, n_steps + 1)
    # First measurement
    results[1] = measurement(ca.σ)
    # Time steps iteration
    @inbounds for t in 2:n_steps+1
        step_async!(ca, p, r)
        results[t] = measurement(ca.σ)
    end
    return results
end

"""
    BrassCAMeanField

Brass CA with mean field interaction:
Every site interacts with every other site.

# Fields:
- `σ::Vector{Int8}`: State of the CA
"""
mutable struct BrassCAMeanField <: BrassCA

    "State of the CA"
    σ::Vector{Int8}

    @doc """
        BrassCAMeanField(N::Integer, ::Val{:rand})

    Construct a Brass CA with mean field interaction with `N` sites and random initial state `σ ∈ {-1, 0, +1}`.
    """
    BrassCAMeanField(N::Integer, ::Val{:rand}) = new(rand(Int8[0, +1, -1], N))

    @doc """
        BrassCAMeanField(N::Integer, (::Val{0} || ::Val{+1} || ::Val{-1}))

    Construct a Brass CA with mean field interaction with `N` sites and a given initial state.
    """
    BrassCAMeanField(N::Integer, ::Val{0}) = new(fill(Int8(0), N))
    BrassCAMeanField(N::Integer, ::Val{+1}) = new(fill(Int8(+1), N))
    BrassCAMeanField(N::Integer, ::Val{-1}) = new(fill(Int8(-1), N))
end

@doc raw"""
    mean_field_nn_sum(σ::Array, i::Integer)

Brass CA mean field sum of nearest neighbors of site `i` given a state `σ`.

That is, the sum of every site in the CA except site `i`:

``sᵢ = ∑_{k≠i} σₖ``

"""
@inline function mean_field_nn_sum(σ::Array, i::Integer)
    N = length(σ)
    sum(σ[k] for ks in (1:(i-1), (i+1):N) for k in ks)
end


"""
Single step of the Brass CA mean field.

# Arguments:
- `ca`: Brass CA mean field
- `σ`: Current state of the CA
- `σ′`: Array to store resulting state of the CA
- `p` and `r`: Probabilities of the model
"""
@inline function step!(ca::BrassCAMeanField, σ::Vector, σ′::Vector, p::Float64, r::Float64)
    # Iterate over every site
    for i in LinearIndices(ca.σ)
        σᵢ = σ[i]
        sᵢ = sign(mean_field_nn_sum(σ, i))
        σ′[i] = brass_ca_new_site_state(σᵢ, sᵢ, p, r)
    end
end

"""
Single step of the Brass CA mean field.

# Arguments:
- `ca`: Brass CA mean field
- `σ`: Current state of the CA
- `σ′`: Array to store resulting state of the CA
- `p` and `r`: Probabilities of the model
"""
@inline function step_parallel!(ca::BrassCAMeanField, σ::Vector, σ′::Vector, p::Float64, r::Float64)
    # Iterate over every site
    @inbounds Threads.@threads for i in LinearIndices(ca.σ)
        σᵢ = σ[i]
        sᵢ = sign(mean_field_nn_sum(σ, i))
        σ′[i] = brass_ca_new_site_state(σᵢ, sᵢ, p, r)
    end
end

"""
Single *asynchronous* step of the Brass CA mean field, updating a given site.

# Arguments:
- `ca`: Brass CA mean field
- `i`: site to be updated
- `p` and `r`: Probabilities of the model
"""
@inline function step_async!(ca::BrassCAMeanField, i::Integer, p::Float64, r::Float64)
    σᵢ = ca.σ[i]
    # Get sign of the sum of nearest neighbors
    sᵢ = sign(mean_field_nn_sum(ca.σ, i))
    # Transition to new site state
    ca.σ[i] = brass_ca_new_site_state(σᵢ, sᵢ, p, r)
end

"""
Single *asynchronous* step of the Brass CA mean field, updating a random site.

# Arguments:
- `ca`: Brass CA mean field
- `p` and `r`: Probabilities of the model
"""
@inline function step_async!(ca::BrassCAMeanField, p::Float64, r::Float64)
    i = rand(1:length(ca))
    step_async!(ca, i, p, r)
end

"""
    BrassCASquareLattice

Brass CA on a periodic multidimensional square lattice.

# Fields:
- `σ::Array{Int8}`: State of the CA
"""
mutable struct BrassCASquareLattice <: BrassCA
    "State of the CA"
    σ::Array{Int8}

    @doc raw"""
        BrassCASquareLattice(L::NTuple{N,Integer}, σ₀::Int8) where {N}

    Construct Brass CA with dimensions `L` and site initial state `σ₀`
    """
    BrassCASquareLattice(L::NTuple{N,Integer}, σ₀::Int8) where {N} = new(fill(σ₀, L...))

    @doc raw"""
        BrassCASquareLattice(::Val{N}, L::Integer, σ₀::Int8) where {N}

    Construct a `dim`-dimensional square Brass CA of side length `L` and site initial state `σ₀`
    """
    BrassCASquareLattice(::Val{N}, L::Integer, σ₀::Int8) where {N} = BrassCASquareLattice(ntuple(_ -> L, Val(N)), σ₀)

    @doc raw"""
        BrassCASquareLattice(L::NTuple{N,Integer}, ::Val{:rand}) where {N}

    Construct Brass CA with dimensions `L` and random initial state
    """
    BrassCASquareLattice(L::NTuple{N,Integer}, ::Val{:rand}) where {N} = new(rand(-1:1, L...))

    @doc raw"""
        BrassCASquareLattice(::Val{N}, L::Integer, ::Val{:rand}) where {N}

    Construct a `dim`-dimensional square Brass CA of side length `L` and random initial state
    """
    BrassCASquareLattice(::Val{N}, L::Integer, ::Val{:rand}) where {N} = BrassCASquareLattice(ntuple(_ -> L, Val(N)), Val(:rand))

end

"""
Generated function for calculating the sum of nearest neighbors of a given site in a periodic multidimensional square lattice.

# Arguments:
- `σ`: State of the lattice
- `s`: Cartesian index of the chosen site

# Returns:
- Function that calculates the sum of the nearest neighbors of `s`
"""
function square_lattice_nn_sum_impl(σ::Type{Array{T,dim}}, i::Type{CartesianIndex{dim}}) where {T,dim}
    # Loop on the dimensions
    terms = map(1:dim) do d
        # Indices for both nearest neighbors in the current dimension
        idx_prev_nn = :(mod1(i[$d] - 1, size(σ, $d)))
        idx_next_nn = :(mod1(i[$d] + 1, size(σ, $d)))
        # Fill indices before and after the current dimension
        idx_before = [:(i[$k]) for k in 1:d-1]
        idx_after = [:(i[$k]) for k in d+1:dim]
        # Term correspondig to dimension $d$
        :(σ[$(idx_before...), $idx_prev_nn, $(idx_after...)] + σ[$(idx_before...), $idx_next_nn, $(idx_after...)])
    end
    # Return sum of all terms
    :(+($(terms...)))
end

"""
Extract generated function that calculates the sum of the nearest neighbors of a give site
"""
@generated function square_lattice_nn_sum(σ::Array{T,dim}, i::CartesianIndex{dim}) where {T,dim}
    square_lattice_nn_sum_impl(σ, i)
end

# Test generated function
#display(square_lattice_nn_sum_impl(Array{Int,3}, CartesianIndex{3}))

"""
Single step of the Brass CA on a square lattice.

# Arguments:
- `ca`: Brass CA on a square lattice
- `σ`: Current state of the CA
- `σ′`: Array to store resulting state of the CA
- `p` and `r`: Probabilities of the model
"""
@inline function step!(ca::BrassCASquareLattice, σ::Array, σ′::Array, p::Float64, r::Float64)
    # Iterate over every site
    for i in CartesianIndices(ca.σ)
        σᵢ = σ[i]
        # Get sign of the sum of nearest neighbors
        sᵢ = sign(square_lattice_nn_sum(σ, i))
        # Transition to new site state
        σ′[i] = brass_ca_new_site_state(σᵢ, sᵢ, p, r)
    end
end

"""
Single step of the Brass CA on a square lattice.

# Arguments:
- `ca`: Brass CA on a square lattice
- `σ`: Current state of the CA
- `σ′`: Array to store resulting state of the CA
- `p` and `r`: Probabilities of the model
"""
@inline function step_parallel!(ca::BrassCASquareLattice, σ::Array, σ′::Array, p::Float64, r::Float64)
    # Iterate over every site
    @inbounds Threads.@threads for i in CartesianIndices(ca.σ)
        σᵢ = σ[i]
        # Get sign of the sum of nearest neighbors
        sᵢ = sign(square_lattice_nn_sum(σ, i))
        # Transition to new site state
        σ′[i] = brass_ca_new_site_state(σᵢ, sᵢ, p, r)
    end
end

"""
Single *asynchronous* step of the Brass CA on a square lattice, updating a given site.

# Arguments:
- `ca`: Brass CA on a square lattice
- `i`: site to be updated
- `p` and `r`: Probabilities of the model
"""
@inline function step_async!(ca::BrassCASquareLattice, i::CartesianIndex{dim}, p::Float64, r::Float64) where {dim}
    σᵢ = ca.σ[i]
    # Get sign of the sum of nearest neighbors
    sᵢ = sign(square_lattice_nn_sum(ca.σ, i))
    # Transition to new site state
    ca.σ[i] = brass_ca_new_site_state(σᵢ, sᵢ, p, r)
end

"""
Single *asynchronous* step of the Brass CA on a square lattice, updating a random site.

# Arguments:
- `ca`: Brass CA on a square lattice
- `p` and `r`: Probabilities of the model
"""
@inline function step_async!(ca::BrassCASquareLattice, p::Float64, r::Float64)
    i = rand(CartesianIndices(ca.σ))
    step_async!(ca, i, p, r)
end

"""
Calculate the average final value of concentration.
"""
function final_concentration_avg(L::NTuple{N,Int}, σ₀::Int, p::Float64, r::Float64, n_steps::Int, n_samples::Int) where {N}
    @assert n_steps > 0 "Number of steps must be positive."
    @assert n_samples > 0 "Number of samples must be positive."

    count = Array{Int}(undef, n_samples, 3)
    @inbounds for k in 1:n_samples
        local brass = BrassCASquareLattice(L, σ₀)
        advance!(brass, p, r, n_steps)
        count[k, :] .= state_count(brass.σ)
    end
    conc = count ./ prod(L)
    conc_avg = mean(conc, dims = 1)
    conc_std = std(conc, dims = 1, corrected = true, mean = conc_avg)
    conc_avg, conc_std
end

"""
Brass CA on an abitrary graph `g` with states of each node stored in the vector `σ`
"""
mutable struct BrassCAGraph <: BrassCA
    "Graph structure"
    g::Graph
    "State at each node"
    σ::Vector{Int8}
    "Construct Brass CA over a given graph `g` with initial state `σ₀`"
    BrassCAGraph(g::Graph, σ₀::Int8) = new(g, fill(σ₀, nv(g)))
    "Construct Brass CA over a given graph `g` with random initial state"
    BrassCAGraph(g::Graph, ::Val{:rand}) = new(g, rand(-1:1, nv(g)))
end

"""
Single step of the Brass CA on an arbitrary graph.

# Arguments:
- `ca`: Brass CA on an arbitrary graph
- `σ`: Current state of the CA
- `σ′`: Array to store resulting state of the CA
- `p` and `r`: Probabilities of the model
"""
@inline function step!(ca::BrassCAGraph, σ::Vector{Int}, σ′::Vector{Int}, p::Float64, r::Float64)
    # Iterate over every site
    for i in 1:nv(ca.g)
        σᵢ = σ[i]
        # Get sign of the sum of nearest neighbors
        sᵢ = sign(sum(σ[neighbors(ca.g, i)]))
        # Transition to new site state
        σ′[i] = brass_ca_new_site_state(σᵢ, sᵢ, p, r)
    end
end

"""
Single step of the Brass CA on an arbitrary graph.

# Arguments:
- `ca`: Brass CA on an arbitrary graph
- `σ`: Current state of the CA
- `σ′`: Array to store resulting state of the CA
- `p` and `r`: Probabilities of the model
"""
@inline function step_parallel!(ca::BrassCAGraph, σ::Vector{Int}, σ′::Vector{Int}, p::Float64, r::Float64)
    # Iterate over every site
    @inbounds Threads.@threads for i in 1:nv(ca.g)
        σᵢ = σ[i]
        # Get sign of the sum of nearest neighbors
        sᵢ = sign(sum(σ[neighbors(ca.g, i)]))
        # Transition to new site state
        σ′[i] = brass_ca_new_site_state(σᵢ, sᵢ, p, r)
    end
end

"""
Single *asynchronous* step of the Brass CA on a square lattice, updating a given site.

# Arguments:
- `ca`: Brass CA on a square lattice
- `i`: site to be updated
- `p` and `r`: Probabilities of the model
"""
@inline function step_async!(ca::BrassCAGraph, i::Integer, p::Float64, r::Float64)
    σᵢ = ca.σ[i]
    # Get sign of the sum of nearest neighbors
    sᵢ = sign(sum(ca.σ[neighbors(ca.g, i)]))
    # Transition to new site state
    ca.σ[i] = brass_ca_new_site_state(σᵢ, sᵢ, p, r)
end

"""
Single *asynchronous* step of the Brass CA on a square lattice, updating a random site.

# Arguments:
- `ca`: Brass CA on a square lattice
- `p` and `r`: Probabilities of the model
"""
@inline function step_async!(ca::BrassCAGraph, p::Float64, r::Float64)
    i = rand(1:length(ca))
    step_async!(ca, i, p, r)
end

end
