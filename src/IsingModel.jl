module IsingModel

export Ising, IsingMeanField, IsingSquareLattice, IsingGraph,
    ISING_SQ_LAT_2D_BETA_CRIT,
    magnet_total, magnet, magnet_moment,
    magnet_total_local, energy_local,
    flip!,
    energy,
    nearest_neighbors,
    metropolis!, metropolis_and_measure_total_magnet!, metropolis_and_measure_energy!

using Graphs

include("Metaprogramming.jl")
include("Geometry.jl")

using .Metaprogramming

"""
    Ising

Supertype for all Ising model systems.
"""
abstract type Ising end

"""
    SpinType

Type for the representation of the spin value in memory.
"""
SpinType = Int8

"""
    SpinVals

The allowed values for the spin on an Ising system:
- Val{-1} corresponds to DOWN
- Val{+1} corresponds to UP
"""
SpinVals = Union{Val{-1},Val{+1}}

"""
    SpinState

Enumeration of possible spin values.
"""
@enum SpinState::SpinType begin
    up = +1
    down = -1
end

"""
    size(ising::Ising)

Size of the state of an Ising system `ising`.
"""
@inline Base.size(ising::Ising) = size(ising.state)

"""
    size(ising::Ising, dim::Integer)

Size of the state of an Ising system `ising` along a given dimension `dim`.
"""
@inline Base.size(ising::Ising, dim::Integer) = size(ising.state, dim)

"""
    length(ising::Ising)

Total number of sites of an Ising system `ising`.
"""
@inline Base.length(ising::Ising) = length(ising.state)

@doc raw"""
    magnet_total(ising::Ising)

Total magnetization of an Ising system `ising`.

The total magnetization is defined as the sum of all site states:

``M = ∑ᵢ σᵢ``

See also: [`magnet`](@ref), [`magnet_moment`](@ref).
"""
@inline magnet_total(ising::Ising) = @inbounds sum(ising.state)

@doc raw"""
    magnet(ising::Ising)

Magnetization per site of an Ising system `ising`.

``m = M / N = ∑ᵢ σᵢ / N``

See also: [`magnet_total`](@ref), [`magnet_moment`](@ref).
"""
@inline magnet(ising::Ising) = magnet_total(ising) / length(ising)

@doc raw"""
    magnet_moment(ising::Ising, k::integer)

Calculates the k-th momentum of the magnetization of an Ising system `ising`.

``mᵏ = 1/nᵏ (∑ᵢ σᵢ)ᵏ``

See also: [`magnet`](@ref), [`magnet_total`](@ref).
"""
@inline magnet_moment(ising::Ising, k::Integer) = magnet(ising)^k

"""
    magnet_total_local(ising::Ising, i::Integer)

The difference in total magnetization of an Ising system `ising` if the `i`-th spin were to be flipped.
"""
@inline magnet_total_local(ising::Ising, i::Integer) = @inbounds -2 * ising.state[i]

"""
    energy_local(ising::Ising, i::Integer, h::Real=0)

Energy difference for an Ising system `ising` associated with the flip of the `i`-th spin
and subject to external magnetic field `h`.

If no external magnetic field is provided, it is assumed to be `h=0`.

Each specific type of Ising model `IsingSpecific` must provide its own implementation of `nearest_neighbors(ising::IsingSpecific, i::Integer)`.
"""
@inline energy_local(ising::Ising, i::Integer, h::Real) = @inbounds 2 * ising.state[i] * (sum(ising.state[nn] for nn ∈ nearest_neighbors(ising, i)) + h)

@inline energy_local(ising::Ising, i::Integer) = @inbounds 2 * ising.state[i] * sum(ising.state[nn] for nn ∈ nearest_neighbors(ising, i))

"""
    flip!(ising::Ising, i::Integer)

Flips the `i-th` spin on the Ising system `ising`.
"""
@inline function flip!(ising::Ising, i::Integer)
    @inbounds ising.state[i] = -ising.state[i]
end

@doc raw"""
    IsingMeanField

Ising system with mean field interaction:
Every spin interacts equally with every other spin.

# Fields:
- `state::Vector{SpinType}`: State of the system
"""
mutable struct IsingMeanField <: Ising

    "State of the system"
    state::Vector{SpinType}

    @doc raw"""
        IsingMeanField(N::Integer, σ::SpinState)

    Construct an Ising system with mean field interaction with `N` sites and and a given initial state `σ`.
    """
    IsingMeanField(N::Integer, σ::SpinState) = new(fill(Integer(σ), N))

    @doc raw"""
        IsingMeanField(N::Integer, ::Val{:rand})

    Construct an Ising system with mean field interaction with `N` sites and random initial state `σ ∈ {-1, +1}`.
    """
    IsingMeanField(N::Integer, ::Val{:rand}) = new(rand(Integer.(instances(SpinState)), N))

    @doc raw"""
        IsingMeanField(N::Integer, (::Val{-1} || ::Val{+1}))

    Construct an Ising system with mean field interaction with `N` sites and and a given initial state.
    """
    IsingMeanField(N::Integer, σ::SpinVals) = new(fill(SpinType(extract_val(σ)), N))
end

"""
    energy(ising::IsingMeanField, h::Real=0)

Total energy of an Ising system `ising` with mean field interaction subject to an external magnetic field `h`.

If no external magnetic field is provided, it is assumed to be `h=0`.
"""
@inline energy(ising::IsingMeanField) = @inbounds -sum(ising.state[i] * sum(ising.state[begin:(i-1)]) for i ∈ eachindex(ising.state))

@inline energy(ising::IsingMeanField, h::Real) = @inbounds energy(ising) - h * magnet_total(ising)

"""
    nearest_neighbors(ising::IsingMeanField, i::Integer)

For an Ising system `ising` with mean field interaction, get the nearest neighobors of a given site `i`.

That is, every site except for `i` itself.
"""
@inline nearest_neighbors(ising::IsingMeanField, i::Integer) = vcat(1:i-1, i+1:length(ising))

"""
    IsingSquareLattice{N}

Ising system on a `N`-dimensional square lattice with nearest neighbor interaction.

# Fields:
- `state::Array{SpinType,N}`: State of the system
"""
mutable struct IsingSquareLattice{N} <: Ising

    "State of the Ising system"
    state::Array{SpinType,N}

    """
        IsingSquareLattice(size::NTuple{N,Integer}, σ::SpinState) where {N}

    Construct a new Ising system in a multidimensional square lattice of dimensions provided by `size`,
    with nearest neighbor interaction and with all spins with same initial state `σ`.
    """
    IsingSquareLattice(size::NTuple{N,Integer}, σ::SpinState) where {N} = new{N}(fill(Integer(σ), size))

    """
        IsingSquareLattice(size::NTuple{N,Integer}, ::Val{:rand}) where {N}

    Construct a new Ising system in a multidimensional square lattice of dimensions provided by `size`,
    with nearest neighbor interaction and with spins in random states
    """
    IsingSquareLattice(size::NTuple{N,Integer}, ::Val{:rand}) where {N} = new{N}(rand(Integer.(instances(SpinState)), size))

    """
        IsingSquareLattice(size::NTuple{N,Integer}, (::Val{+1} || ::Val{-1})) where {N}

    Construct a new Ising system in a multidimensional square lattice of dimensions provided by `size`,
    with nearest neighbor interaction and with all spins with same initial state.
    """
    IsingSquareLattice(size::NTuple{N,Integer}, σ::SpinVals) where {N} = new{N}(fill(SpinType(extract_val(σ)), size))
end

@doc raw"""
    ISING_SQ_LAT_2D_BETA_CRIT

Critical temperature for the Ising system on a 2D square lattice.

``β_c = \frac{\log{(1 + √{2})}}{2}``
"""
const ISING_SQ_LAT_2D_BETA_CRIT = 0.5 * log1p(sqrt(2))

"""
    flip!(ising::IsingSquareLattice, idx::CartesianIndex)

Flips the spin at site described by the cartesian index `idx` in the Ising system in a multidimensional square lattice `ising`.
"""
@inline function flip!(ising::IsingSquareLattice, idx::CartesianIndex)
    ising.state[idx] = -ising.state[idx]
end

@doc raw"""
    energy(ising::IsingSquareLattice{N}, h::Real=0) where {N}

Total energy of an `N`-dimensional Ising square lattice system `ising` with external magnetic field `h`.

If no external magnetic field is provided it is assumed to be `h=0`.

Given by the Hamiltonian:

``H = - J ∑_⟨i,j⟩ sᵢ sⱼ - h ∑_i sᵢ``

where `⟨i,j⟩` means that `i` and `j` are nearest neighbors.
"""
function energy(ising::IsingSquareLattice{N}) where {N}
    # Interaction energy
    H::Int64 = 0
    # Loop on dimensions
    @inbounds for d ∈ 1:N
        # Bulk
        front_bulk = selectdim(ising.state, d, 1:(size(ising.state, d)-1))
        back_bulk = selectdim(ising.state, d, 2:size(ising.state, d))
        H -= sum(front_bulk .* back_bulk)
        # Periodic boundaries
        first_slice = selectdim(ising.state, d, 1)
        last_slice = selectdim(ising.state, d, size(ising.state, d))
        H -= sum(last_slice .* first_slice)
    end
    return H
end

@inline energy(ising::IsingSquareLattice, h::Real) = energy(ising) - h * magnet_total(ising)

"""
    nearest_neighbors(ising::IsingSquareLattice{N}, idx::CartesianIndex{N}) where {N}

Get the cartesian coordinates of the nearest neighbours of a given spin located at `idx`
in a multidimensional square lattice `ising`.

For a `N`-dimensional lattice each spin has 2`N` nearest neighbors.
"""
@inline nearest_neighbors(ising::IsingSquareLattice{N}, idx::CartesianIndex{N}) where {N} = @inbounds Geometry.square_lattice_nearest_neighbors_flat(ising.state, idx)

"""
    nearest_neighbors(ising::IsingSquareLattice, i::Integer)

Get the cartesian coordinates of the nearest neighbours of the `i`-th spin
in a multidimensional square lattice `ising`.

For a `N`-dimensional lattice each spin has 2`N` nearest neighbors.
"""
@inline function nearest_neighbors(ising::IsingSquareLattice, i::Integer)
    @inbounds idx = CartesianIndices(ising.state)[i]
    return nearest_neighbors(ising, idx)
end

@doc raw"""
    energy_local(ising::IsingSquareLattice, idx::CartesianIndex, h::Real=0)

Energy difference for an Ising system in a multidimensional square lattice with nearest neighbor interaction `ising`
associated with a single spin flip at site `idx` subject to external magnetic field `h`.

If no external magnetic field is provided, it is assumed `h=0`.

# Arguments:
- `ising::IsingSquareLattice`: Ising system
- `idx::CartesianIndex`: Spin site
- `h::Real=0`: External magnetic field
"""
@inline energy_local(ising::IsingSquareLattice, idx::CartesianIndex, h::Real) = @inbounds 2 * ising.state[idx] * (Geometry.square_lattice_nearest_neighbors_sum(ising.state, idx) + h)

@inline energy_local(ising::IsingSquareLattice, idx::CartesianIndex) = @inbounds 2 * ising.state[idx] * Geometry.square_lattice_nearest_neighbors_sum(ising.state, idx)

@doc raw"""
    energy_local(ising::IsingSquareLattice, i::Integer, h::Real=0)

Energy difference for an Ising system in a multidimensional square lattice with nearest neighbor interaction `ising`
associated with the flip of the `i`-th spin and subject to external magnetic field `h`.

If no external magnetic field is provided, it is assumed `h=0`.

# Arguments:
- `ising::IsingSquareLattice`: Ising system
- `i::Integer`: Spin number
- `h::Real=0`: External magnetic field
"""
@inline energy_local(ising::IsingSquareLattice, i::Integer, h::Real) = energy_local(ising, CartesianIndices(ising.state)[i], h)

@inline energy_local(ising::IsingSquareLattice, i::Integer) = energy_local(ising, CartesianIndices(ising.state)[i])


"""
    IsingGraph

Ising model on an arbitrary graph.

# Fields:
- `g::Graph`: Graph structure of the system
- `state::Vector`: Vector containing the state of each node in the system
"""
mutable struct IsingGraph <: Ising

    "Graph structure of the system"
    g::Graph

    "State at each node"
    state::Vector{SpinType}

    """
        IsingGraph(g::Graph, σ::SpinState)

    Construct a new Ising system with graph structure `g` with all spins with same initial state `σ`.
    """
    IsingGraph(g::Graph, σ::SpinState) = new(g, fill(Integer(σ), nv(g)))

    """
        IsingGraph(g::Graph, ::Val{:rand})

    Construct a new Ising system with graph structure `g` and random initial states at each node.
    """
    IsingGraph(g::Graph, ::Val{:rand}) = new(g, rand(Integer.(instances(SpinState)), nv(g)))

    """
        IsingGraph(g::Graph, (::Val{+1} || ::Val{-1}))

    Construct a new Ising system with graph structure `g` with all spins with same initial state.
    """
    IsingGraph(g::Graph, σ::SpinVals) = new(g, fill(SpinType(extract_val(σ)), nv(g)))
end

"""
    energy(ising::IsingGraph, h::Real=0)

Total energy of an Ising system `ising` over a graph subject to external magnetic field `h`.

If no external magnetic field is provided it is assumed to be `h=0`.
"""
@inline energy(ising::IsingGraph) = @inbounds -sum(ising.state[src(e)] * ising.state[dst(e)] for e ∈ edges(ising.g))


@inline energy(ising::IsingGraph, h::Real) = @inbounds energy(ising) - h * magnet_total(ising)

"""
    nearest_neighbors(ising::IsingGraph, v::Integer)

For an Ising system over a graph `ising`, get the nearest neighobors of a given site `v`.
"""
@inline nearest_neighbors(ising::IsingGraph, v::Integer) = neighbors(ising.g, v)

@doc raw"""
    metropolis!(ising::Ising, β::Real, n_steps::Integer, h::Real=0)

Metropolis sampling with external magnetic filed `h`.

If no `h` is provided it is assumed that there is no external magnetic field.

The Boltzmann constant `k_B` and the interaction strenght `J` are assumed to be unitary.

Each specific type of Ising system must `IsingSpecific` provide its own implementation of the `energy_local(ising::IsingSpecific, i::Integer, h::Real=0)` method.

# Arguments:
- `ising::Ising`: Ising system to be sampled
- `β::Real`: Inverse of the temperature (`1/T`)
- `n_steps::Integer`: Number of steps to sample
- `h::Real=0`: Intensity of the external magnetic field
"""
function metropolis!(ising::Ising, β::Real, n_steps::Integer, h::Real)
    # Sampling loop
    @inbounds for _ ∈ 1:n_steps
        # Select random spin
        i = rand(1:length(ising))
        # Get energy difference
        ΔH = energy_local(ising, i, h)
        # Metropolis prescription
        (ΔH < 0 || exp(-β * ΔH) > rand()) && flip!(ising, i)
    end
end

function metropolis!(ising::Ising, β::Real, n_steps::Integer)
    # Sampling loop
    @inbounds for _ ∈ 1:n_steps
        # Select random spin
        i = rand(1:length(ising))
        # Get energy difference
        ΔH = energy_local(ising, i)
        # Metropolis prescription
        (ΔH < 0 || exp(-β * ΔH) > rand()) && flip!(ising, i)
    end
end

"""
    metropolis_and_measure_total_magnet!(ising::Ising, β::Real, h::Real=0, n_steps::Integer)

Metropolis sample an Ising system `ising` at a given temperature `β` and subject to external magnetic field `h`
for a number of steps `n_steps` and measure the total magnetization at the end of each step.

If no `h` is provided it is assumed that there is no external magnetic field.

# Arguments:
- `ising::Ising`: Ising system to be sampled
- `β::Real`: Temperature of the system (β = 1/T)
- `h::Real=0`: External magnetic field
- `n_steps::Integer`: Number of samples to be generated

# Returns:
- `M_T::Vector`: The total magnetization at each time step
"""
function metropolis_and_measure_total_magnet!(ising::Ising, β::Real, h::Real, n_steps::Integer)
    # Vector to store results
    M_T = Vector{Int64}(undef, n_steps + 1)
    # Initial magnetization
    M_T[1] = magnet_total(ising)
    # Sampling loop
    @inbounds for t ∈ 1:n_steps
        # Select random spin
        i = rand(1:length(ising))
        # Get energy difference
        ΔH = energy_local(ising, i, h)
        # Metropolis prescription
        if ΔH < 0 || exp(-β * ΔH) > rand()
            # Flip spin
            M_T[t+1] = M_T[t] + magnet_total_local(ising, i)
            flip!(ising, i)
        else
            # Do NOT flip spin
            M_T[t+1] = M_T[t]
        end
    end
    return M_T
end

function metropolis_and_measure_total_magnet!(ising::Ising, β::Real, n_steps::Integer)
    # Vector to store results
    M_T = Vector{Int64}(undef, n_steps + 1)
    # Initial magnetization
    M_T[1] = magnet_total(ising)
    # Sampling loop
    @inbounds for t ∈ 1:n_steps
        # Select random spin
        i = rand(1:length(ising))
        # Get energy difference
        ΔH = energy_local(ising, i)
        # Metropolis prescription
        if ΔH < 0 || exp(-β * ΔH) > rand()
            # Flip spin
            M_T[t+1] = M_T[t] + magnet_total_local(ising, i)
            flip!(ising, i)
        else
            # Do NOT flip spin
            M_T[t+1] = M_T[t]
        end
    end
    return M_T
end

"""
    metropolis_and_measure_energy!(ising::Ising, β::Real, h::T=0, n_steps::Integer) where {T<:Real}

Metropolis sample an Ising system `ising` at a given temperature `β` and subject to external magnetic field `h`
for a number of steps `n_steps` and measure the energy at the end of each step.

If no `h` is provided it is assumed that there is no external magnetic field.

# Arguments:
- `ising::Ising`: Ising system to be sampled
- `β::Real`: Temperature of the system (β = 1/T)
- `h::Real=0`: External magnetic field
- `n_steps::Integer`: Number of samples to be generated

# Returns:
- `M_T::Vector{Int64}`: The total magnetization at each time step
"""
function metropolis_and_measure_energy!(ising::Ising, β::Real, h::T, n_steps::Integer) where {T<:Real}
    # Vector to store results
    H = Vector{T}(undef, n_steps + 1)
    # Initial magnetization
    H[1] = energy(ising, h)
    # Sampling loop
    @inbounds for t ∈ 1:n_steps
        # Select random spin
        i = rand(1:length(ising))
        # Get energy difference
        ΔH = energy_local(ising, i, h)
        # Metropolis prescription
        if ΔH < 0 || exp(-β * ΔH) > rand()
            # Flip spin
            H[t+1] = H[t] + ΔH
            flip!(ising, i)
        else
            # Do NOT flip spin
            H[t+1] = H[t]
        end
    end
    return H
end

function metropolis_and_measure_energy!(ising::Ising, β::Real, n_steps::Integer)
    # Vector to store results
    H = Vector{Int64}(undef, n_steps + 1)
    # Initial magnetization
    H[1] = energy(ising)
    # Sampling loop
    @inbounds for t ∈ 1:n_steps
        # Select random spin
        i = rand(1:length(ising))
        # Get energy difference
        ΔH = energy_local(ising, i)
        # Metropolis prescription
        if ΔH < 0 || exp(-β * ΔH) > rand()
            # Flip spin
            H[t+1] = H[t] + ΔH
            flip!(ising, i)
        else
            # Do NOT flip spin
            H[t+1] = H[t]
        end
    end
    return H
end

end
