module IsingModel

using Graphs

export Ising, IsingSquareLattice, IsingMeanField, IsingGraph, magnet_total, magnet, ising_square_lattice_2d_βc, energy_naive, energy, flip!, nearest_neighbors, energy_local, metropolis!

"""
    Ising

Supertype for all Ising model systems.
"""
abstract type Ising end

"""
    size(ising::Ising)

Size of the state of an Ising system `ising`.
"""
@inline Base.size(ising::Ising) = size(ising.σ)

"""
    size(ising::Ising, d::Integer)

Size of the state of an Ising system `ising` along a given dimension `d`.
"""
@inline Base.size(ising::Ising, d::Integer) = size(ising.σ, d)

"""
    length(ising::Ising)

Total number of sites of an Ising system `ising`.
"""
@inline Base.length(ising::Ising) = length(ising.σ)

"""
    flip!(ising::Ising, i::Integer)

Flips the `i-th` spin on the Ising system `ising`.
"""
@inline function flip!(ising::Ising, i::Integer)
    @inbounds ising.σ[i] = -ising.σ[i]
end

@doc raw"""
    magnet_total(ising::Ising)

Total magnetization of an Ising system `ising`.

The total magnetization is defined as the sum of all site states:

``M = ∑ᵢ σᵢ``

See also: [`magnet`](@ref), [`magnet_moment`](@ref).
"""
@inline magnet_total(ising::Ising)::Integer = @inbounds sum(ising.σ)

@doc raw"""
    magnet(ising::Ising)

Magnetization per site of an Ising system `ising`.

``m = M / N = ∑ᵢ σᵢ / N``

See also: [`magnet_total`](@ref), [`magnet_moment`](@ref).
"""
@inline magnet(ising::Ising)::Real = magnet_total(ising) / length(ising)

@doc raw"""
    magnet_moment(ising::Ising, k::integer)

Calculates the k-th momentum of the magnetization of an Ising system `ising`.

``mᵏ = 1/nᵏ (∑ᵢ σᵢ)ᵏ``

See also: [`magnet`](@ref), [`magnet_total`](@ref).
"""
@inline magnet_moment(ising::Ising, k::Integer) = magnet(ising)^k

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
    @inbounds for _ in 1:n_steps
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
    @inbounds for _ in 1:n_steps
        # Select random spin
        i = rand(1:length(ising))
        # Get energy difference
        ΔH = energy_local(ising, i)
        # Metropolis prescription
        (ΔH < 0 || exp(-β * ΔH) > rand()) && flip!(ising, i)
    end
end

"""
    IsingSquareLattice{N}

Ising system with nearest neighbor interaction on a multidimensional square lattice.

# Fields:
- `σ::Array{Int8,N}`: State of the system
"""
mutable struct IsingSquareLattice{N} <: Ising

    "State of the Ising system"
    σ::Array{Int8,N}

    """
        IsingSquareLattice(S::NTuple{N}) where {N}

    Construct with spins in random states
    """
    IsingSquareLattice(S::NTuple{N}) where {N} = new{N}(rand(Int8[-1, +1], S))
    """
        IsingSquareLattice(S::NTuple{N}, (::Val{+1} || ::Val{-1})) where {N}

    Construct with all spins with same state
    """
    IsingSquareLattice(S::NTuple{N}, ::Val{+1}) where {N} = new{N}(fill(Int8(+1), S))
    IsingSquareLattice(S::NTuple{N}, ::Val{-1}) where {N} = new{N}(fill(Int8(-1), S))
end

"""
    ISING_SQ_LAT_2D_BETA_CRIT

Critical temperature for the Ising system on a 2D square lattice.

``β_c = \frac{\log{(1 + √{2})}}{2}``
"""
const ISING_SQ_LAT_2D_BETA_CRIT = 0.5 * log1p(sqrt(2))

@doc raw"""
    energy(ising::IsingSquareLattice{N}, h::Real=0)::Real where {N}

Total energy of an Ising square lattice system `ising` with external magnetic field `h`.

If no external magnetic field is provided it is assumed to be `h=0`.

Given by the Hamiltonian:

``H = - J ∑_⟨i,j⟩ sᵢ sⱼ - h ∑_i sᵢ``

where `⟨i,j⟩` means that `i` and `j` are nearest neighbors.
"""
function energy(ising::IsingSquareLattice{N})::Real where {N}
    # Total energy
    H::Real = 0
    # Interaction energy
    S = size(ising.σ)
    @inbounds for d in 1:N
        # Bulk
        front_bulk = selectdim(ising.σ, d, 1:(S[d]-1))
        back_bulk = selectdim(ising.σ, d, 2:S[d])
        H -= sum(front_bulk .* back_bulk)
        # Periodic boundaries
        first_slice = selectdim(ising.σ, d, 1)
        last_slice = selectdim(ising.σ, d, S[d])
        H -= sum(last_slice .* first_slice)
    end
    return H
end

function energy(ising::IsingSquareLattice{N}, h::Real)::Real where {N}
    # Interaction energy
    H::Real = energy(ising)
    # External field
    H -= h * magnet_total(ising)
    return H
end

"""
    flip!(ising::IsingSquareLattice, idx::CartesianIndex)

Flips the spin at site `idx` in the Ising system in a multidimensional square lattice `ising`.
"""
@inline function flip!(ising::IsingSquareLattice, idx::CartesianIndex)
    ising.σ[idx] = -ising.σ[idx]
end

"""
    nearest_neighbors(ising::IsingSquareLattice{N}, idx::CartesianIndex)::NTuple{2 * N,CartesianIndex{N}} where {N}

Get the cartesian coordinates of the nearest neighbours to a given spin in a multidimensional square lattice

    For a N-dimensional lattice each spin has 2N nearest neighbors.
"""
@inline function nearest_neighbors(ising::IsingSquareLattice{N}, idx::CartesianIndex)::NTuple{2 * N,CartesianIndex{N}} where {N}
    S = size(ising.σ)
    return (ntuple(d -> CartesianIndex(ntuple(i -> i == d ? mod1(idx[i] + 1, S[i]) : idx[i], Val(N))), Val(N))..., ntuple(d -> CartesianIndex(ntuple(i -> i == d ? mod1(idx[i] - 1, S[i]) : idx[i], Val(N))), Val(N))...)
end

@inline function nearest_neighbors(ising::IsingSquareLattice, i::Integer)
    @inbounds idx = CartesianIndices(ising.σ)[i]
    nearest_neighbors(ising, idx)
end

@doc raw"""
    energy_local(ising::IsingSquareLattice, idx::CartesianIndex, h::Real=0)::Real

Energy difference for an Ising system in a multidimensional square lattice with nearest neighbor interaction `ising` associated with a single spin flip at site `idx` subject to external magnetic field `h`.

``ΔH = 2 sᵢ (J ∑_<j> sⱼ + h)``

where the sum `∑_<j>` is over the nearest neighbors of `i`.

If no external magnetic field is provided, it is assumed `h=0`.

# Arguments:
- `ising::IsingSquareLattice`: Ising system
- `idx::CartesianIndex`: Spin site
- `h::Real=0`: External magnetic field
"""
@inline function energy_local(ising::IsingSquareLattice, idx::CartesianIndex, h::Real)
    @inbounds ΔH = 2 * ising.σ[idx] * (sum(ising.σ[nn] for nn ∈ nearest_neighbors(ising, idx)) + h)
    return ΔH
end
@inline function energy_local(ising::IsingSquareLattice, idx::CartesianIndex)
    @inbounds ΔH = 2 * ising.σ[idx] * sum(ising.σ[nn] for nn ∈ nearest_neighbors(ising, idx))
    return ΔH
end

@doc raw"""
    energy_local(ising::IsingSquareLattice, i::Integer, h::Real=0)::Real

Energy difference for an Ising system in a multidimensional square lattice with nearest neighbor interaction `ising` associated with a single spin flip of the `i`-th spin, subject to external magnetic field `h`.

``ΔH = 2 sᵢ (J ∑_<j> sⱼ + h)``

where the sum `∑_<j>` is over the nearest neighbors of `i`.

If no external magnetic field is provided, it is assumed `h=0`.

# Arguments:
- `ising::IsingSquareLattice`: Ising system
- `i::Integer`: Spin number
- `h::Real=0`: External magnetic field
"""
@inline function energy_local(ising::IsingSquareLattice, i::Integer, h::Real)
    @inbounds ΔH = 2 * ising.σ[i] * (sum(ising.σ[nn] for nn ∈ nearest_neighbors(ising, i)) + h)
    return ΔH
end

@inline function energy_local(ising::IsingSquareLattice, i::Integer)
    @inbounds ΔH = 2 * ising.σ[i] * sum(ising.σ[nn] for nn ∈ nearest_neighbors(ising, i))
    return ΔH
end

"""
    IsingMeanField

Ising mean field model in which it is assumed that every spin interacts equally with every other spin.

# Fields:
- `σ::Vector{Int8}`: State of the system
"""
mutable struct IsingMeanField <: Ising

    "State of the system"
    σ::Vector{Int8}

    """
        IsingMeanField(::Val{N}) where {N}

    Construct with random initial state.
    """
    IsingMeanField(::Val{N}) where {N} = new(rand(Int8[-1, +1], Val(N)))

    """
        IsingMeanField(::Val{N}, (::Val{+1} || ::Val{-1})) where {N}

    Construct with all spins with same state.
    """
    IsingMeanField(::Val{N}, ::Val{+1}) where {N} = new(fill(Int8(+1), N))
    IsingMeanField(::Val{N}, ::Val{-1}) where {N} = new(fill(Int8(-1), N))
end

"""
Total energy of the system
"""
function energy(ising::IsingMeanField; J::Real = 1, h::Real = 0)::Real
    @inbounds H = -J * sum(ising.σ[i] * ising.σ[j] for i = 1:length(ising) for j = union(1:i-1, i+1:length(ising)))
    # External field
    if h != 0
        @inbounds H -= h * magnet_total(ising)
    end
    return H
end

"""
Get the nearest neighobors of a given spin
"""
@inline function nearest_neighbors(ising::IsingMeanField, v::Integer)
    union(1:v-1, v+1:length(ising))
end

"""
Energy difference associated with a single spin flip

    ΔH = 2 sᵢ (J ∑_<j> sⱼ + h)
where the sum ∑_<j> is over the nearest neighbors of i.
"""
@inline function energy_local(ising::IsingMeanField, v::Integer; J::Real = 1, h::Real = 0)::Real
    @inbounds ΔH = 2 * ising.σ[v] * (J * sum(ising.σ[nn] for nn ∈ nearest_neighbors(ising, v)) + h)
    return ΔH
end

"""
Ising model on an arbitrary graph
"""
mutable struct IsingGraph <: Ising
    # Graph structure
    g::Graph
    # State at each node
    σ::Vector{Int8}
    # Construct system given a graph with random initial states
    IsingGraph(g::Graph) = new(g, rand(Int8[-1, 1], nv(g)))
    # Construct system over a given graph with all spins with same state
    IsingGraph(g::Graph, ::Val{1}) = new(g, fill(1, nv(g)))
    IsingGraph(g::Graph, ::Val{-1}) = new(g, fill(-1, nv(g)))
end

"""
Total energy of the system
"""
function energy_naive(ising::IsingGraph; J::Real = 1, h::Real = 0)::Real
    H::Real = 0
    @inbounds for e in edges(ising.g)
        H -= ising.σ[src(e)] * ising.σ[dst(e)]
    end
    H *= J
    # External field
    if h != 0
        @inbounds H -= h * magnet_total(ising)
    end
    return H
end

function energy(ising::IsingGraph; J::Real = 1, h::Real = 0)::Real
    @inbounds H = -J * sum(ising.σ[src(e)] * ising.σ[dst(e)] for e in edges(ising.g))
    # External field
    if h != 0
        @inbounds H -= h * magnet_total(ising)
    end
    return H
end

"""
Get the nearest neighobors of a given spin
"""
@inline function nearest_neighbors(ising::IsingGraph, v::Integer)
    neighbors(ising.g, v)
end

"""
Energy difference associated with a single spin flip

    ΔH = 2 sᵢ (J ∑_<j> sⱼ + h)
where the sum ∑_<j> is over the nearest neighbors of i.
"""
@inline function energy_local(ising::IsingGraph, v::Integer; J::Real = 1, h::Real = 0)::Real
    @inbounds ΔH = 2 * ising.σ[v] * (J * sum(ising.σ[nn] for nn ∈ nearest_neighbors(ising, v)) + h)
    return ΔH
end

end
