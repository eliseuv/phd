module IsingModel

export Ising, IsingMeanField, IsingSquareLattice, IsingGraph,
    ISING_SQ_LAT_2D_BETA_CRIT,
    magnet_total, magnet, magnet_moment,
    flip!,
    metropolis!, metropolis_and_measure_total_magnet!,
    nearest_neighbors, energy, energy_local

using Graphs

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

"""
    flip!(ising::Ising, i::Integer)

Flips the `i-th` spin on the Ising system `ising`.
"""
@inline function flip!(ising::Ising, i::Integer)
    @inbounds ising.σ[i] = -ising.σ[i]
end

"""
    flip_total_manget_diff(ising::Ising, i::Integer)

The difference in total magnetization of an Ising system `ising` if the spin at site `i` were to be flipped.
"""
@inline flip_manget_total_diff(ising::Ising, i::Integer) = @inbounds -2 * ising.σ[i]

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
- `M_T::Vector{Int64}`: The total magnetization at each time step
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
            M_T[t+1] = M_T[t] + flip_manget_total_diff(ising, i)
            flip!(ising, i)
        else
            # Do NOT flip spin
            M_T[t+1] = M_T[t]
        end
    end
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
            M_T[t+1] = M_T[t] + flip_manget_total_diff(ising, i)
            flip!(ising, i)
        else
            # Do NOT flip spin
            M_T[t+1] = M_T[t]
        end
    end
end

"""
    metropolis_and_measure_energy!(ising::Ising, β::Real, h::Real=0, n_steps::Integer)

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
function metropolis_and_measure_energy!(ising::Ising, β::Real, h::Real, n_steps::Integer)
    # Vector to store results
    H = Vector(undef, n_steps + 1)
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
end

function metropolis_and_measure_energy!(ising::Ising, β::Real, n_steps::Integer)
    # Vector to store results
    H = Vector(undef, n_steps + 1)
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
    nearest_neighbors(ising::IsingMeanField, i::Integer)

For an Ising system `ising` with mean field interaction, get the nearest neighobors of a given site `i`.

That is, every site except for `i` itself.
"""
@inline nearest_neighbors(ising::IsingMeanField, i::Integer) = union(1:i-1, i+1:length(ising))

"""
    energy(ising::IsingMeanField, h::Real=0)::Real

Total energy of an Ising system `ising` with mean field interaction subject to an external magnetic field `h`.

If no external magnetic field is provided, it is assumed to be `h=0`.
"""
energy(ising::IsingMeanField)::Real = @inbounds -sum(ising.σ[i] * ising.σ[j] for i ∈ LinearIndices(ising.σ) for j ∈ nearest_neighbors(ising, i))

energy(ising::IsingMeanField, h::Real)::Real = @inbounds energy(ising) - h * magnet_total(ising)

"""
    energy_local(ising::IsingMeanField, i::Integer, h::Real=0)::Real

Energy difference for an Ising system with mean field interaction `ising` associated with a single spin flip at site `i` subject to external magnetic field `h`.

If no external magnetic field is provided, it is assumed to be `h=0`.
"""
@inline function energy_local(ising::IsingMeanField, i::Integer, h::Real)::Real
    @inbounds ΔH = 2 * ising.σ[i] * (sum(ising.σ[nn] for nn ∈ nearest_neighbors(ising, i)) + h)
    return ΔH
end

@inline function energy_local(ising::IsingMeanField, i::Integer)::Real
    @inbounds ΔH = 2 * ising.σ[i] * sum(ising.σ[nn] for nn ∈ nearest_neighbors(ising, i))
    return ΔH
end

"""
    IsingSquareLattice{N}

Ising system on a `N`-dimensional square lattice with nearest neighbor interaction.

# Fields:
- `σ::Array{Int8,N}`: State of the system
"""
mutable struct IsingSquareLattice{N} <: Ising

    "State of the Ising system"
    σ::Array{Int8,N}

    """
        IsingSquareLattice(S::NTuple{N,Integer}) where {N}

    Construct a new Ising system in a multidimensional square lattice of dimensions provided by `S`,
    with nearest neighbor interaction and with spins in random states
    """
    IsingSquareLattice(S::NTuple{N,Integer}) where {N} = new{N}(rand(Int8[-1, +1], S))
    """
        IsingSquareLattice(S::NTuple{N,Integer}, (::Val{+1} || ::Val{-1})) where {N}

    Construct a new Ising system in a multidimensional square lattice of dimensions provided by `S`,
    with nearest neighbor interaction and with all spins with same state
    """
    IsingSquareLattice(S::NTuple{N,Integer}, ::Val{+1}) where {N} = new{N}(fill(Int8(+1), S))
    IsingSquareLattice(S::NTuple{N,Integer}, ::Val{-1}) where {N} = new{N}(fill(Int8(-1), S))
end

"""
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
    ising.σ[idx] = -ising.σ[idx]
end

@doc raw"""
    energy(ising::IsingSquareLattice{N}, h::Real=0)::Real where {N}

Total energy of an `N`-dimensional Ising square lattice system `ising` with external magnetic field `h`.

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
    @inbounds for d ∈ 1:Val(N)
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

energy(ising::IsingSquareLattice, h::Real)::Real = @inbounds energy(ising) - h * magnet_total(ising)

"""
    nearest_neighbors(ising::IsingSquareLattice{N}, idx::CartesianIndex)::NTuple{2 * N,CartesianIndex{N}} where {N}

Get the cartesian coordinates of the nearest neighbours of a given spin located at `idx`
in a multidimensional square lattice `ising`.

For a `N`-dimensional lattice each spin has 2`N` nearest neighbors.
"""
@inline function nearest_neighbors(ising::IsingSquareLattice{N}, idx::CartesianIndex)::NTuple{2 * N,CartesianIndex{N}} where {N}
    S = size(ising.σ)
    return (ntuple(d -> CartesianIndex(ntuple(i -> i == d ? mod1(idx[i] + 1, S[i]) : idx[i], Val(N))), Val(N))..., ntuple(d -> CartesianIndex(ntuple(i -> i == d ? mod1(idx[i] - 1, S[i]) : idx[i], Val(N))), Val(N))...)
end

"""
    nearest_neighbors(ising::IsingSquareLattice{N}, i::Integer)::NTuple{2 * N,CartesianIndex{N}} where {N}

Get the cartesian coordinates of the nearest neighbours of the `i`-th spin
in a multidimensional square lattice `ising`.

For a `N`-dimensional lattice each spin has 2`N` nearest neighbors.
"""
@inline function nearest_neighbors(ising::IsingSquareLattice, i::Integer)
    @inbounds idx = CartesianIndices(ising.σ)[i]
    nearest_neighbors(ising, idx)
end

@doc raw"""
    energy_local(ising::IsingSquareLattice, idx::Union{Integer,CartesianIndex}, h::Real=0)::Real

Energy difference for an Ising system in a multidimensional square lattice with nearest neighbor interaction `ising`
associated with a single spin flip at site `idx` subject to external magnetic field `h`.

``ΔH = 2 sᵢ (J ∑_<j> sⱼ + h)``

where the sum `∑_<j>` is over the nearest neighbors of `i`.

If no external magnetic field is provided, it is assumed `h=0`.

# Arguments:
- `ising::IsingSquareLattice`: Ising system
- `idx::CartesianIndex`: Spin site
- `h::Real=0`: External magnetic field
"""
@inline function energy_local(ising::IsingSquareLattice, idx::Union{Integer,CartesianIndex}, h::Real)::Real
    @inbounds ΔH = 2 * ising.σ[idx] * (sum(ising.σ[nn] for nn ∈ nearest_neighbors(ising, idx)) + h)
    return ΔH
end
@inline function energy_local(ising::IsingSquareLattice, idx::Union{Integer,CartesianIndex})::Real
    @inbounds ΔH = 2 * ising.σ[idx] * sum(ising.σ[nn] for nn ∈ nearest_neighbors(ising, idx))
    return ΔH
end

"""
    IsingGraph

Ising model on an arbitrary graph.

# Fields:
- `g::Graph`: Graph structure of the system
- `σ::Vector`: Vector containing the state of each node in the system
"""
mutable struct IsingGraph <: Ising

    "Graph structure of the system"
    g::Graph

    "State at each node"
    σ::Vector{Int8}

    """
        IsingGraph(g::Graph)

    Construct a new Ising system with graph structuer `g` and random initial states at each node.
    """
    IsingGraph(g::Graph) = new(g, rand(Int8[-1, 1], nv(g)))

    """
        IsingGraph(g::Graph, (::Val{+1} || ::Val{-1}))

    Construct a new Ising system with graph structuer `g` with all spins with same state.
    """
    IsingGraph(g::Graph, ::Val{+1}) = new(g, fill(Int8(+1), nv(g)))
    IsingGraph(g::Graph, ::Val{-1}) = new(g, fill(Int8(-1), nv(g)))
end

"""
    energy(ising::IsingGraph, h::Real=0)::Real

Total energy of an Ising system `ising` over a graph subject to external magnetic field `h`.

If no external magnetic field is provided it is assumed to be `h=0`.
"""
@inline energy(ising::IsingGraph)::Real = @inbounds -sum(ising.σ[src(e)] * ising.σ[dst(e)] for e ∈ edges(ising.g))


@inline energy(ising::IsingGraph, h::Real)::Real = @inbounds energy(ising) - h * magnet_total(ising)

"""
    nearest_neighbors(ising::IsingGraph, v::Integer)

For an Ising system over a graph `ising`, get the nearest neighobors of a given site `v`.
"""
@inline nearest_neighbors(ising::IsingGraph, v::Integer) = neighbors(ising.g, v)

"""
    energy_local(ising::IsingGraph, v::Integer, h::Real=0)::Real

Energy difference for an Ising system over a graph `ising` associated with a single flip of an spin at node `v`.
"""
@inline energy_local(ising::IsingGraph, v::Integer, h::Real)::Real = @inbounds 2 * ising.σ[v] * (sum(ising.σ[nn] for nn ∈ nearest_neighbors(ising, v)) + h)

@inline energy_local(ising::IsingGraph, v::Integer)::Real = @inbounds 2 * ising.σ[v] * sum(ising.σ[nn] for nn ∈ nearest_neighbors(ising, v))

end
