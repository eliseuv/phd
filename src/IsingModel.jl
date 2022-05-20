module IsingModel

using Folds, Graphs

export Ising, IsingSquareLattice, IsingCompleteGraph, IsingGraph, magnet_total, magnet, ising_square_lattice_2d_βc, energy_naive, energy, flip!, nearest_neighbors, energy_local, metropolis!

"""
Ising supertype
"""
abstract type Ising end

"""
Ising state size
"""
@inline Base.size(ising::Ising) = size(ising.σ)
@inline Base.size(ising::Ising, d::Integer) = size(ising.σ, d)
@inline Base.length(ising::Ising) = length(ising.σ)

"""
Single spin flip
"""
@inline function flip!(ising::Ising, idx::Integer)
    @inbounds ising.σ[idx] = -ising.σ[idx]
end

"""
Total magnetization:
    M = ∑_i sᵢ
"""
@inline magnet_total(ising::Ising)::Integer = @inbounds Folds.sum(ising.σ)

"""
Magnetization per spin:
    m = ∑_i sᵢ / N
"""
@inline magnet(ising::Ising)::Real = @inbounds magnet_total(ising) / length(ising)

"""
Metropolis sampling
"""
function metropolis!(ising::Ising, β::Real, n_steps::Integer; J::Real = 1, h::Real = 0)
    # Sampling loop
    @inbounds for _ in 1:n_steps
        # Select random spin
        idx = rand(1:length(ising))
        # Get energy difference
        ΔH = energy_local(ising, idx, J = J, h = h)
        # Metropolis prescription
        (ΔH < 0 || exp(-β * ΔH) > rand()) && flip!(ising, idx)
    end
end

"""
Ising model on a multidimensional (N dimensions) square lattice
"""
mutable struct IsingSquareLattice{N} <: Ising
    # State of the system
    σ::Array{Int8,N}
    # Construct with spins in random states
    IsingSquareLattice(S::NTuple{N}) where {N} = new{N}(rand([-1, 1], S))
    # Construct with all spins with same state
    IsingSquareLattice(S::NTuple{N}, ::Val{1}) where {N} = new{N}(fill(1, S))
    IsingSquareLattice(S::NTuple{N}, ::Val{-1}) where {N} = new{N}(fill(-1, S))
end

# Critical temperature
@inline ising_square_lattice_2d_βc(J::Real = 1) = log1p(sqrt(2)) / (2 * J)

"""
Total energy

Given by the Hamiltonian:
    H = - J ∑_⟨i,j⟩ sᵢ sⱼ - h ∑_i sᵢ
where ⟨i,j⟩ means that i and j are nearest neighbors.

Unless explicitly stated, it is assumed J=1 and h=0.
"""
function energy(ising::IsingSquareLattice{N}; J::Real = 1, h::Real = 0)::Real where {N}
    # Total energy
    H::Real = 0
    # Interaction energy
    S = size(ising.σ)
    @inbounds for d in 1:N
        # Bulk
        front_bulk = selectdim(ising.σ, d, 1:(S[d]-1))
        back_bulk = selectdim(ising.σ, d, 2:S[d])
        H -= Folds.sum(front_bulk .* back_bulk)
        # Periodic boundaries
        first_slice = selectdim(ising.σ, d, 1)
        last_slice = selectdim(ising.σ, d, S[d])
        H -= Folds.sum(last_slice .* first_slice)
    end
    H *= J
    # External field
    if h != 0
        @inbounds H -= h * magnet_total(ising)
    end
    return H
end

"""
Single spin flip
"""
@inline function flip!(ising::IsingSquareLattice, idx::CartesianIndex)
    ising.σ[idx] = -ising.σ[idx]
end

"""
Get the cartesian coordinates of the nearest neighbours to a given spin in a multidimensional square lattice

    For a N-dimensional lattice each spin has 2N nearest neighbors.
"""
@inline function nearest_neighbors(ising::IsingSquareLattice{N}, idx::CartesianIndex)::NTuple{2 * N,CartesianIndex{N}} where {N}
    S = size(ising.σ)
    (ntuple(d -> CartesianIndex(ntuple(i -> i == d ? mod1(idx[i] + 1, S[i]) : idx[i], Val(N))), Val(N))..., ntuple(d -> CartesianIndex(ntuple(i -> i == d ? mod1(idx[i] - 1, S[i]) : idx[i], Val(N))), Val(N))...)
    #ntuple(d -> CartesianIndex(ntuple(i -> i == d ? mod1(idx[i] + 1, S[i]) : idx[i], Val(D))), Val(D))
end

@inline function nearest_neighbors(ising::IsingSquareLattice, idx::Integer)
    @inbounds pos = CartesianIndices(ising.σ)[idx]
    nearest_neighbors(ising, pos)
end

"""
Energy difference associated with a single spin flip

    ΔH = 2 sᵢ (J ∑_<j> sⱼ + h)
where the sum ∑_<j> is over the nearest neighbors of i.
"""
@inline function energy_local(ising::IsingSquareLattice, idx::Integer; J::Real = 1, h::Real = 0)::Real
    @inbounds pos = CartesianIndices(ising.σ)[idx]
    @inbounds ΔH = 2 * ising.σ[idx] * (J * sum(ising.σ[nn] for nn ∈ nearest_neighbors(ising, pos)) + h)
    return ΔH
end

"""
Ising model on a complete graph
"""
mutable struct IsingCompleteGraph <: Ising
    # State at each node
    σ::Vector{Int8}
    # Construct system given a graph with random initial states
    IsingCompleteGraph(::Val{N}) where {N} = new(rand(Int8[-1, 1], Val(N)))
    # Construct system over a given graph with all spins with same state
    IsingCompleteGraph(::Val{N}, ::Val{1}) where {N} = new(fill(1, N))
    IsingCompleteGraph(::Val{N}, ::Val{-1}) where {N} = new(fill(-1, N))
end

"""
Total energy of the system
"""
function energy(ising::IsingCompleteGraph; J::Real = 1, h::Real = 0)::Real
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
@inline function nearest_neighbors(ising::IsingCompleteGraph, v::Integer)
    union(1:v-1, v+1:length(ising))
end

"""
Energy difference associated with a single spin flip

    ΔH = 2 sᵢ (J ∑_<j> sⱼ + h)
where the sum ∑_<j> is over the nearest neighbors of i.
"""
@inline function energy_local(ising::IsingCompleteGraph, v::Integer; J::Real = 1, h::Real = 0)::Real
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
