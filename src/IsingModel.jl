module IsingModel

export Ising, IsingMeanField, IsingSquareLattice, IsingGraph,
    SpinHalfState, down, up,
    SpinOneState, down, zero, up,
    ISING_SQ_LAT_2D_BETA_CRIT, ising_square_lattice_2d_beta_critical,
    set_state!, randomize_state!,
    magnet_total, magnet, magnet_moment,
    magnet_total_local, energy_local,
    flip!,
    energy,
    nearest_neighbors,
    metropolis!, metropolis_and_measure_total_magnet!, metropolis_and_measure_energy!

using Random, Distributions, Graphs

include("Metaprogramming.jl")
include("Geometry.jl")

using .Metaprogramming

"""
    SpinState

Supertype for all spin states.
"""
abstract type SpinState end

"""
    convert(::Type{T}, σ::SpinState) where {T<:Number}

Use the integer representation of `σ::SpinState` in order to convert it to a numerical type `T<:Number`.
"""
@inline Base.convert(::Type{T}, σ::SpinState) where {T<:Number} = T(Integer(σ))

"""
    promote_rule(T::Type, ::Type{SpinState})

Always promote the `SpinState` to whatever the other type is.
"""
@inline Base.promote_rule(T::Type, ::Type{SpinState}) = T

# Arithmetic with numbers and Spin States
for op in (:*, :/, :+, :-)
    @eval begin
        @inline Base.$op(x::Number, σ::SpinState) = $op(promote(x, σ)...)
        @inline Base.$op(σ::SpinState, y::Number) = $op(promote(σ, y)...)
    end
end

"""
    *(σ₁::SpinState, σ₂::SpinState)

Multiplication of spin states.
"""
@inline Base.:*(σ₁::SpinState, σ₂::SpinState) = Integer(σ₁) * Integer(σ₂)

"""
    SpinHalfState::Int8 <: SpinState

Enumeration of possible spin `1/2` values.
"""
@enum SpinHalfState::Int8 <: SpinState begin
    down = -1
    up = +1
end

"""
    SpinOneState::Int8 <: SpinState

Enumeration of possible spin `1` values.
"""
@enum SpinOneState::Int8 <: SpinState begin
    down = -1
    zero = 0
    up = +1
end

"""
    show(io::IO, ::MIME"text/plain", σ::SpinHalfState)

Text representation of `SpinHalfState`.
"""
function Base.show(io::IO, ::MIME"text/plain", σ::SpinHalfState)
    spin_char = σ == up ? '↑' : '↓'
    print(io, spin_char)
end

"""
    show(io::IO, ::MIME"text/plain", σ::SpinOneState)

Text representation of `SpinOneState`.
"""
function Base.show(io::IO, ::MIME"text/plain", σ::SpinOneState)
    spin_char = σ == up ? '↑' : σ == down ? '↓' : '-'
    print(io, spin_char)
end

@doc raw"""
    IsingMeanField

Ising system with mean field interaction:
Every spin interacts equally with every other spin.

Since in the mean field model there is no concept of space and locality,
we represent the state of the system simply by total number of spin up and spin down sites.

An `AbstractVector{SpinHalfState}` interface for the `IsingMeanField` type can be implemented
if we assume that the all spin states are stored in a sorted vector with ``N = N₊ + N₋`` elements:

    σ = (↑, ↑, …, ↑, ↓, ↓, …, ↓)
        |--- N₊ ---||--- N₋ ---|
        |---------- N ---------|

Therefore, for an `ising::IsingMeanField` we can access the `i`-th spin `σᵢ = ising[i]`:
If `i ≤ N₊` then `σᵢ = ↑` else (`N₊ < i ≤ N`) `σᵢ = ↓`.

# Fields:
- `state::NamedTuple{(:up, :down),NTuple{2,Int64}}`: State of the system given by the number of spins in each state.
"""
mutable struct IsingMeanField <: AbstractVector{SpinHalfState}

    "State of the system"
    state::NamedTuple{(:up, :down),NTuple{2,Int64}}

    @doc raw"""
        IsingMeanField(; up::Int64, down::Int64)

    Construct an Ising system with mean field interaction with a given number of spins in each state.
    """
    IsingMeanField(; up::Integer=0, down::Integer=0) = new((up=up, down=down))

    @doc raw"""
        IsingMeanField(N::Integer, σ₀::SpinHalfState)

    Construct an Ising system with mean field interaction with `N` spins, all in a given initial state `σ₀`.
    """
    function IsingMeanField(N::Integer, σ₀::SpinHalfState)
        return σ₀ == up ? new((up=N, down=0)) : new((up=0, down=N))
    end

    @doc raw"""
        IsingMeanField(N::Integer, ::Val{:rand})

    Construct an Ising system with mean field interaction with `N` spins in a random initial state.
    """
    function IsingMeanField(N::Integer, ::Val{:rand})
        N₊ = rand(1:N)
        N₋ = N - N₊
        return new((up=N₊, down=N₋))
    end
end

@doc raw"""
    length(ising::IsingMeanField)

Total number of spins (`N`) in an Ising system with mean field interaction `ising`.
"""
Base.length(ising::IsingMeanField) = sum(ising.state)

@doc raw"""
    IndexStyle(::IsingMeanField)

Use only linear indices for the `AbstractVector{SpinHalfState}` interface for the `IsingMeanField` type.
"""
@inline Base.IndexStyle(::Type{<:IsingMeanField}) = IndexLinear()

@doc raw"""
    getindex(ising::IsingMeanField, i::Integer)

Get the state of the `i`-th spin in the Ising system with mean field interaction `ising`.
"""
@inline Base.getindex(ising::IsingMeanField, i::Integer) = i <= ising.state.up ? up : down

@doc raw"""
    setindex!(ising::IsingMeanField, σ::SpinHalfState, i::Integer)

Set the state of the `i`-th spin to `σ` in the Ising system with mean field interaction `ising`.
"""
@inline function Base.setindex!(ising::IsingMeanField, σ::SpinHalfState, i::Integer)
    if i <= ising.state.up && σ == down
        ising.state = (up=ising.state.up - 1, down=ising.state.down + 1)
    elseif σ == up
        ising.state = (up=ising.state.up + 1, down=ising.state.down - 1)
    end
end

@doc raw"""
    firstindex(ising::IsingMeanField)

The first spin in the `AbstractVector{SpinHalfState}` interface of `IsingMeanField`.
"""
@inline Base.firstindex(ising::IsingMeanField) = 1

@doc raw"""
    lastindex(ising::IsingMeanField)

The last spin in the `AbstractVector{SpinHalfState}` interface of `IsingMeanField`.
"""
@inline Base.lastindex(ising::IsingMeanField) = sum(ising.state)

"""
    set_state!(ca::IsingMeanField, σ₀::SpinHalfState)

Set the state of all sites of an Ising system `ising` to a given site state `σ₀`.
"""
@inline function set_state!(ising::IsingMeanField, σ₀::SpinHalfState)
    N = length(ising)
    ising.state = if σ₀ == up
        (up=N, down=0)
    else
        (up=0, down=N)
    end
end

"""
    randomize_state!(ising::IsingMeanField, p::Real=0.5)

Set the state of each site of an Ising system `ising` to a random state `σ ∈ {↑, ↓}` with a probability `p` of being `↑`.
"""
@inline function randomize_state!(ising::IsingMeanField, p::Real)
    N = length(ising)
    dist = Binomial(N, p)
    N₊ = rand(dist)
    N₋ = N - N₊
    ising.state = (up=N₊, down=N₋)
end

@inline function randomize_state!(ising::IsingMeanField)
    N = length(ising)
    N₊ = rand(1:N)
    N₋ = N - N₊
    ising.state = (up=N₊, down=N₋)
end

@doc raw"""
    flip!(ising::IsingMeanField, i::Integer)

Flip the state of the `i`-th spin in the Ising system with mean field interaction `ising`.
"""
@inline function flip!(ising::IsingMeanField, i::Integer)
    σᵢ = Integer(ising[i])
    N₊, N₋ = ising.state.up, ising.state.down
    ising.state = (up=N₊ - σᵢ, down=N₋ + σᵢ)
end

@doc raw"""
    magnet_total(ising::IsingMeanField)

Total magnetization of an Ising.

    ``M = N₊ - N₋``
"""
@inline magnet_total(ising::IsingMeanField) = ising.state.up - ising.state.down

@doc raw"""
    energy(ising::IsingMeanField, h::Real=0)

Total magnetization of an Ising system with mean field interaction.

    ``H = \frac{N - M^2}{2} + Mh``
"""
@inline energy(ising::IsingMeanField) = Integer((length(ising) - magnet_total(ising)^2) / 2)

@inline energy(ising::IsingMeanField, h::Real) = energy(ising) + h * magnet_total(ising)

@doc raw"""
    magnet_total_local(ising::IsingMeanField, i::Integer)

Change in local magnetization of an Ising system with mean field interaction if the `i`-th were to be flipped.
"""
@inline magnet_total_local(ising::IsingMeanField, i::Integer) = @inbounds -2 * Integer(ising[i])

@doc raw"""
    nearest_neighbors_sum(ising::IsingMeanField, i::Integer)

Sum of the values of the nearest neighbors of the `i`-th spin in the Ising mean field system `ising`.
"""
@inline nearest_neighbors_sum(ising::IsingMeanField, i::Integer) = @inbounds magnet_total(ising) - Integer(ising[i])

@doc raw"""
    energy_local(ising::IsingMeanField, i::Integer, h::Real=0)

Change in energy of an Ising system with mean field interaction if the `i`-th were to be flipped.
"""
@inline energy_local(ising::IsingMeanField, i::Integer, h::Real) = @inbounds 2 * Integer(ising[i]) * (nearest_neighbors_sum(ising, i) + h)

@inline energy_local(ising::IsingMeanField, i::Integer) = @inbounds 2 * Integer(ising[i]) * nearest_neighbors_sum(ising, i)

"""
    IsingConcrete{N} <: AbstractArray{SpinHalfState,N}

Supertype for all Ising systems that have a concrete representation of its state in memory
in the form of a concrete array member `state::Array{SpinHalfState,N}`.

The whole indexing interface of the `state::Array{SpinHalfState,N}` can be passed to the `::IsingConcrete{N}` object itself.
"""
abstract type IsingConcrete{N} <: AbstractArray{SpinHalfState,N} end

"""
    length(ising::IsingConcrete)

Total number of sites of an Ising system `ising`.
"""
@inline Base.length(ising::IsingConcrete) = length(ising.state)

"""
    size(ising::IsingConcrete)

Size of the state of an Ising system `ising`.
"""
@inline Base.size(ising::IsingConcrete) = size(ising.state)

"""
    size(ising::IsingConcrete, dim)

Size of the state of an Ising system `ising` along a given dimension `dim`.
"""
@inline Base.size(ising::IsingConcrete, dim) = size(ising.state, dim)

"""
    IndexStyle(::Type{<:IsingConcrete{N}}) where {N}

Use same indexing style used to index the state array.
"""
@inline Base.IndexStyle(::Type{<:IsingConcrete{N}}) where {N} = IndexStyle(Array{SpinHalfState,N})

"""
    getindex(ising::IsingConcrete, inds...)

Index the Ising system itself to access its state.
"""
@inline Base.getindex(ising::IsingConcrete, inds...) = getindex(ising.state, inds...)

"""
    setindex!(ising::IsingConcrete, σ, inds...)

Set the state of a given spin at site `i` to `σ` in the Ising system `ising`.
"""
@inline Base.setindex!(ising::IsingConcrete, σ, inds...) = setindex!(ising.state, σ, inds...)

"""
    firstindex(ising::IsingConcrete)

Get the index of the first spin in the system.
"""
@inline Base.firstindex(ising::IsingConcrete) = firstindex(ising.state)

"""
    lastindex(ising::IsingConcrete)

Get the index of the last spin in the system.
"""
@inline Base.lastindex(ising::IsingConcrete) = lastindex(ising.state)

"""
    set_state!(ca::IsingConcrete, σ₀::SpinHalfState)

Set the state of all sites of an Ising system `ising` to a given site state `σ₀`.
"""
@inline function set_state!(ising::IsingConcrete, σ₀::SpinHalfState)
    fill!(ising, σ₀)
end

"""
    randomize_state!(ising::IsingConcrete, p::Real=0.5)

Set the state of each site of an Ising system `ising` to a random state `σ₀ ∈ {↑, ↓}` with probability of `p` of being `↑`.
"""
@inline function randomize_state!(ising::IsingConcrete, p::Real)
    dist = Distributions.Bernoulli(p)
    for idx ∈ eachindex(ising)
        ising[idx] = rand(dist) ? up : down
    end
end

@inline function randomize_state!(ising::IsingConcrete)
    rand!(ising, instances(SpinHalfState))
end

@doc raw"""
    magnet_total(ising::IsingConcrete)

Total magnetization of an Ising system `ising`.

The total magnetization is defined as the sum of all site states:

``M = ∑ᵢ σᵢ``

See also: [`magnet`](@ref), [`magnet_moment`](@ref).
"""
@inline magnet_total(ising::IsingConcrete) = @inbounds sum(Integer, ising.state)

@doc raw"""
    magnet(ising::IsingConcrete)

Magnetization per site of an Ising system `ising`.

``m = M / N = ∑ᵢ σᵢ / N``

See also: [`magnet_total`](@ref), [`magnet_moment`](@ref).
"""
@inline magnet(ising::IsingConcrete) = magnet_total(ising) / length(ising)

@doc raw"""
    magnet_moment(ising::IsingConcrete, k::integer)

Calculates the k-th momentum of the magnetization of an Ising system `ising`.

``mᵏ = 1/nᵏ (∑ᵢ σᵢ)ᵏ``

See also: [`magnet`](@ref), [`magnet_total`](@ref).
"""
@inline magnet_moment(ising::IsingConcrete, k::Integer) = magnet(ising)^k

"""
    magnet_total_local(ising::IsingConcrete{N}, i::Union{Integer,CartesianIndex{N}}) where {N}

The difference in total magnetization of an Ising system `ising` if the `i`-th spin were to be flipped.
"""
@inline magnet_total_local(ising::IsingConcrete{N}, i::Union{Integer,CartesianIndex{N}}) where {N} = @inbounds -2 * Integer(ising[i])

@doc raw"""
    nearest_neighbors_sum(ising::IsingConcrete{N}, i::Union{Integer,CartesianIndex{N}})

Sum of the values of the nearest neighbors of the `i`-th spin in the Ising system `ising`.
"""
@inline nearest_neighbors_sum(ising::IsingConcrete{N}, i::Union{Integer,CartesianIndex{N}}) where {N} = @inbounds sum(Integer, ising[nn] for nn ∈ nearest_neighbors(ising, i))

"""
    energy_local(ising::IsingConcrete{N}, i::Union{Integer,CartesianIndex{N}}, h::Real=0) where {N}

Energy difference for an Ising system `ising` associated with the flip of the `i`-th spin
and subject to external magnetic field `h`.

If no external magnetic field is provided, it is assumed to be `h=0`.

This is the default implementation for any specific type of Ising model `IsingSpecific <: IsingConcrete` that provides an implementation of `nearest_neighbors(ising::IsingSpecific, i::Union{Integer,CartesianIndex{N}})`.
"""
@inline energy_local(ising::IsingConcrete{N}, i::Union{Integer,CartesianIndex{N}}, h::Real) where {N} = @inbounds 2 * Integer(ising[i]) * (nearest_neighbors_sum(ising, i) + h)

@inline energy_local(ising::IsingConcrete{N}, i::Union{Integer,CartesianIndex{N}}) where {N} = @inbounds 2 * Integer(ising[i]) * nearest_neighbors_sum(ising, i)

"""
    flip!(ising::IsingConcrete{N}, i::Union{Integer,CartesianIndex{N}}) where {N}

Flips the `i`-th spin on the Ising system `ising`.
"""
@inline function flip!(ising::IsingConcrete{N}, i::Union{Integer,CartesianIndex{N}}) where {N}
    @inbounds ising[i] = SpinHalfState(-Integer(ising[i]))
end

"""
    IsingSquareLattice{N} <: IsingConcrete{N}

Ising system on a `N`-dimensional periodic square lattice with nearest neighbor interaction.

# Fields:
- `state::Array{SpinHalfState,N}`: State of the system
"""
mutable struct IsingSquareLattice{N} <: IsingConcrete{N}

    "State of the Ising system"
    state::Array{SpinHalfState,N}

    """
        IsingSquareLattice(size::NTuple{N,Integer}, σ₀::SpinHalfState) where {N}

    Construct a new Ising system in a multidimensional square lattice of dimensions provided by `size`,
    with nearest neighbor interaction and with all spins with same initial state `σ₀`.
    """
    IsingSquareLattice(size::NTuple{N,Integer}, σ₀::SpinHalfState) where {N} = new{N}(fill(σ₀, size))

    """
        IsingSquareLattice(size::NTuple{N,Integer}, ::Val{:rand}) where {N}

    Construct a new Ising system in a multidimensional square lattice of dimensions provided by `size`,
    with nearest neighbor interaction and with spins in random states.
    """
    IsingSquareLattice(size::NTuple{N,Integer}, ::Val{:rand}) where {N} = new{N}(rand(instances(SpinHalfState), size))

    @doc raw"""
        IsingSquareLattice(::Val{N}, L::Integer, σ₀::BrassState) where {N}

    Construct a `dim`-dimensional square Ising system of side length `L` and a given initial state `σ₀`.
    """
    IsingSquareLattice(::Val{N}, L::Integer, σ₀::SpinHalfState) where {N} = IsingSquareLattice(ntuple(_ -> L, Val(N)), σ₀)

    @doc raw"""
        IsingSquareLattice(::Val{N}, L::Integer, ::Val{:rand}) where {N}

    Construct a `dim`-dimensional square Ising system of side length `L` and random initial state.
    """
    IsingSquareLattice(::Val{N}, L::Integer, ::Val{:rand}) where {N} = IsingSquareLattice(ntuple(_ -> L, Val(N)), Val(:rand))
end

@doc raw"""
    ISING_SQ_LAT_2D_BETA_CRIT

Critical temperature for the Ising system on a 2D square lattice.

``β_c = \frac{\log{(1 + √{2})}}{2}``
"""
const ISING_SQ_LAT_2D_BETA_CRIT = 0.5 * log1p(sqrt(2))

@doc raw"""
    ising_square_lattice_2d_beta_critical(τ::Real)

Returns the critical β ``\beta_C`` for a given value of ``\tau = \frac{T}{T_C}``.
"""
ising_square_lattice_2d_beta_critical(τ::Real) = ISING_SQ_LAT_2D_BETA_CRIT / τ

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
    H = zero(Int64)
    # Loop on dimensions
    @inbounds for d ∈ 1:N
        # Bulk
        front_bulk = selectdim(ising, d, 1:(size(ising, d)-1))
        back_bulk = selectdim(ising, d, 2:size(ising, d))
        H -= sum(front_bulk .* back_bulk)
        # Periodic boundaries
        last_slice = selectdim(ising, d, size(ising, d))
        first_slice = selectdim(ising, d, 1)
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
@inline nearest_neighbors(ising::IsingSquareLattice{N}, idx::CartesianIndex{N}) where {N} = @inbounds Geometry.square_lattice_nearest_neighbors_flat(ising, idx)

"""
    nearest_neighbors(ising::IsingSquareLattice, i::Integer)

Get the cartesian coordinates of the nearest neighbours of the `i`-th spin
in a multidimensional square lattice `ising`.

For a `N`-dimensional lattice each spin has 2`N` nearest neighbors.
"""
@inline function nearest_neighbors(ising::IsingSquareLattice, i::Integer)
    @inbounds idx = CartesianIndices(ising)[i]
    return nearest_neighbors(ising, idx)
end

@doc raw"""
    nearest_neighbors_sum(ising::IsingSquareLattice{N}, idx::CartesianIndex{N}) where {N}

Sum of the values of the nearest neighbors of the spin at `idx` in the Ising system `ising`.
"""
@inline nearest_neighbors_sum(ising::IsingSquareLattice{N}, idx::CartesianIndex{N}) where {N} = @inbounds Geometry.square_lattice_nearest_neighbors_sum(ising, idx)

@doc raw"""
    energy_local(ising::IsingSquareLattice{N}, idx::CartesianIndex{N}, h::Real=0) where {N}

Energy difference for an Ising system in a multidimensional square lattice with nearest neighbor interaction `ising`
associated with a single spin flip at site `idx` subject to external magnetic field `h`.

If no external magnetic field is provided, it is assumed `h=0`.

# Arguments:
- `ising::IsingSquareLattice`: Ising system
- `idx::CartesianIndex`: Spin site
- `h::Real=0`: External magnetic field
"""
@inline energy_local(ising::IsingSquareLattice{N}, idx::CartesianIndex{N}, h::Real) where {N} = @inbounds 2 * Integer(ising[idx]) * (nearest_neighbors_sum(ising, idx) + h)

@inline energy_local(ising::IsingSquareLattice{N}, idx::CartesianIndex{N}) where {N} = @inbounds 2 * Integer(ising[idx]) * nearest_neighbors_sum(ising, idx)

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
@inline function energy_local(ising::IsingSquareLattice, i::Integer, h::Real)
    @inbounds idx = CartesianIndices(ising)[i]
    return energy_local(ising, idx, h)
end
@inline function energy_local(ising::IsingSquareLattice, i::Integer)
    @inbounds idx = CartesianIndices(ising)[i]
    return energy_local(ising, idx)
end

"""
    show(io::IO, ::MIME"mime", ising::IsingSquareLattice{N}) where {N}

Plain text representation of the state of an Ising system to be used in the REPL.
"""
function Base.show(io::IO, ::MIME"text/plain", ising::IsingSquareLattice{N}) where {N}
    # Get output from printing state
    io_temp = IOBuffer()
    show(IOContext(io_temp, :limit => true), "text/plain", Integer.(ising.state))
    str = String(take!(io_temp))
    # Modify output
    # Remove type info
    #str = replace(str, r".+\:" => "")
    # Use symbols instead of numbers
    str = replace(str, "-1" => " ↓", "1" => "↑")
    # Fix horizontal spacing
    str = replace(str, "  " => " ")
    str = replace(str, "⋮" => "   ⋮", "⋱" => "   ⋱", " …  " => " … ")
    # Output final result
    print(io, str)
end

"""
    IsingGraph

Ising model on an arbitrary graph.

# Fields:
- `g::Graph`: Graph structure of the system
- `state::Vector`: Vector containing the state of each node in the system
"""
mutable struct IsingGraph <: IsingConcrete{1}

    "Graph structure of the system"
    g::Graph

    "State at each node"
    state::Vector{SpinHalfState}

    """
        IsingGraph(g::Graph, σ₀::SpinHalfState)

    Construct a new Ising system with graph structure `g` with all spins with same initial state `σ₀`.
    """
    IsingGraph(g::Graph, σ₀::SpinHalfState) = new(g, fill(σ₀, nv(g)))

    """
        IsingGraph(g::Graph, ::Val{:rand})

    Construct a new Ising system with graph structure `g` and random initial states at each node.
    """
    IsingGraph(g::Graph, ::Val{:rand}) = new(g, rand(instances(SpinHalfState), nv(g)))
end

"""
    energy(ising::IsingGraph, h::Real=0)

Total energy of an Ising system `ising` over a graph subject to external magnetic field `h`.

If no external magnetic field is provided it is assumed to be `h=0`.
"""
@inline energy(ising::IsingGraph) = @inbounds -sum(Integer, ising[src(e)] * ising[dst(e)] for e ∈ edges(ising.g))


@inline energy(ising::IsingGraph, h::Real) = @inbounds energy(ising) - h * magnet_total(ising)

"""
    nearest_neighbors(ising::IsingGraph, i::Integer)

For an Ising system over a graph `ising`, get the nearest neighobors of a given site `i`.
"""
@inline nearest_neighbors(ising::IsingGraph, i::Integer) = neighbors(ising.g, i)

"""
    Ising

Supertype for all Ising systems.
"""
Ising = Union{IsingMeanField,IsingConcrete}

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
        i = rand(eachindex(ising))
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
        i = rand(eachindex(ising))
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
        # Loop on site
        for s in rand(eachindex(ising), length(ising))
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
        i = rand(eachindex(ising))
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

function metropolis_and_measure_total_magnet_2!(ising::Ising, β::Real, n_steps::Integer)
    # Vector to store results
    M_T = Vector{Int64}(undef, n_steps + 1)
    # Initial magnetization
    M_T[1] = magnet_total(ising)
    N = length(ising)
    # Sampling loop
    @inbounds for t ∈ 1:n_steps
        for i ∈ rand(1:N, N)
            # Get energy difference
            ΔH = energy_local(ising, i)
            # Metropolis prescription
            if ΔH < 0 || exp(-β * ΔH) > rand()
                # Flip spin
                flip!(ising, i)
            end
        end
        M_T[t+1] = magnet_total(ising)
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
        i = rand(eachindex(ising))
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
        i = rand(eachindex(ising))
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
