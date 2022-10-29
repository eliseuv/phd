using BenchmarkTools

BenchmarkTools.DEFAULT_PARAMETERS.overhead = BenchmarkTools.estimate_overhead()
BenchmarkTools.DEFAULT_PARAMETERS.samples = 100000
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 10

include("../src/Geometry.jl")
using .Geometry

function square_lattice_nearest_neighbors_single_impl(lattice::Type{Array{T,N}}, idx::Type{CartesianIndex{N}}) where {T,N}
    # Loop on the dimensions
    terms = map(1:N) do d
        # Indices for both nearest neighbors in the current dimension `d`
        idx_prev_nn = :(mod1(idx[$d] - 1, size(lattice, $d)))
        idx_next_nn = :(mod1(idx[$d] + 1, size(lattice, $d)))
        # Fill indices for dimensions before and after the current one
        idx_before = [:(idx[$k]) for k in 1:d-1]
        idx_after = [:(idx[$k]) for k in d+1:N]
        # Neighbors for current dimension
        neighbor_prev = :(CartesianIndex($(idx_before...), $idx_prev_nn, $(idx_after...)))
        neighbor_next = :(CartesianIndex($(idx_before...), $idx_next_nn, $(idx_after...)))
        # Return neighbors for the current dimension `d`
        :(tuple($neighbor_prev, $neighbor_next))
    end
    # Return nearest neighbors for all dimensions
    return :(tuple($(terms...)))
end
@generated function square_lattice_nearest_neighbors_single(lattice::Array{T,N}, idx::CartesianIndex{N}) where {T,N}
    square_lattice_nearest_neighbors_single_impl(lattice, idx)
end

function square_lattice_nearest_neighbors_flat_single_impl(lattice::Type{Array{T,N}}, idx::Type{CartesianIndex{N}}) where {T,N}
    # Loop on the dimensions
    terms = map(1:N) do d
        # Indices for both nearest neighbors in the current dimension `d`
        idx_prev_nn = :(mod1(idx[$d] - 1, size(lattice, $d)))
        idx_next_nn = :(mod1(idx[$d] + 1, size(lattice, $d)))
        # Fill indices for dimensions before and after the current one
        idx_before = [:(idx[$k]) for k in 1:d-1]
        idx_after = [:(idx[$k]) for k in d+1:N]
        # Neighbors for current dimension
        neighbor_prev = :(CartesianIndex($(idx_before...), $idx_prev_nn, $(idx_after...)))
        neighbor_next = :(CartesianIndex($(idx_before...), $idx_next_nn, $(idx_after...)))
        # Return neighbors for the current dimension `d`
        :($neighbor_prev, $neighbor_next)
    end
    # Return nearest neighbors for all dimensions
    return :(tuple($(terms...)))
end
@generated function square_lattice_nearest_neighbors_flat_single(lattice::Array{T,N}, idx::CartesianIndex{N}) where {T,N}
    square_lattice_nearest_neighbors_flat_single_impl(lattice, idx)
end

function square_lattice_nearest_neighbors_sum_single_impl(lattice::Type{Array{T,N}}, idx::Type{CartesianIndex{N}}) where {T,N}
    # Loop on the dimensions
    terms = map(1:N) do d
        # Indices for both nearest neighbors in the current dimension
        idx_prev_nn = :(mod1(idx[$d] - 1, size(lattice, $d)))
        idx_next_nn = :(mod1(idx[$d] + 1, size(lattice, $d)))
        # Fill indices before and after the current dimension
        idx_before = [:(idx[$k]) for k in 1:d-1]
        idx_after = [:(idx[$k]) for k in d+1:N]
        # Term correspondig to dimension $d$
        :(lattice[$(idx_before...), $idx_prev_nn, $(idx_after...)] + lattice[$(idx_before...), $idx_next_nn, $(idx_after...)])
    end
    # Return sum of all terms
    return :(+($(terms...)))
end
@generated function square_lattice_nearest_neighbors_sum_single(lattice::Array{T,N}, idx::CartesianIndex{N}) where {T,N}
    square_lattice_nearest_neighbors_sum_single_impl(lattice, idx)
end

# System
const dim = 3
const L = 256
@show dim L

lattice = rand(Int8[-1, +1], ntuple(_ -> L, Val(dim)))

idx_first = CartesianIndex(ntuple(_ -> 1, Val(dim)))
idx_last = CartesianIndex(ntuple(_ -> L, Val(dim)))

println("\nNN Nested")
println("\nNested tuples")
@btime Geometry.square_lattice_nearest_neighbors_($lattice, $idx_first)
println("\nSingle genfunc")
@btime square_lattice_nearest_neighbors_single($lattice, $idx_first)
println("\nComposite genfunc")
@btime square_lattice_nearest_neighbors($lattice, $idx_first)

println("\nNN Flat")
println("\nFlat tuples")
@btime Geometry.square_lattice_nearest_neighbors_flat_($lattice, $idx_first)
println("\nSingle genfunc")
@btime square_lattice_nearest_neighbors_flat_single($lattice, $idx_first)
println("\nComposite genfunc")
@btime square_lattice_nearest_neighbors_flat($lattice, $idx_first)

println("\nNN Sum")
println("\nSum of nested tuples")
@btime sum(lattice[nn] for nn in tuple((Geometry.square_lattice_nearest_neighbors_($lattice, $idx_first)...)...))
println("\nSum of flat tuples")
@btime sum(lattice[nn] for nn in Geometry.square_lattice_nearest_neighbors_flat_($lattice, $idx_first))
println("\nSingle genfunc")
@btime square_lattice_nearest_neighbors_sum_single($lattice, $idx_first)
println("\nComposite genfunc")
@btime square_lattice_nearest_neighbors_sum($lattice, $idx_first)
