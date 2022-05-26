@doc raw"""
    Geometry

General
"""
module Geometry

export square_gird_iter,
    square_lattice_nearest_neighbors, square_lattice_nearest_neighbors_,
    square_lattice_nearest_neighbors_sum

"""
    square_gird_iter(::Val{N}, iter::Integer) where {N}

Fill a multidimensional unit domain with uniformly spaced points iteratively.
The coordinates of the points in this unit domain are given as a fraction

Example of a two dimensional grid iteration:
Points associated with current iteration are represented by `*` and those associated with previous iterations are represented by `o`.

    - 0th iteration

        .........           *.......*
        .........           .........
        .........     =>    .........
        .........           .........
        .........           *.......*

    - 1st iteration

        o.......o           o...*...o
        .........           .........
        .........     =>    *...*...*
        .........           .........
        o.......o           o...*...o

    - 2nd iteration

        o...o...o           o.*.o.*.o
        .........           *.*.*.*.*
        o...o...o     =>    o.*.o.*.o
        .........           *.*.*.*.*
        o...o...o           o.*.o.*.o

# Arguments:
    - `dim`: Dimensionality
    - `iter`: Iteration number

# Returns:
    - Tuple of cartesian indices (given as fractions) of the points associated with the iteration `iter`.
"""
function square_gird_iter(::Val{N}, iter::Integer) where {N}
    @assert iter >= 0
    if iter == 0
        # Just the endpoints
        Tuple.(CartesianIndices(ntuple(_ -> 0:1, Val(N))))
    else
        L = 2^iter
        Tuple.(CartesianIndices(ntuple(_ -> 0:L, Val(N)))) |> x -> filter(y -> any(isodd, y), x) .|> x -> x .// L
    end
end

"""
    square_lattice_nearest_neighbors_(lattice::Array{T,N}, idx::CartesianIndex{N}) where {T,N}

Get the cartesian coordinates of the nearest neighbours of a given site located at `idx` of a `N`-dimensional periodic square lattice `lattice`.

For a `N`-dimensional lattice each spin has 2`N` nearest neighbors.
"""
@inline function square_lattice_nearest_neighbors_(lattice::Array{T,N}, idx::CartesianIndex{N}) where {T,N}
    return @inbounds (ntuple(d -> CartesianIndex(ntuple(i -> i == d ? mod1(idx[i] + 1, size(lattice, i)) : idx[i], Val(N))), Val(N))...,
        ntuple(d -> CartesianIndex(ntuple(i -> i == d ? mod1(idx[i] - 1, size(lattice, i)) : idx[i], Val(N))), Val(N))...)
end

function square_lattice_nearest_neighbors_impl(lattice::Type{Array{T,N}}, idx::Type{CartesianIndex{N}}) where {T,N}
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
        :(($neighbor_prev, $neighbor_next)...)
    end
    # Return nearest neighbors for all dimensions
    :(tuple($(terms...)))
end

"""
    square_lattice_nearest_neighbors(lattice::Array{T,N}, idx::CartesianIndex{N}) where {T,N}

Get the cartesian coordinates of the nearest neighbours of a given site located at `idx` of a `N`-dimensional periodic square lattice `lattice`.

For a `N`-dimensional lattice each spin has 2`N` nearest neighbors.
"""
@generated function square_lattice_nearest_neighbors(lattice::Array{T,N}, idx::CartesianIndex{N}) where {T,N}
    square_lattice_nearest_neighbors_impl(lattice, idx)
end

function square_lattice_nearest_neighbors_sum_impl(lattice::Type{Array{T,N}}, idx::Type{CartesianIndex{N}}) where {T,N}
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
    :(+($(terms...)))
end

"""
    square_lattice_nearest_neighbors_sum(lattice::Array{T,N}, idx::CartesianIndex{N}) where {T,N}

Get the sum of the values of the nearest neighbours of a given site located at `idx` of a `N`-dimensional periodic square lattice `lattice`.
"""
@generated function square_lattice_nearest_neighbors_sum(lattice::Array{T,N}, idx::CartesianIndex{N}) where {T,N}
    square_lattice_nearest_neighbors_sum_impl(lattice, idx)
end

end
