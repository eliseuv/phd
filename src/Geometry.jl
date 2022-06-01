@doc raw"""
    Geometry

General geometric utilities.
"""
module Geometry

export square_gird_iter,
    square_lattice_nearest_neighbors, square_lattice_nearest_neighbors_flat,
    square_lattice_nearest_neighbors_sum

@doc raw"""
    square_gird_iter(::Val{N}, iter::Integer) where {N}

Fill a `N`-dimensional unit domain with uniformly spaced points iteratively.
The coordinates of the points in this unit domain are given as a fraction.

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
    - `::Val{N}`: Dimensionality
    - `iter::Integer`: Iteration number

# Returns:
    - `::Vector{NTuple{N,Rational}}`: Array of cartesian indices (given as fractions) of the points associated with the iteration `iter`.
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

@inline function square_lattice_nearest_neighbors_(lattice::Array{T,N}, idx::CartesianIndex{N}) where {T,N}
    return @inbounds (ntuple(d -> CartesianIndex(ntuple(i -> i == d ? mod1(idx[i] + 1, size(lattice, i)) : idx[i], Val(N))), Val(N)),
        ntuple(d -> CartesianIndex(ntuple(i -> i == d ? mod1(idx[i] - 1, size(lattice, i)) : idx[i], Val(N))), Val(N)))
end
@inline function square_lattice_nearest_neighbors_flat_(lattice::Array{T,N}, idx::CartesianIndex{N}) where {T,N}
    return @inbounds (ntuple(d -> CartesianIndex(ntuple(i -> i == d ? mod1(idx[i] + 1, size(lattice, i)) : idx[i], Val(N))), Val(N))...,
        ntuple(d -> CartesianIndex(ntuple(i -> i == d ? mod1(idx[i] - 1, size(lattice, i)) : idx[i], Val(N))), Val(N))...)
end

@doc raw"""
    square_lattice_nearest_neighbors_exprs(lattice::Type{Array{T,N}}, idx::Type{CartesianIndex{N}}) where {T,N}

Returns the expressions for the nearest neighbors of site at `idx` in a `N`-dimensional preiodic square lattice `lattice`
in a nested tuple of the form `((nn_prev_1, nn_next_1),...,(nn_prev_N, nn_next_N))::NTuple{N,Tuple{Expr,Expr}}`.
"""
function square_lattice_nearest_neighbors_exprs(lattice::Type{Array{T,N}}, idx::Type{CartesianIndex{N}}) where {T,N}
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
        tuple(neighbor_prev, neighbor_next)
    end
    # Return nearest neighbors for all dimensions
    return tuple(terms...)
end

function square_lattice_nearest_neighbors_impl(lattice::Type{Array{T,N}}, idx::Type{CartesianIndex{N}}) where {T,N}
    # Get NN expressions
    nn_exprs = square_lattice_nearest_neighbors_exprs(lattice, idx)
    # Mult
    terms = map(nn_exprs) do nn_dim_exprs
        :(tuple($(nn_dim_exprs...)))
    end

    return :(tuple($(terms...)))
end
"""
    square_lattice_nearest_neighbors(lattice::Array{T,N}, idx::CartesianIndex{N}) where {T,N}

Get the cartesian coordinates of the nearest neighbours of a given site located at `idx` of a `N`-dimensional periodic square lattice `lattice`
in a nested tuple of the form `NTuple{N,NTuple{2,CartesianIndex{N}}}`.
"""
@generated function square_lattice_nearest_neighbors(lattice::Array{T,N}, idx::CartesianIndex{N}) where {T,N}
    square_lattice_nearest_neighbors_impl(lattice, idx)
end

function square_lattice_nearest_neighbors_flat_impl(lattice::Type{Array{T,N}}, idx::Type{CartesianIndex{N}}) where {T,N}
    # Get NN expressions
    nn_exprs = square_lattice_nearest_neighbors_exprs(lattice, idx)
    # Unpack NN expressions to flat tuple
    return :([$((nn_exprs...)...)])
end
"""
    square_lattice_nearest_neighbors_flat(lattice::Array{T,N}, idx::CartesianIndex{N}) where {T,N}

Get a vector of the cartesian coordinates of the nearest neighbours of a given site located at `idx` of a `N`-dimensional periodic square lattice `lattice`.
"""
@generated function square_lattice_nearest_neighbors_flat(lattice::Array{T,N}, idx::CartesianIndex{N}) where {T,N}
    square_lattice_nearest_neighbors_flat_impl(lattice, idx)
end

function square_lattice_nearest_neighbors_sum_impl(lattice::Type{Array{T,N}}, idx::Type{CartesianIndex{N}}) where {T,N}
    # Get NN expressions
    nn_exprs = square_lattice_nearest_neighbors_exprs(lattice, idx)
    # Unpack NN expressions to flat tuple of lattice site values
    nn_values = (:(lattice[$nn_expr]) for nn_expr in tuple((nn_exprs...)...))
    # Return sum of all these terms
    return :(+($(nn_values...)))
end
"""
    square_lattice_nearest_neighbors_sum(lattice::Array{T,N}, idx::CartesianIndex{N}) where {T,N}

Get the sum of the values of the nearest neighbours of a given site located at `idx` of a `N`-dimensional periodic square lattice `lattice`.
"""
@generated function square_lattice_nearest_neighbors_sum(lattice::Array{T,N}, idx::CartesianIndex{N}) where {T,N}
    square_lattice_nearest_neighbors_sum_impl(lattice, idx)
end

end
