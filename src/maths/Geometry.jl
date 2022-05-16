module Geometry

export square_gird_iter

"""
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
    - `k`: Iteration number

# Returns:
    - Tuple of cartesian indices (given as fractions) of the points associated with the iteration `k`.
"""
function square_gird_iter(::Val{dim}, iter::Integer) where {dim}
    @assert iter >= 0
    if iter == 0
        # Just the endpoints
        Tuple.(CartesianIndices(ntuple(_ -> 0:1, Val(dim))))
    else
        L = 2^iter
        Tuple.(CartesianIndices(ntuple(_ -> 0:L, Val(dim)))) |> x -> filter(y -> any(isodd, y), x) .|> x -> x .// L
    end
end

end
