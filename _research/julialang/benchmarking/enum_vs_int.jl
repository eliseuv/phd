@doc raw"""
    Benchmarking operations on arrays of integers vs. arrays of enums.

"""
using BenchmarkTools

BenchmarkTools.DEFAULT_PARAMETERS.overhead = BenchmarkTools.estimate_overhead()
BenchmarkTools.DEFAULT_PARAMETERS.samples = 100000
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 10

# System

SpinType = Int8

@enum SpinState::SpinType begin
    up = +1
    down = -1
end

const dim = 3
const L = 256

arr_enum = rand(instances(SpinState), ntuple(_ -> L, Val(dim)))
arr_int = Integer.(arr_enum)

# Spin flip

idx = CartesianIndex(rand(1:L, dim)...)

function flip!(arr::Array{T,N}, idx::CartesianIndex{N}) where {T<:Integer,N}
    arr[idx] = -arr[idx]
end

function flip!(arr::Array{SpinState,N}, idx::CartesianIndex{N}) where {N}
    arr[idx] = SpinState(-Integer(arr[idx]))
end

function flip_if!(arr::Array{SpinState,N}, idx::CartesianIndex{N}) where {N}
    if arr[idx] == up
        arr[idx] = down
    else
        arr[idx] = up
    end
end

println("\nSpin flip\n")

println("Flip array of int")
@btime flip!($arr_int, $idx)

println("Flip array of enum")
@btime flip!($arr_enum, $idx)

println("Flip array of enum using if statement")
@btime flip_if!($arr_enum, $idx)

# Sum whole array

Base.:+(x::SpinState, y::SpinState) = Integer(x) + Integer(y)
Base.:+(x::Integer, y::SpinState) = x + Integer(y)

sum_dot(arr::Array{SpinState}) = sum(Integer.(arr))

sum_f_itr(arr::Array{SpinState}) = sum(Integer, arr)

println("\nSum whole array\n")

@show sum(arr_int)
@btime sum($arr_int)

@show sum(arr_enum)
@btime sum($arr_enum)

@show sum_dot(arr_enum)
@btime sum_dot($arr_enum)

@show sum_f_itr(arr_enum)
@btime sum_f_itr($arr_enum)

# Sum certain indices

const n_indices = 6

idx_tuple = ntuple(_ -> CartesianIndex(ntuple(_ -> rand(1:L), Val(dim))), Val(n_indices))
idx_arr = collect(idx_tuple)

function sum_iterate_over_collection(arr::Array{<:Integer}, indices)
    return sum(arr[i] for i ∈ indices)
end

function sum_iterate_over_collection(arr::Array{SpinState}, indices)
    return sum(Integer, arr[i] for i ∈ indices)
end

println("\nSum certain indices\n")

@show sum_iterate_over_collection(arr_int, idx_tuple)
@btime sum_iterate_over_collection($arr_int, $idx_tuple)

@show sum_iterate_over_collection(arr_enum, idx_tuple)
@btime sum_iterate_over_collection($arr_enum, $idx_tuple)

@show sum_iterate_over_collection(arr_int, idx_arr)
@btime sum_iterate_over_collection($arr_int, $idx_arr)

@show sum_iterate_over_collection(arr_enum, idx_arr)
@btime sum_iterate_over_collection($arr_enum, $idx_arr)

# Sum of product

const indices_pair_count = 16

indices_pairs_tuple = ntuple(_ -> ntuple(_ -> CartesianIndex(ntuple(_ -> rand(1:L), Val(dim))), Val(2)), Val(indices_pair_count))
indices_pairs_arr = collect(indices_pairs_tuple)

Base.:*(x::SpinState, y::SpinState) = SpinState(Integer(x) * Integer(y))

function sum_prod(arr::Array{<:Integer}, indices_pairs)
    return sum(arr[idx_pair[1]] * arr[idx_pair[2]] for idx_pair ∈ indices_pairs)
end

function sum_prod_f_itr(arr::Array{SpinState}, indices_pairs)
    return sum(Integer, arr[idx_pair[1]] * arr[idx_pair[2]] for idx_pair ∈ indices_pairs)
end

function sum_prod_f_itr_2(arr::Array{SpinState}, indices_pairs)
    return sum(Integer(arr[idx_pair[1]]) * Integer(arr[idx_pair[2]]) for idx_pair ∈ indices_pairs)
end

println("\nSum of products\n")

@show sum_prod(arr_int, indices_pairs_tuple)
@btime sum_prod($arr_int, $indices_pairs_tuple)

@show sum_prod_f_itr(arr_enum, indices_pairs_tuple)
@btime sum_prod_f_itr($arr_enum, $indices_pairs_tuple)

@show sum_prod_f_itr_2(arr_enum, indices_pairs_tuple)
@btime sum_prod_f_itr_2($arr_enum, $indices_pairs_tuple)

@show sum_prod(arr_int, indices_pairs_arr)
@btime sum_prod($arr_int, $indices_pairs_arr)

@show sum_prod_f_itr(arr_enum, indices_pairs_arr)
@btime sum_prod_f_itr($arr_enum, $indices_pairs_arr)

@show sum_prod_f_itr_2(arr_enum, indices_pairs_arr)
@btime sum_prod_f_itr_2($arr_enum, $indices_pairs_arr)
