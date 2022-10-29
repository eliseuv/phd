using BenchmarkTools

BenchmarkTools.DEFAULT_PARAMETERS.overhead = BenchmarkTools.estimate_overhead()
BenchmarkTools.DEFAULT_PARAMETERS.samples = 100000
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 10

# Array
const dim = 3
const L = 512
arr = rand(Int8, ntuple(_ -> L, Val(dim)))

# Indices to be summed
const n_indices = 6
idx_tuple = ntuple(_ -> CartesianIndex(ntuple(_ -> rand(1:L), Val(dim))), Val(n_indices))
idx_arr = collect(idx_tuple)

function sum_iterate_over_collection(arr, idx)
    return sum(arr[i] for i ∈ idx)
end

function sum_array_constructed_using_collection(arr, idx)
    return sum([arr[i] for i ∈ idx])
end

function sum_array_with_array_of_idx(arr, idx::Array)
    return sum(arr[idx])
end

function sum_array_with_tuple_of_idx(arr, idx::Tuple)
    return sum(arr[collect(idx)])
end

println("\nSum using iteration over tuple")
@show sum_iterate_over_collection(arr, idx_tuple)
@btime sum_iterate_over_collection($arr, $idx_tuple)
println("\nSum using iteration over array")
@show sum_iterate_over_collection(arr, idx_arr)
@btime sum_iterate_over_collection($arr, $idx_arr)
println("\nSum array constructed using iteration over tuple")
@show sum_array_constructed_using_collection(arr, idx_tuple)
@btime sum_array_constructed_using_collection($arr, $idx_tuple)
println("\nSum array constructed using iteration over array")
@show sum_array_constructed_using_collection(arr, idx_arr)
@btime sum_array_constructed_using_collection($arr, $idx_arr)
println("\nSum using array of indices")
@show sum_array_with_array_of_idx(arr, idx_arr)
@btime sum_array_with_array_of_idx($arr, $idx_arr)
println("\nSum using array of indices")
@show sum_array_with_tuple_of_idx(arr, idx_tuple)
@btime sum_array_with_tuple_of_idx($arr, $idx_tuple)
