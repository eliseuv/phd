using BenchmarkTools

BenchmarkTools.DEFAULT_PARAMETERS.overhead = BenchmarkTools.estimate_overhead()
BenchmarkTools.DEFAULT_PARAMETERS.samples = 100000
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 10

# Array
const dim = 3
const L = 512
arr = rand(Int8, ntuple(_ -> L, Val(dim)))

# Indices to be summed
const n_indices = 128
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

const N = 10000
#vec = rand(Float64, N) .- 0.5
vec = rand(Int8, N)

function pairwise_product_sum_naive(vec)
    return -sum(Int64(vec[i]) * vec[j] for i ∈ eachindex(vec) for j in (i+1):length(vec))
end

function pairwise_product_sum_1(vec)
    return -sum(vec[i] * sum(vec[begin:(i-1)]) for i ∈ eachindex(vec))
end

function pairwise_product_sum_1_1(vec)
    return -sum(vec[i] * sum(vec[begin:(i-1)]) for i ∈ 2:length(vec))
end

function pairwise_product_sum_1_2(vec)
    return -sum(vec[i] * sum(vec[(i+1):end]) for i ∈ eachindex(vec))
end

function pairwise_product_sum_2(vec)
    return -sum(vec[i] * sum(vec[j] for j ∈ 1:(i-1)) for i ∈ 2:length(vec))
end

@show pairwise_product_sum_naive(vec)
@btime pairwise_product_sum_naive($vec)
@show pairwise_product_sum_1(vec)
@btime pairwise_product_sum_1($vec)
@show pairwise_product_sum_1_1(vec)
@btime pairwise_product_sum_1_1($vec)
@show pairwise_product_sum_1_2(vec)
@btime pairwise_product_sum_1_2($vec)
@show pairwise_product_sum_2(vec)
@btime pairwise_product_sum_2($vec)
