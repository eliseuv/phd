using BenchmarkTools

BenchmarkTools.DEFAULT_PARAMETERS.overhead = BenchmarkTools.estimate_overhead()
BenchmarkTools.DEFAULT_PARAMETERS.samples = 100000
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 10

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

# @show pairwise_product_sum_naive(vec)
# @btime pairwise_product_sum_naive($vec)
# @show pairwise_product_sum_1(vec)
# @btime pairwise_product_sum_1($vec)
# @show pairwise_product_sum_1_1(vec)
# @btime pairwise_product_sum_1_1($vec)
# @show pairwise_product_sum_1_2(vec)
# @btime pairwise_product_sum_1_2($vec)
# @show pairwise_product_sum_2(vec)
# @btime pairwise_product_sum_2($vec)

k = 5000

function mean_field_nn_arr(state::Vector, idx::Integer)
    return vcat(1:idx-1, idx+1:length(state))
end

function mean_field_sum_1(state::Vector, idx::Integer)
    state[idx] * sum(state[nn] for nn ∈ mean_field_nn_arr(state, idx))
end

function mean_field_sum_2(state::Vector, idx::Integer)
    state[idx] * (sum(state[begin:idx-1]) + sum(state[idx+1:end]))
end

function mean_field_sum_3(state::Vector, idx::Integer)
    state[idx] * sum(state[vcat(1:idx-1, idx+1:length(state))])
end

@show mean_field_sum_1(vec, k)
@btime mean_field_sum_1($vec, $k)
@show mean_field_sum_2(vec, k)
@btime mean_field_sum_2($vec, $k)
@show mean_field_sum_3(vec, k)
@btime mean_field_sum_3($vec, $k)
