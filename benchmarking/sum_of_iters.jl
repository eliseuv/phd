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
idx = ntuple(_ -> CartesianIndex(ntuple(_ -> rand(1:L), Val(dim))), Val(n_indices))
idx_vec = collect(idx)

println("\nSum using iteration over tuple")
@btime sum(arr[i] for i ∈ $idx)
@show sum(arr[i] for i ∈ idx)
println("\nSum using iteration over array")
@btime sum(arr[i] for i ∈ $idx_vec)
@show sum(arr[i] for i ∈ idx_vec)
println("\nSum array constructed using iteration over tuple")
@btime sum([arr[i] for i ∈ $idx])
@show sum([arr[i] for i ∈ idx])
println("\nSum array constructed using iteration over array")
@btime sum([arr[i] for i ∈ $idx_vec])
@show sum([arr[i] for i ∈ idx_vec])
println("\nSum using array of indices")
@btime sum(arr[$idx_vec])
@show sum(arr[idx_vec])
println("\nSum using array of indices")
@btime sum(arr[collect($idx)])
@show sum(arr[collect(idx)])
