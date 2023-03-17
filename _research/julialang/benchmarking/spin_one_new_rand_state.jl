using BenchmarkTools

BenchmarkTools.DEFAULT_PARAMETERS.overhead = BenchmarkTools.estimate_overhead()
BenchmarkTools.DEFAULT_PARAMETERS.samples = 1000000
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 10

include("../../../src/Thesis.jl")

using .Thesis.SpinModels

@inline test_rand_new_spin(σ::SpinOneState.T)::SpinOneState.T = SpinOneState.T(mod(Int(σ) + rand(1:2), -1:1))

@info "Compiling..."

rand_new_spin(SpinOneState.zero)

test_rand_new_spin(SpinOneState.zero)


@info "Benchmarking..."

@benchmark rand_new_spin($(rand(instances(SpinOneState.T))))

@benchmark test_rand_new_spin($(rand(instances(SpinOneState.T))))
