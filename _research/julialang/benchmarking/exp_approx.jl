using BenchmarkTools

BenchmarkTools.DEFAULT_PARAMETERS.overhead = BenchmarkTools.estimate_overhead()
BenchmarkTools.DEFAULT_PARAMETERS.samples = 1000000
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 10

@inline exp_approx(x::Real) = 1 + x + ((x^2) / 2) + ((x^3) / 6)

@benchmark exp($(rand()*2 - 1))
@benchmark exp_approx($(rand()*2 - 1))