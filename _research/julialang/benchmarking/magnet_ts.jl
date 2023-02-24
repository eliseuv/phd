@doc raw"""
"""

using BenchmarkTools

BenchmarkTools.DEFAULT_PARAMETERS.overhead = BenchmarkTools.estimate_overhead()
BenchmarkTools.DEFAULT_PARAMETERS.samples = 100000
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 1200

include("../../../src/Thesis.jl")
using .Thesis.FiniteStates
using .Thesis.SpinModels

# System parameters
const dim = 2
const L = 64

# Simulation parameters
const beta = 1.0
const n_samples = 128
const n_steps = 512
const n_runs = 1024

# System
system = BlumeCapelIsotropicModel(SquareLatticeFiniteState(Val(dim), L, SpinOneState.T, Val(:rand)))

# Magnetization time series matrix
@inline magnet_ts_matrix!(spinmodel::AbstractSpinModel, β::Real, n_steps::Integer, n_samples::Integer)::Matrix{Float64} =
    hcat(map(1:n_samples) do _
        randomize_state!(state(spinmodel))
        return metropolis_measure!(SpinModels.magnet, spinmodel, β, n_steps)
    end...)

println("\nMagnetization time series...\n")

@show magnet_ts_matrix!(system, beta, n_steps, n_samples)
@benchmark magnet_ts_matrix!($system, beta, n_steps, n_samples)

# Magnetization time series matrix
@inline magnet_total_ts_matrix!(spinmodel::AbstractSpinModel, β::Real, n_steps::Integer, n_samples::Integer)::Matrix{Float64} =
    hcat(map(1:n_samples) do _
        randomize_state!(state(spinmodel))
        return metropolis_measure!(SpinModels.magnet_total, spinmodel, β, n_steps)
    end...) ./ length(spinmodel)

println("\nMagnetization time series (magnet_total)...\n")

@show magnet_total_ts_matrix!(system, beta, n_steps, n_samples)
@benchmark magnet_total_ts_matrix!($system, beta, n_steps, n_samples)
