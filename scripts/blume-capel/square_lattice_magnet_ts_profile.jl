@doc raw"""
    Blume-Capel spin model magnetization time series analysis
"""

# Dr Watson helper
using DrWatson
@quickactivate "phd"

# External libraries
using Logging, LinearAlgebra, JLD2, Profile

# Custom modules
include("../../src/Thesis.jl")
using .Thesis.Metaprogramming
using .Thesis.DataIO
using .Thesis.FiniteStates
using .Thesis.SpinModels
using .Thesis.TimeSeries

# Magnetization time series matrix
@inline magnet_ts_matrix!(spinmodel::AbstractSpinModel{<:AbstractFiniteState{SpinOneState.T}}, β::Real, n_steps::Integer, n_samples::Integer)::Matrix{Float64} =
    hcat(map(1:n_samples) do _
        randomize_state!(state(spinmodel))
        return metropolis_measure!(SpinModels.magnet, spinmodel, β, n_steps)
    end...)

@inline correlation_matrix!(M::AbstractMatrix{<:Real}) = (cross_correlation_matrix ∘ normalize_ts_matrix!)(M)

# System parameters
const dim = 2
const L = 64
const D = 0

# Simulation parameters
const beta = 1.0
const n_samples = 1000
const n_steps = 300
const n_runs = 4

@show dim L D beta n_samples n_steps n_runs

# Output data directory
output_dir = datadir("sims", "blume-capel", "square_lattice")
mkpath(output_dir)
@show output_dir

# Construct system
@info "Constructing system..."
system = BlumeCapelModel(SquareLatticeFiniteState(Val(dim), L, SpinOneState.up), Val(D))
@show typeof(system)

# Calculate correlation matrices
@info "Calculating cross correlation matrices..."
# Compile
@info "Compiling..."
correlation_matrix!(magnet_ts_matrix!(system, beta, n_steps, n_samples))
# Profile
@info "Profiling..."
@profview correlation_matrix!(magnet_ts_matrix!(system, beta, n_steps, n_samples))
