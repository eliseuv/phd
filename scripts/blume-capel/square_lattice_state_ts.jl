@doc raw"""
    Blume-Capel spin model magnetization time series analysis
"""

# Dr Watson helper
using DrWatson
@quickactivate "phd"

# External libraries
using Logging, LinearAlgebra, JLD2

# Custom modules
include("../../src/Thesis.jl")
using .Thesis.Metaprogramming
using .Thesis.DataIO
using .Thesis.FiniteStates
using .Thesis.SpinModels
using .Thesis.TimeSeries

# State time series matrix
@inline function vinayak_ts_matrix!(spinmodel::AbstractSpinModel{<:AbstractFiniteState{SpinOneState.T}}, β::Real, n_steps::Integer)
    randomize_state!(state(spinmodel))
    return map(Integer, hcat(metropolis_measure!(copy ∘ vec ∘ container ∘ state, spinmodel, β, n_steps)...)) |> transpose |> copy
end

# System parameters
const dim = 2
const L = parse(Int64, ARGS[1])
const D = parse(Float64, ARGS[2])

# Simulation parameters
const beta = parse(Float64, ARGS[3])
const n_steps = 2048
const n_runs = 1024

@show dim L D beta n_steps n_runs

# Output data directory
output_dir = datadir("sims", "blume-capel", "square_lattice", "vinayak")
mkpath(output_dir)
@show output_dir

# Construct system
@info "Constructing system..."
system = BlumeCapelModel(SquareLatticeFiniteState(Val(dim), L, SpinOneState.up), Val(D))
@show typeof(system)

# Calculate correlation matrices
@info "Calculating cross correlation matrices..."
Gs = map(cross_correlation_matrix ∘ normalize_ts_matrix,
    vinayak_ts_matrix!(system, beta, n_steps) for _ ∈ 1:n_runs)

# Get correlation values
corr_vals = map(triu_values, Gs)

# Get eigenvalues
@info "Calculating eigenvalues..."
λs = map(eigvals, Gs)

params_dict =
    output_path = joinpath(output_dir,
        filename("BlumeCapelSqLatticeCorrMatEigvals",
            @varsdict(dim, L, D, beta, n_steps, n_runs)))
@show output_path

# Save result
@info "Saving result to disk..."
jldsave(output_path; corr_vals, eigvals=λs)
