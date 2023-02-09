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
using .Thesis.DataIO
using .Thesis.FiniteStates
using .Thesis.SpinModels
using .Thesis.TimeSeries

# Magnetization time series matrix
@inline magnet_ts_matrix!(spinmodel::AbstractSpinModel{<:AbstractFiniteState{SpinOneState.T}}, β::Real, n_steps::Integer, n_samples::Integer)::Matrix{Float64} =
    hcat(map(1:n_samples) do _
        set_state!(spinmodel.state, SpinOneState.up)
        return metropolis_measure!(SpinModels.magnet, spinmodel, β, n_steps)
    end...)

# System parameters
const dim = 2
const L = parse(UInt64, ARGS[1])

# Simulation parameters
const β = parse(Float64, ARGS[2])
const n_samples = 512
const n_steps = 128
const n_runs = 16

# Output data directory
output_dir = datadir("sims", "blume-capel", "square_lattice")
mkpath(output_dir)
@show output_dir

# Construct system
@info "Constructing system..."
system = BlumeCapelIsotropicModel(SquareLatticeFiniteState(Val(dim), L, SpinOneState.up))

# Construct magnetization time series
@info "Calculating normalized magnetization time series correlation matrix eigenvalues..."
λs = sort(vcat(
    map(eigvals ∘ cross_correlation_matrix ∘ normalize_ts_matrix!,
        magnet_ts_matrix!(system, β, n_steps, n_samples) for _ ∈ 1:n_runs)...))

output_path = joinpath(output_dir,
    filename("BlumeCapelSqLatticeCorrMatEigvals",
        "dim" => dim, "L" => L, "D" => 0,
        "beta" => β,
        "n_samples" => n_samples, "n_steps" => n_steps, "n_runs" => n_runs))

# Save result
@info "Saving result to disk..."
save_object(output_path, λs)
