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
const beta = parse(Float64, ARGS[2])
const n_samples = 128
const n_steps = 512
const n_runs = 1024

# Output data directory
output_dir = datadir("sims", "blume-capel", "square_lattice")
mkpath(output_dir)
@show output_dir

# Construct system
@info "Constructing system..."
system = BlumeCapelIsotropicModel(SquareLatticeFiniteState(Val(dim), L, SpinOneState.up))

# Calculate correlation matrices
@info "Calculating cross correlation matrices..."
Gs = map(cross_correlation_matrix ∘ normalize_ts_matrix!,
    magnet_ts_matrix!(system, beta, n_steps, n_samples) for _ ∈ 1:n_runs)

# Get correlation values
corr_vals = sort(vcat(map(triu_values, Gs)...))

# Get eigenvalues
@info "Calculating eigenvalues..."
λs = sort(vcat(map(eigvals, Gs)...))

params_dict =
    output_path = joinpath(output_dir,
        filename("BlumeCapelSqLatticeCorrMatEigvals",
            "D" => 0,
            @varsdict(dim, L, beta, n_samples, n_steps, n_runs)))
@show output_path

# Save result
@info "Saving result to disk..."
jldsave(output_path; corr_vals, eigvals=λs)