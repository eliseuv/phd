@doc raw"""
    Calculation of the eigenvalues of time
    Generate a magnetization time series
"""

# Dr Watson helper
using DrWatson
@quickactivate "phd"

# External libraries
using Logging, Random, LinearAlgebra, JLD2

# Custom modules
include("../../src/Thesis.jl")
using .Thesis.CorrelatedPairs
using .Thesis.RandomMatrices
using .Thesis.DataIO

# Fake command line arguments
# push!(ARGS, "1")

# Parameters to be run
const parameters_combi::Dict{String} = Dict(
    "rho" => parse(Float64, ARGS[1]),
    "t_max" => 2 .^ (1:9),
    "n_pairs" => 2 .^ (0:8),
    "n_samples" => 8192
)

# Output data path
output_datadir = datadir("sims", "random_matrices", "corr_ts")
mkpath(output_datadir)

# Loop on simulation parameters
const parameters_list = dict_list(parameters_combi)
@info "Running $(length(parameters_list)) simulations"
for params in parameters_list

    @info params

    # Fetch parameters
    ρ = params["rho"]
    t_max = params["t_max"]
    n_pairs = params["n_pairs"]
    n_samples = params["n_samples"]

    # Sampler
    spl = CorrelatedTimeSeriesMatrixSampler(ρ, t_max, n_pairs)

    # Calculate eigenvalues
    λs = reduce(vcat,
        map(eigvals ∘ cross_correlation_matrix ∘ normalize_ts_matrix!,
            rand(spl, n_samples))) |> sort!

    # Output datafile
    output_filepath = joinpath(output_datadir, filename("CorrTSEigvals", params))
    save_object(output_filepath, λs)

end
