@doc raw"""
    Calculate the mean and variance of the eigenvalues for the normalized correlation matrices of the magnetization time series matrices for Brass cellular automaton
"""

using DrWatson

@quickactivate "phd"

using Logging, JLD2, Statistics, DataFrames, UnicodePlots

include("../src/DataIO.jl")
using .DataIO

# Path for datafiles
data_dirpath = datadir("ada-lovelace", "brass_ca_ts_matrix_eigvals")

# Desired parameters
prefix = "BrassCA2DMagnetTSMatrix"
const params_req = Dict(
    "prefix" => prefix,
    "L" => 100,
    "n_runs" => 1000,
    "n_samples" => 100,
    "n_steps" => 300
)

# Resulting dataframes
df = DataFrame(p = Float64[], r = Float64[], lambda_mean = Float64[], lambda_var = Float64[])

for data_filename in readdir(data_dirpath)

    filename_params = parse_filename(data_filename)
    # script_show(filename_params)

    # Ignore unrelated data files
    if !check_params(parse_filename(data_filename), params_req)
        continue
    end

    # Load data
    data_filepath = joinpath(data_dirpath, data_filename)
    @info "Loading data:" data_filepath
    data = load(data_filepath)

    # Skip files without eigenvalues calculated
    if !haskey(data, "eigvals")
        @info "Skipping file:" data_filename
        continue
    end

    # Fetch parameters
    params = data["Params"]
    print_dict(params)
    r = params["r"]
    p = params["p"]

    # Fetch eigenvalues
    λs = vcat(data["eigvals"]...)

    # Calculate stats
    λ_mean = mean(λs)
    λ_stats = Dict(
        :p => p,
        :r => r,
        :lambda_mean => λ_mean,
        :lambda_var => varm(λs, λ_mean)
    )
    print_dict(λ_stats)

    # Add data to dataframe
    push!(df, λ_stats)

end

# Display result
display(df)
println()

# Save results
results_params = params_req
results_prefix = params_req["prefix"] * "EigvalsStats"
delete!(results_params, "prefix")
results_filepath = joinpath(data_dirpath, filename(results_prefix, results_params))
@info "Saving data:" results_filepath
save(results_filepath, Dict("eigvals_stats" => df))
