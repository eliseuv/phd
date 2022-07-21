@doc raw"""
    Calculate the mean and variance of the eigenvalues for the normalized correlation matrices of the magnetization time series matrices using histograms.
"""

using DrWatson

@quickactivate "phd"

using Logging, JLD2, Statistics, StatsBase, DataFrames, Gadfly, Cairo

include("../../../../src/DataIO.jl")
# include(srcdir("DataIO.jl"))
using .DataIO

# Path for datafiles
data_dirpath = datadir("sims", "ising", "magnet_ts", "mult_mat", "rand_start")

# Selected parameters
prefix = "IsingMagnetTSMatrix"
const params_req = Dict(
    "prefix" => prefix,
    "dim" => 2,
    # "D" => 0,
    # "p" => 0.3,
    "L" => 100,
    "n_samples" => 100,
    "n_steps" => 300,
    "n_runs" => 1000
)

# Interval interpolation function
interpolate(a, b, x) = a + (b - a) * x

# Resulting dataframe
df = DataFrame(
    beta=Float64[],
    n_bins=Int64[],
    interp=Float64[],
    lambda_hist_mean=Float64[],
    lambda_hist_var=Float64[]
)

# Loop on datafiles
for data_filename in readdir(data_dirpath)

    filename_params = parse_filename(data_filename)
    @info data_filename filename_params

    # Ignore unrelated data files
    if !check_params(filename_params, params_req)
        @info "Skipping unrelated file..."
        continue
    end

    # Load data
    data_filepath = joinpath(data_dirpath, data_filename)
    @info "Loading data file..."
    data = load(data_filepath)

    # Skip files without eigenvalues calculated
    if !haskey(data, "eigvals")
        @info "Skipping file without eigenvalues..." keys(data)
        continue
    end

    # Fetch parameters
    params = data["Params"]
    print_dict(params)
    β = params["beta"]
    # r = params["r"]

    # Fetch eigenvalues matrix
    λs = hcat(data["eigvals"]...)

    # Loop on bin count and interpolation
    for (interp, n_bins) in Iterators.product(range(0, 1, step=0.2), 2 .^ (3:11))

        # Build histogram
        hist = fit(Histogram, vec(λs), range(extrema(λs)..., length=n_bins))

        # Histogram analysis
        hist_bins = interpolate(hist.edges[1][1:end-1], hist.edges[1][2:end], interp)
        hist_weights = (x -> (x ./ sum(x)))(hist.weights)

        # Calculate statistics
        λ_hist_mean = sum(hist_bins .* hist_weights)
        λ_hist_var = sum((hist_bins .^ 2) .* hist_weights) - λ_hist_mean^2

        # Add data to dataframe
        push!(df, Dict(
            :beta => β,
            :n_bins => n_bins,
            :interp => interp,
            :lambda_hist_mean => λ_hist_mean,
            :lambda_hist_var => λ_hist_var
        ))

    end

end

# Process data

# blume_capel_D0_beta_crit = 0.590395
# df[!, :tau] = blume_capel_D0_beta_crit ./ (df[!, :beta])

ising_2d_temp_crit = 2 / log1p(sqrt(2))
df[!, :tau] = 1.0 ./ (df[!, :beta] .* ising_2d_temp_crit)

# Display result
display(df)
println()

# Save results
results_params = deepcopy(params_req)
delete!(results_params, "prefix")
results_prefix = prefix * "EigvalsStatsHist"
results_filepath = joinpath(data_dirpath, filename(results_prefix, results_params))
@info "Saving data:" results_filepath
JLD2.save_object(results_filepath, df)
