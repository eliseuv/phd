@doc raw"""
    Calculate the mean and variance of the eigenvalues for the normalized correlation matrices of the magnetization time series matrices for Brass cellular automaton
"""

using DrWatson

@quickactivate "phd"

using Logging, JLD2, Statistics, StatsBase, DataFrames, UnicodePlots, Gadfly, Cairo

include("../../../../src/DataIO.jl")
# include(srcdir("DataIO.jl"))
using .DataIO

# Path for datafiles
data_dirpath = datadir("sims", "brass_ca", "magnet_ts", "mult_mat", "rand_start")

# Desired parameters
prefix = "BrassCA2DMagnetTSMatrix"
const params_req = Dict(
    "prefix" => prefix,
    "L" => 100,
    "p" => 0.3,
    "n_runs" => 1000,
    "n_samples" => 100,
    "n_steps" => 300
)

# Resulting dataframes
df = DataFrame(p = Float64[], r = Float64[], lambda_mean = Float64[], lambda_var = Float64[])
df_hist = DataFrame(p = Float64[], r = Float64[], n_bins = Int64[], lambda_mean = Float64[], lambda_var = Float64[])

n_bins_vals = [128, 256, 512]

for data_filename in readdir(data_dirpath)

    @debug data_filename
    filename_params = parse_filename(data_filename)
    # script_show(filename_params)

    # Ignore unrelated data files
    if !check_params(parse_filename(data_filename), params_req)
        @debug "Skipping unrelated file..."
        continue
    end

    # Load data
    data_filepath = joinpath(data_dirpath, data_filename)
    @debug "Loading data file..."
    data = load(data_filepath)

    # Skip files without eigenvalues calculated
    if !haskey(data, "eigvals")
        @debug "Skipping file without eigenvalues..." keys(data)
        continue
    end

    # Fetch parameters
    params = data["Params"]
    print_dict(params)
    r = params["r"]
    p = params["p"]

    # Fetch eigenvalues
    λs = vcat(data["eigvals"]...)

    # Calculate stats directly
    @show sum(λs)
    λ_mean = sum(λs) / length(λs)
    λ_var = varm(λs, λ_mean)

    λ_stats = Dict(
        :p => p,
        :r => r,
        :lambda_mean => λ_mean,
        :lambda_var => λ_var
    )

    # Add data to dataframe
    push!(df, λ_stats)

    # Calculate stats using histogram
    for n_bins ∈ n_bins_vals
        # Calculate histogram
        hist = fit(Histogram, λs, range(extrema(λs)..., length = n_bins), closed = :left)

        # Calculate expected value
        bin_edges = collect(hist.edges[1])
        λ_bins = bin_edges[1:end-1] + (diff(bin_edges) ./ 2)
        λ_weights = hist.weights ./ sum(hist.weights)
        λ_mean_hist = sum(λ_weights .* λ_bins)
        λ_var_hist = sum((λ_bins .^ 2) .* λ_weights) - (λ_mean_hist^2)

        λ_stats_hist = Dict(
            :p => p,
            :r => r,
            :n_bins => n_bins,
            :lambda_mean => λ_mean_hist,
            :lambda_var => λ_var_hist
        )

        # Add data to dataframe
        push!(df_hist, λ_stats_hist)
    end
end

# Display result
display(df)
println()
display(df_hist)
println()

# Plot results
plot_filepath = plotsdir("lambda_mean.png")
plt = plot(df, x = :r, y = :lambda_mean,
    Geom.point, Geom.line,
    Guide.title("Brass CA correlation matrix eigenvalues mean (p = 0.3)"),
    Guide.xlabel("r"), Guide.ylabel("⟨λ⟩"))
draw(PNG(plot_filepath, 25cm, 15cm), plt)

# Plot results
# plot_filepath = plotsdir("lambda_mean_hist.png")
# plt_hist = plot(df_hist, x = :r, y = :lambda_mean, color = :n_bins,
#     Geom.point, Geom.line,
#     Guide.title("Brass CA correlation matrix eigenvalues mean (p = 0.3)"),
#     Guide.xlabel("r"), Guide.ylabel("⟨λ⟩"),
#     Guide.colorkey(title = "Bin count"), Scale.color_discrete)
# draw(PNG(plot_filepath, 25cm, 15cm), plt_hist)

# Save results
results_params = params_req
results_prefix = params_req["prefix"] * "EigvalsStats"
delete!(results_params, "prefix")
results_filepath = joinpath(data_dirpath, filename(results_prefix, results_params))
@debug "Saving data:" results_filepath
JLD2.save(results_filepath, Dict("eigvals_stats" => df))
