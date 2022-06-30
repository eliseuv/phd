@doc raw"""
    Calculate the mean and variance of the eigenvalues for the normalized correlation matrices of the magnetization time series matrices for Brass cellular automaton
"""

using DrWatson

@quickactivate "phd"

using Logging, JLD2, Statistics, DataFrames, UnicodePlots, Gadfly, Cairo

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

# Resulting dataframe
df = DataFrame(p=Float64[], r=Float64[],
    lambda_mean=Float64[], lambda_var=Float64[],
    lambda_max_mean=Float64[],
    lambda_gap_mean=Float64[])

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
    r = params["r"]
    p = params["p"]

    # Fetch eigenvalues matrix
    λs = hcat(data["eigvals"]...)

    # Calculate stats directly
    λ_mean = mean(λs)
    λ_var = varm(λs, λ_mean)

    # Calculate avg max eigval
    λ_max_mean = mean(λs[end, :])

    # Calculate average eigenvalue gap
    λ_gap_mean = mean(diff(λs, dims=1))

    λ_stats = Dict(
        :p => p,
        :r => r,
        :lambda_mean => λ_mean,
        :lambda_var => λ_var,
        :lambda_max_mean => λ_max_mean,
        :lambda_gap_mean => λ_gap_mean
    )

    # Add data to dataframe
    push!(df, λ_stats)

end

# Display result
display(df)
println()

# Save results
results_params = params_req
delete!(results_params, "prefix")
results_prefix = prefix * "EigvalsStats"
results_filepath = joinpath(data_dirpath, filename(results_prefix, results_params))
@info "Saving data:" results_filepath
JLD2.save(results_filepath, Dict("eigvals_stats" => df))

# Plot results
p = params_req["p"]
L = params_req["L"]

plot_prefix = prefix * "EigvalsMean"
plot_filepath = plotsdir(filename(plot_prefix, results_params, ext=".png"))
plt = plot(df, x=:r, y=:lambda_mean,
    Geom.point, Geom.line,
    Guide.title("Brass CA correlation matrix eigenvalues mean (L = $L, p = $p)"),
    Guide.xlabel("r"), Guide.ylabel("⟨λ⟩"))
draw(PNG(plot_filepath, 25cm, 15cm), plt)

plot_prefix = prefix * "EigvalsVar"
plot_filepath = plotsdir(filename(plot_prefix, results_params, ext=".png"))
plt = plot(df, x=:r, y=:lambda_var,
    Geom.point, Geom.line,
    Guide.title("Brass CA correlation matrix eigenvalues variance (L = $L, p = $p)"),
    Guide.xlabel("r"), Guide.ylabel("⟨λ⟩"))
draw(PNG(plot_filepath, 25cm, 15cm), plt)

plot_prefix = prefix * "EigvalsMaxMean"
plot_filepath = plotsdir(filename(plot_prefix, results_params, ext=".png"))
plt = plot(df, x=:r, y=:lambda_max_mean,
    Geom.point, Geom.line,
    Guide.title("Brass CA correlation matrix average maximum eigenvalue (L = $L, p = $p)"),
    Guide.xlabel("r"), Guide.ylabel("λ₀"))
draw(PNG(plot_filepath, 25cm, 15cm), plt)

plot_prefix = prefix * "EigvalsGapMean"
plot_filepath = plotsdir(filename(plot_prefix, results_params, ext=".png"))
plt = plot(df, x=:r, y=:lambda_gap_mean,
    Geom.point, Geom.line,
    Guide.title("Brass CA correlation matrix average eigenvalue gap (L = $L, p = $p)"),
    Guide.xlabel("r"), Guide.ylabel("⟨Δλ⟩"))
draw(PNG(plot_filepath, 25cm, 15cm), plt)
