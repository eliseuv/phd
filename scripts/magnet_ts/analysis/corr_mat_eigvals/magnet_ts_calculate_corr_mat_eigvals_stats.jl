@doc raw"""
    Calculate the mean and variance of the eigenvalues for the normalized correlation matrices of the magnetization time series matrices.
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
    "L" => 100,
    "n_samples" => 100,
    "n_steps" => 300,
    "n_runs" => 1000
)

# Resulting dataframe
df = DataFrame(beta=Float64[],
    lambda_mean=Float64[], lambda_var=Float64[],
    lambda_max_mean=Float64[],
    lambda_gap_mean=Float64[],
    lambda_hist_mean=Float64[],
    lambda_hist_var=Float64[])

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

    # Fetch eigenvalues matrix
    λs = hcat(data["eigvals"]...)

    # Calculate stats directly
    λ_mean = mean(λs)
    λ_var = varm(λs, λ_mean)

    # Calculate avg max eigval
    λ_max_mean = mean(λs[end, :])

    # Calculate average eigenvalue gap
    λ_gap_mean = mean(diff(λs, dims=1))

    # Build histogram
    n_bins = 100
    hist = fit(Histogram, vec(λs), range(extrema(λs)..., length=n_bins))

    # Histogram analysis
    # hist_bins = hist.edges[1][1:end-1]
    # hist_bins = (x -> x[1:end-1] + (diff(x) ./ 2))(collect(hist.edges[1]))
    hist_bins = hist.edges[1][2:end]
    hist_weights = (x -> (x ./ sum(x)))(hist.weights)

    # script_show(collect(hist.edges[1]))
    # script_show(hist_bins)
    # script_show(hist_weights)
    # Base.exit(0)

    λ_hist_mean = sum(hist_bins .* hist_weights)
    λ_hist_var = sum((hist_bins .^ 2) .* hist_weights) - λ_hist_mean^2

    # Add data to dataframe
    push!(df, Dict(
        :beta => β,
        :lambda_mean => λ_mean,
        :lambda_var => λ_var,
        :lambda_max_mean => λ_max_mean,
        :lambda_gap_mean => λ_gap_mean,
        :lambda_hist_mean => λ_hist_mean,
        :lambda_hist_var => λ_hist_var
    ))

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
results_prefix = prefix * "EigvalsStats"
results_filepath = joinpath(data_dirpath, filename(results_prefix, results_params))
@info "Saving data:" results_filepath
JLD2.save(results_filepath, Dict("eigvals_stats" => df))

# Plot results
@info "Plotting results..."
L = params_req["L"]
plot_title = "Ising (L = $L)"
xdata = :tau => "τ"
xintercept = [1]

# plot_prefix = prefix * "EigvalsMean"
# plot_filepath = plotsdir(filename(plot_prefix, results_params, ext=".png"))
# plt = plot(df, x=xdata.first, y=:lambda_mean,
#     Geom.point, Geom.line,
#     Guide.title(plot_title * " correlation matrix eigenvalues mean"),
#     Guide.xlabel(xdata.second), Guide.ylabel("⟨λ⟩"),
#     Coord.cartesian(ymin=0),
#     xintercept=xintercept, Geom.vline)
# draw(PNG(plot_filepath, 25cm, 15cm), plt)

# plot_prefix = prefix * "EigvalsVar"
# plot_filepath = plotsdir(filename(plot_prefix, results_params, ext=".png"))
# plt = plot(df, x=xdata.first, y=:lambda_var,
#     Geom.point, Geom.line,
#     Guide.title(plot_title * " correlation matrix eigenvalues variance"),
#     Guide.xlabel(xdata.second), Guide.ylabel("⟨λ²⟩ - ⟨λ⟩²"),
#     xintercept=xintercept, Geom.vline)
# draw(PNG(plot_filepath, 25cm, 15cm), plt)

# plot_prefix = prefix * "EigvalsMaxMean"
# plot_filepath = plotsdir(filename(plot_prefix, results_params, ext=".png"))
# plt = plot(df, x=xdata.first, y=:lambda_max_mean,
#     Geom.point, Geom.line,
#     Guide.title(plot_title * " correlation matrix average maximum eigenvalue"),
#     Guide.xlabel(xdata.second), Guide.ylabel("λ₀"),
#     xintercept=xintercept, Geom.vline)
# draw(PNG(plot_filepath, 25cm, 15cm), plt)

# plot_prefix = prefix * "EigvalsGapMean"
# plot_filepath = plotsdir(filename(plot_prefix, results_params, ext=".png"))
# plt = plot(df, x=xdata.first, y=:lambda_gap_mean,
#     Geom.point, Geom.line,
#     Guide.title(plot_title * " correlation matrix average eigenvalue gap"),
#     Guide.xlabel(xdata.second), Guide.ylabel("⟨Δλ⟩"),
#     xintercept=xintercept, Geom.vline)
# draw(PNG(plot_filepath, 25cm, 15cm), plt)

plot_prefix = prefix * "EigvalsHistMeanRight"
plot_filepath = plotsdir(filename(plot_prefix, results_params, ext=".png"))
plt = plot(df, x=xdata.first, y=:lambda_hist_mean,
    Geom.point, Geom.line,
    Guide.title(plot_title * " correlation matrix eigenvalues mean (histogram right alignment)"),
    Guide.xlabel(xdata.second), Guide.ylabel("⟨λ⟩"),
    # Coord.cartesian(ymin=0),
    xintercept=xintercept, Geom.vline)
draw(PNG(plot_filepath, 25cm, 15cm), plt)

plot_prefix = prefix * "EigvalsHistVarRight"
plot_filepath = plotsdir(filename(plot_prefix, results_params, ext=".png"))
plt = plot(df, x=xdata.first, y=:lambda_hist_var,
    Geom.point, Geom.line,
    Guide.title(plot_title * " correlation matrix eigenvalues variance (histogram right alignment)"),
    Guide.xlabel(xdata.second), Guide.ylabel("⟨λ²⟩ - ⟨λ⟩²"),
    xintercept=xintercept, Geom.vline)
draw(PNG(plot_filepath, 25cm, 15cm), plt)
