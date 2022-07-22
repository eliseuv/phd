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
data_dirpath = datadir("sims", "blume_capel", "magnet_ts", "mult_mat", "rand_start")

# Selected parameters
prefix = "BlumeCapel2DMagnetTSMatrix"
const params_req = Dict(
    "prefix" => prefix,
    # "dim" => 2,
    "D" => 0,
    # "p" => 0.3,
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
    lambda_integral=Float64[])

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

    # Calculate stats directly
    λ_mean = mean(λs)
    λ_var = varm(λs, λ_mean)

    # Calculate avg max eigval
    λ_max_mean = mean(λs[end, :])

    # Calculate average eigenvalue gap
    λ_gap_mean = mean(diff(λs, dims=1))

    # Function integral test
    f(x) = x^(-0.1)
    λ_integral = sum(f, λs) / length(λs)

    # Add data to dataframe
    push!(df, Dict(
        :beta => β,
        :lambda_mean => λ_mean,
        :lambda_var => λ_var,
        :lambda_max_mean => λ_max_mean,
        :lambda_gap_mean => λ_gap_mean,
        :lambda_integral => λ_integral,
    ))

end

# Process data

blume_capel_D0_beta_crit = 0.590395
df[!, :tau] = blume_capel_D0_beta_crit ./ (df[!, :beta])

# ising_2d_temp_crit = 2 / log1p(sqrt(2))
# df[!, :tau] = 1.0 ./ (df[!, :beta] .* ising_2d_temp_crit)

# Display result
display(df)
println()

# Save results
results_params = deepcopy(params_req)
delete!(results_params, "prefix")
results_prefix = prefix * "EigvalsStats"
results_filepath = joinpath(data_dirpath, filename(results_prefix, results_params))
@info "Saving data:" results_filepath
JLD2.save_object(results_filepath, df)

# Plot results
@info "Plotting results..."
L = params_req["L"]

# plot_title = "Ising 2D (L = $L)"

D = params_req["D"]
plot_title = "Blume-Capel 2D (L = $L, D = $D)"

xdata = "τ" => :tau
xintercept = [1]

plot_prefix = prefix * "EigvalsMean"
plot_filepath = plotsdir(filename(plot_prefix, results_params, ext=".png"))
plt = plot(df, x=xdata.second, y=:lambda_mean,
    Geom.point, Geom.line,
    Guide.title(plot_title * " correlation matrix eigenvalues mean"),
    Guide.xlabel(xdata.first), Guide.ylabel("⟨λ⟩"),
    Coord.cartesian(ymin=0),
    xintercept=xintercept, Geom.vline)
draw(PNG(plot_filepath, 25cm, 15cm), plt)

plot_prefix = prefix * "EigvalsVar"
plot_filepath = plotsdir(filename(plot_prefix, results_params, ext=".png"))
plt = plot(df, x=xdata.second, y=:lambda_var,
    Geom.point, Geom.line,
    Guide.title(plot_title * " correlation matrix eigenvalues variance"),
    Guide.xlabel(xdata.first), Guide.ylabel("⟨λ²⟩ - ⟨λ⟩²"),
    xintercept=xintercept, Geom.vline)
draw(PNG(plot_filepath, 25cm, 15cm), plt)

plot_prefix = prefix * "EigvalsMaxMean"
plot_filepath = plotsdir(filename(plot_prefix, results_params, ext=".png"))
plt = plot(df, x=xdata.second, y=:lambda_max_mean,
    Geom.point, Geom.line,
    Guide.title(plot_title * " correlation matrix average maximum eigenvalue"),
    Guide.xlabel(xdata.first), Guide.ylabel("λ₀"),
    xintercept=xintercept, Geom.vline)
draw(PNG(plot_filepath, 25cm, 15cm), plt)

plot_prefix = prefix * "EigvalsGapMean"
plot_filepath = plotsdir(filename(plot_prefix, results_params, ext=".png"))
plt = plot(df, x=xdata.second, y=:lambda_gap_mean,
    Geom.point, Geom.line,
    Guide.title(plot_title * " correlation matrix average eigenvalue gap"),
    Guide.xlabel(xdata.first), Guide.ylabel("⟨Δλ⟩"),
    xintercept=xintercept, Geom.vline)
draw(PNG(plot_filepath, 25cm, 15cm), plt)

plot_prefix = prefix * "EigvalsIntegral"
plot_filepath = plotsdir(filename(plot_prefix, results_params, ext=".png"))
plt = plot(df, x=xdata.second, y=:lambda_integral,
    Geom.point, Geom.line,
    Guide.title(plot_title * " correlation matrix eigenvalue distribution integral"),
    Guide.xlabel(xdata.first), Guide.ylabel("∫ dρ(λ) f(λ)"),
    xintercept=xintercept, Geom.vline)
draw(PNG(plot_filepath, 25cm, 15cm), plt)