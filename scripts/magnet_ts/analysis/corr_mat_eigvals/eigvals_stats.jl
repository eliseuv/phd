@doc raw"""
    Calculate the mean and variance of the eigenvalues for the normalized correlation matrices of the magnetization time series matrices.
"""

using DrWatson

@quickactivate "phd"

using Logging, JLD2, Statistics, StatsBase, DataFrames, Gadfly, Cairo, LsqFit

include("../../../../src/Thesis.jl")
using .Thesis.DataIO

# Path for datafiles
data_dirpath = datadir("sims", "brass_ca", "magnet_ts", "mult_mat", "rand_start")

# Selected parameters
const prefix_req = "BrassCA2DMagnetTSMatrix"
const params_req = Dict(
    # "dim" => 2,
    "L" => 100,
    "p" => 0.3,
    "n_runs" => 1000,
    "n_samples" => 100,
    "n_steps" => 300
)

# Resulting dataframe
df = DataFrame(r=Float64[],
    lambda_mean=Float64[], lambda_var=Float64[],
    lambda_max_mean=Float64[],
    lambda_gap_mean=Float64[],
    lambda_integral=Float64[])

# Loop on datafiles
for data_filename in readdir(data_dirpath)

    @info data_filename
    (filename_prefix, filename_params) = parse_filename(data_filename)

    # Ignore unrelated data files
    if filename_prefix != prefix_req || !check_params(filename_params, params_req)
        # @info "Skipping unrelated file..."
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
    @show params
    # β = params["beta"]
    r = params["r"]

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
    f(x) = x
    λ_integral = sum(f, λs) / length(λs)

    # Add data to dataframe
    push!(df, Dict(
        # :beta => β,
        :r => r,
        :lambda_mean => λ_mean,
        :lambda_var => λ_var,
        :lambda_max_mean => λ_max_mean,
        :lambda_gap_mean => λ_gap_mean,
        :lambda_integral => λ_integral,
    ))

end

# Process data

# blume_capel_D0_beta_crit = 0.590395
# df[!, :tau] = blume_capel_D0_beta_crit ./ (df[!, :beta])

# ising_2d_temp_crit = 2 / log1p(sqrt(2))
# df[!, :tau] = 1.0 ./ (df[!, :beta] .* ising_2d_temp_crit)

# Display result
display(df)
println()

# Save results
results_prefix = prefix_req * "EigvalsStats"
results_filepath = joinpath(data_dirpath, filename(results_prefix, params_req))
@info "Saving data:" results_filepath
JLD2.save_object(results_filepath, df)

# Fit sigmoid
sigmoid(x, p) = p[1] ./ (1 .+ exp.(-p[2] .* (x .- p[3])))
fit = curve_fit(sigmoid, df[!, :r], df[!, :lambda_var], [40, -1, -0.25])
@show fit.param
x_fit = range(0, 1, 100)
y_fit = sigmoid(x_fit, fit.param)

# Plot results
@info "Plotting results..."
L = params_req["L"]

# plot_title = "Ising 2D (L = $L)"

# D = params_req["D"]
# plot_title = "Blume-Capel 2D (L = $L, D = $D)"

p = params_req["p"]
plot_title = "Brass CA 2D (L = $L, p = $p)"

xdata = "r" => :r
xintercept = [1]

@info "Plotting eigenvalues mean..."
plot_prefix = prefix_req * "EigvalsMean"
plot_filepath = plotsdir(filename(plot_prefix, params_req, ext=".png"))
plt = plot(df, x=xdata.second, y=:lambda_mean,
    Geom.point, Geom.line,
    Guide.title(plot_title * " correlation matrix eigenvalues mean"),
    Guide.xlabel(xdata.first), Guide.ylabel("⟨λ⟩"),
    Coord.cartesian(ymin=0),
    xintercept=xintercept, Geom.vline)
draw(PNG(plot_filepath, 25cm, 15cm), plt)

@info "Plotting eigenvalues variance..."
plot_prefix = prefix_req * "EigvalsVar"
plot_filepath = plotsdir(filename(plot_prefix, params_req, ext=".png"))
plt_fit = layer(x=x_fit, y=y_fit,
    Geom.line)
plt_data = layer(df, x=xdata.second, y=:lambda_var,
    Geom.point)
plt = plot(plt_fit, plt_data,
    Guide.title(plot_title * " correlation matrix eigenvalues variance"),
    Guide.xlabel(xdata.first), Guide.ylabel("⟨λ²⟩ - ⟨λ⟩²"),
    xintercept=xintercept, Geom.vline)
draw(PNG(plot_filepath, 25cm, 15cm), plt)

@info "Plotting maximum eigenvalue mean..."
plot_prefix = prefix_req * "EigvalsMaxMean"
plot_filepath = plotsdir(filename(plot_prefix, params_req, ext=".png"))
plt = plot(df, x=xdata.second, y=:lambda_max_mean,
    Geom.point, Geom.line,
    Guide.title(plot_title * " correlation matrix average maximum eigenvalue"),
    Guide.xlabel(xdata.first), Guide.ylabel("λ₀"),
    xintercept=xintercept, Geom.vline)
draw(PNG(plot_filepath, 25cm, 15cm), plt)

@info "Plotting eigenvalues gap mean..."
plot_prefix = prefix_req * "EigvalsGapMean"
plot_filepath = plotsdir(filename(plot_prefix, params_req, ext=".png"))
plt = plot(df, x=xdata.second, y=:lambda_gap_mean,
    Geom.point, Geom.line,
    Guide.title(plot_title * " correlation matrix average eigenvalue gap"),
    Guide.xlabel(xdata.first), Guide.ylabel("⟨Δλ⟩"),
    xintercept=xintercept, Geom.vline)
draw(PNG(plot_filepath, 25cm, 15cm), plt)

@info "Plotting eigenvalues integral..."
plot_prefix = prefix_req * "EigvalsIntegral"
plot_filepath = plotsdir(filename(plot_prefix, params_req, ext=".png"))
plt = plot(df, x=xdata.second, y=:lambda_integral,
    Geom.point, Geom.line,
    Guide.title(plot_title * " correlation matrix eigenvalue distribution integral"),
    Guide.xlabel(xdata.first), Guide.ylabel("∫ dρ(λ) f(λ)"),
    xintercept=xintercept, Geom.vline)
draw(PNG(plot_filepath, 25cm, 15cm), plt)
