@doc raw"""
    Calculate the mean and variance of the eigenvalues for the normalized correlation matrices of the magnetization time series matrices.
"""

using DrWatson

@quickactivate "phd"

using Logging, JLD2, Statistics, StatsBase, DataFrames, Gadfly, Cairo, LsqFit, BSplineKit

include("../../../../src/Thesis.jl")
using .Thesis.DataIO

# Path for datafiles
data_dirpath = datadir("sims", "brass_ca", "magnet_ts", "mult_mat", "rand_start")

# Selected parameters
prefix = "BrassCA2DMagnetTSMatrixEigvalsStats"
const params_req = Dict(
    # "dim" => 2,
    "p" => 0.3,
    # "p" => 0.3,
    "L" => 100,
    "n_samples" => 100,
    "n_steps" => 300,
    "n_runs" => 1000
)

# Search for datafile
@info "Searching data file..."
datafile_path = find_datafiles_with_params(data_dirpath, prefix, params_req)[begin]
df = JLD2.load_object(datafile_path)

# Display result
display(df)
println()

# # Fit sigmoid
# @info "Fitting sigmoid curve..."
# logistic(x, p) = p[1] ./ (1 .+ exp.(-p[2] * (x .- p[3])))
# generalized_logistic(x, p) = p[1] .+ ((p[2] - p[1]) ./ (p[3] .+ p[4] * exp.(-p[5] * (x .- p[7])) .^ (1 / p[6])))
# logistic2(x, p) = p[1] ./ (1 .+ p[3] * exp.(-p[2] * (x .- p[3])))
# sigmoid = generalized_logistic
# # p0 = [40, -10, -0.25]
# p0 = [0, 40, 1, 1, -1, 1, -0.25]
# fit_range = 4:55
# fit = curve_fit(sigmoid, df[fit_range, :r], df[fit_range, :lambda_var], p0)
# @show fit.param
# x_fit = range(0, 1, 100)
# y_fit = sigmoid(x_fit, fit.param)

# Interpolate spline for differentiation
df′ = df[(0.1 .< df.r .< 0.3), :]
spl = interpolate(df′[!, :r], df′[!, :lambda_var], BSplineOrder(5))
x_spl = range(extrema(df′[!, :r])..., 1000)
y_spl = spl.(x_spl)
D2spl = diff(spl, Derivative(2))
y_D2spl = D2spl.(x_spl)

# Plot results
@info "Plotting results..."
plots_params = deepcopy(params_req)
delete!(plots_params, "prefix")
L = params_req["L"]

# plot_title = "Ising 2D (L = $L)"

# D = params_req["D"]
# plot_title = "Blume-Capel 2D (L = $L, D = $D)"

p = params_req["p"]
plot_title = "Brass CA 2D (L = $L, p = $p)"

xdata = "r" => :r
xintercept = [1]

# @info "Plotting eigenvalues mean..."
# plot_prefix = prefix * "EigvalsMean"
# plot_filepath = plotsdir(filename(plot_prefix, results_params, ext=".png"))
# plt = plot(df, x=xdata.second, y=:lambda_mean,
#     Geom.point, Geom.line,
#     Guide.title(plot_title * " correlation matrix eigenvalues mean"),
#     Guide.xlabel(xdata.first), Guide.ylabel("⟨λ⟩"),
#     Coord.cartesian(ymin=0),
#     xintercept=xintercept, Geom.vline)
# draw(PNG(plot_filepath, 25cm, 15cm), plt)

@info "Plotting eigenvalues variance..."
plot_prefix = prefix * "EigvalsVar"
plot_filepath = plotsdir(filename(plot_prefix, plots_params, ext=".png"))
plt_fit = layer(x=x_spl, y=y_spl,
    Geom.line)
plt_data = layer(df, x=xdata.second, y=:lambda_var,
    Geom.point)
plt = plot(plt_fit, plt_data,
    Guide.title(plot_title * " correlation matrix eigenvalues variance"),
    Guide.xlabel(xdata.first), Guide.ylabel("⟨λ²⟩ - ⟨λ⟩²"),
    xintercept=xintercept, Geom.vline)
@info plot_filepath
draw(PNG(plot_filepath, 25cm, 15cm), plt)

@info "Plotting eigenvalues variance..."
plot_prefix = prefix * "EigvalsVarD2"
plot_filepath = plotsdir(filename(plot_prefix, plots_params, ext=".png"))
plt = plot(x=x_spl, y=y_D2spl,
    Geom.line,
    Guide.title(plot_title * " correlation matrix eigenvalues variance (second derivative)"),
    Guide.xlabel(xdata.first), Guide.ylabel("D²(⟨λ²⟩ - ⟨λ⟩²)"))
@info plot_filepath
draw(PNG(plot_filepath, 25cm, 15cm), plt)

# @info "Plotting maximum eigenvalue mean..."
# plot_prefix = prefix * "EigvalsMaxMean"
# plot_filepath = plotsdir(filename(plot_prefix, results_params, ext=".png"))
# plt = plot(df, x=xdata.second, y=:lambda_max_mean,
#     Geom.point, Geom.line,
#     Guide.title(plot_title * " correlation matrix average maximum eigenvalue"),
#     Guide.xlabel(xdata.first), Guide.ylabel("λ₀"),
#     xintercept=xintercept, Geom.vline)
# draw(PNG(plot_filepath, 25cm, 15cm), plt)

# @info "Plotting eigenvalues gap mean..."
# plot_prefix = prefix * "EigvalsGapMean"
# plot_filepath = plotsdir(filename(plot_prefix, results_params, ext=".png"))
# plt = plot(df, x=xdata.second, y=:lambda_gap_mean,
#     Geom.point, Geom.line,
#     Guide.title(plot_title * " correlation matrix average eigenvalue gap"),
#     Guide.xlabel(xdata.first), Guide.ylabel("⟨Δλ⟩"),
#     xintercept=xintercept, Geom.vline)
# draw(PNG(plot_filepath, 25cm, 15cm), plt)
