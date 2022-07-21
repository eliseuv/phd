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
prefix = "IsingMagnetTSMatrixEigvalsStatsHist"
const params_req = Dict(
    "dim" => 2,
    # "D" => 0,
    # "p" => 0.3,
    "L" => 100,
    "n_samples" => 100,
    "n_steps" => 300,
    "n_runs" => 1000
)

# Load data
data_filepath = find_datafile(data_dirpath, prefix, params_req)
df = load_object(data_filepath)
script_show(df)

# Get parameters values
n_bins_vals = unique(df[:, :n_bins])
interp_vals = unique(df[:, :interp])
@show n_bins_vals interp_vals

# Select parameters
interp = 0

plot_prefix = prefix * "BinCount"

plot_params = deepcopy(params_req)

plot_params["interp"] = interp
df_plot = df[df.interp.==interp, :]

plot_filename = filename(plot_prefix, plot_params, ext=".png")
plot_filepath = plotsdir(plot_filename)

L = plot_params["L"]
# D = plot_params["D"]
plot_title = "Ising (L = $L, interp = $interp) normalized correlation matrix average eigenvalue"
xintercept = [1]

plt = plot(df_plot, x=:tau, y=:lambda_hist_mean,
    color=:n_bins,
    Geom.line,
    Scale.color_discrete(),
    Guide.title(plot_title),
    Guide.xlabel("τ"), Guide.ylabel("⟨λ⟩"),
    xintercept=xintercept, Geom.vline)
draw(PNG(plot_filepath, 25cm, 15cm), plt)

# Select parameters
n_bins = 128

plot_prefix = prefix * "Interp"

plot_params = deepcopy(params_req)

plot_params["n_bins"] = n_bins
df_plot = df[df.n_bins.==n_bins, :]

plot_filename = filename(plot_prefix, plot_params, ext=".png")
plot_filepath = plotsdir(plot_filename)

L = plot_params["L"]
# D = plot_params["D"]
plot_title = "Ising (L = $L, n_bins = $n_bins) normalized correlation matrix average eigenvalue"
xintercept = [1]

plt = plot(df_plot, x=:tau, y=:lambda_hist_mean,
    color=:interp,
    Geom.line,
    Scale.color_discrete(),
    Guide.title(plot_title),
    Guide.xlabel("τ"), Guide.ylabel("⟨λ⟩"),
    xintercept=xintercept, Geom.vline)
draw(PNG(plot_filepath, 25cm, 15cm), plt)
