@doc raw"""
"""

# Dr Watson helper
using DrWatson
@quickactivate "phd"

# External libraries
using Logging, JLD2, StatsBase, CairoMakie

# Custom modules
include("../../../src/Thesis.jl")
using .Thesis.DataIO

const data_dirpath = datadir("sims", "blume-capel", "square_lattice")
const prefix = "BlumeCapelSqLatticeCorrMatEigvals"
const params_req = Dict(
    "dim" => 2,
    "L" => 64,
    "D" => 0,
    "n_runs" => 1024,
    "n_samples" => 128,
    "n_steps" => 512
)

const β_c = 1.69378

for datafile in find_datafiles(data_dirpath, prefix, params_req)

    @info datafile.params

    # Load data from file
    (corr_vals, eigvals) = load(datafile.path, "corr_vals", "eigvals")

    tau = β_c / datafile.params["beta"]

    plt = hist(eigvals, bins=64, normalization=:pdf;
        axis=(; title=L"Eigenvalues distribution $\tau = %$(tau)$"))

    plot_path = plotsdir("blume-capel", filename("BlumeCapelSquareLatticeEigvalsHist", params_req, "tau" => tau, ext="svg"))
    mkpath(dirname(plot_path))
    @info plot_path

    save(plot_path, plt)

end
