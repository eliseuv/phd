@doc raw"""
"""

# Dr Watson helper
using DrWatson
@quickactivate "phd"

# External libraries
using Logging, JLD2, StatsBase, UnicodePlots

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

    # @info datafile

    # Load data from file
    (corr_vals, eigvals) = load(datafile.path, "corr_vals", "eigvals")

    tau = datafile.params["beta"] / β_c
    println("tau = $tau")

    show(histogram(eigvals, nbins=32, closed=:left))

end
