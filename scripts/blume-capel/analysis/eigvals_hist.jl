@doc raw"""
"""

# Dr Watson helper
using DrWatson
@quickactivate "phd"

# External libraries
using Logging, CSV, DataFrames, JLD2, StatsBase, LinearAlgebra, LaTeXStrings, CairoMakie

# Custom modules
include("../../../src/Thesis.jl")
using .Thesis.DataIO

make_ticks_log(powers::AbstractVector{<:Real}, base::Integer=10) = (Float64(base) .^ powers, (map(x -> latexstring("$(base)^{$(x)}"), powers)))

const data_dirpath = datadir("sims", "blume-capel", "square_lattice")
const prefix = "BlumeCapelSqLatticeCorrMatEigvals"
const params_req = Dict(
    "dim" => 2,
    "L" => 64,
    # "D" => 0,
    "n_runs" => 1024,
    "n_samples" => 128,
    "n_steps" => 512
)

const resolution = (600, 400)

# Load critical temperatures dataframes
df_temperatures = DataFrame(CSV.File(projectdir("tables", "butera_and_pernici_2018", "blume-capel_square_lattice.csv")))

for datafile in find_datafiles(data_dirpath, prefix, params_req)

    @info datafile.params

    # Get parameters
    D = datafile.params["D"]
    beta = datafile.params["beta"]

    # Find critical temperature in table
    df_crit_row = df_temperatures[only(findall(==(D), df_temperatures.anisotropy_field)), 2:end]
    crit_temp_source = findfirst(!ismissing, df_crit_row)
    T_c = df_crit_row[crit_temp_source]
    crit_temp_source_str = replace(string(crit_temp_source), "_" => " ")

    tau = round(1.0 / (T_c * beta), digits=5)
    @info D tau

    # Load data from file
    (corr_vals, eigvals) = load(datafile.path, "corr_vals", "eigvals")

    # Sort eigenvalues
    foreach(sort!, eigvals)

    # Plot eigenvalues distribution
    fig = Figure(resolution=resolution)
    ax = Axis(fig[1, 1],
        title=L"Eigenvalues distribution $D = %$(D)$, $\tau = %$(tau)$",
        xlabel=L"\lambda",
        #ylabel=L"\rho(\lambda)",
        limits=((0, nothing), (0, nothing)),
        yticks=make_ticks_log(0:5),
        yscale=Makie.pseudolog10
    )
    plt = hist!(ax, vcat(eigvals...), bins=100;)

    plot_path = plotsdir("blume-capel", filename("BlumeCapelSquareLatticeEigvalsHist", params_req, "tau" => tau, "D" => D, ext="svg"))
    mkpath(dirname(plot_path))
    @info plot_path

    save(plot_path, fig)

    # Plot eigenvalues gap distribution
    fig = Figure(resolution=resolution)
    ax = Axis(fig[1, 1],
        title=L"Eigenvalues gaps $D = %$(D)$, $\tau = %$(tau)$",
        xlabel=L"\Delta\lambda",
        #ylabel=L"\rho(\Delta\lambda)",
        limits=((0, nothing), (0, nothing)),
        yticks=make_ticks_log(0:5),
        yscale=Makie.pseudolog10
    )
    plt = hist!(ax, vcat(map(diff, eigvals)...), bins=100;)

    plot_path = plotsdir("blume-capel", filename("BlumeCapelSquareLatticeEigvalsGapHist", params_req, "tau" => tau, "D" => D, ext="svg"))
    mkpath(dirname(plot_path))
    @info plot_path

    save(plot_path, fig)

end
