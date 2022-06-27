@doc raw"""
    Generate several runs of the magnetization time series for Brass cellular automaton
"""

using DrWatson

@quickactivate "phd"

using Logging, JLD2, UnicodePlots

include("../../../src/DataIO.jl")
include("../../../src/BrassCellularAutomaton.jl")

using .DataIO
using .BrassCellularAutomaton

@doc raw"""
    magnet_ts_matrix!(ca::BrassCA, p::Real, r::Real, n_steps::Integer, n_samples::Integer)

Generate a magnetization time series matrix for the Brass CA `ca` starting from a randomized initial state
and using the model probabilities `p` and `r`.

The resulting matrix has `n_samples` columns, each representing a single run of `n_steps` CA steps.
Final matrix dimensions: `n_steps × n_samples`.
"""
magnet_ts_matrix!(ca::BrassCA, p::Real, r::Real, n_steps::Integer, n_samples::Integer) = hcat(map(1:n_samples) do _
    set_state!(ca, TH1)
    return advance_parallel_and_measure!(magnet, ca, p, r, n_steps)
end...)

# Parameters to be run
const parameters_combi = Dict(
    "L" => 100,
    "n_steps" => 300,
    "n_samples" => 1024,
    "p" => 0.3,
    "r" => 0.194421 .+ collect(range(-0.07, 0.07, length = 51))
)

# Serialize parameters
const parameters_list = dict_list(parameters_combi)
println("Running $(length(parameters_list)) simulations.")

# Loop on simulation parameters
for params in parameters_list

    @info "Parameters:" params

    # Parameters
    p = params["p"]
    r = params["r"]
    L = params["L"]
    n_steps = params["n_steps"]
    n_samples = params["n_samples"]

    # Output filename
    data_filepath = datadir("sims", "brass_ca", "magnet_ts", "single_mat", "th1_start", savename("BrassCA2DMagnetTS", params, "jld2"))
    @info "Data file:" data_filepath

    # Check if file already exists
    if isfile(data_filepath)
        @info "File already exists. Skipping..."
        continue
    end

    # Create system
    @info "Creating system..."
    ca = BrassCASquareLattice(Val(2), L, Val(:rand))

    # Calculate magnetization time series matrix
    @info "Calculating magnetization time series matrix..."
    M_ts = magnet_ts_matrix!(ca, p, r, n_steps, n_samples)

    # Plot demo matrix
    # display(heatmap(M_ts,
    #     title = "Magnet time series matrix (p = $p, r = $r)",
    #     xlabel = "i", ylabel = "t", zlabel = "mᵢ(t)",
    #     width = 125))
    # println()

    # Plot demo series
    M_plot = M_ts[:, 1:10]
    x_max = params["n_steps"] + 1
    plt = lineplot(1:x_max, M_plot[:, 1],
        xlim = (0, x_max), ylim = extrema(M_plot),
        xlabel = "t", ylabel = "m",
        width = 125, height = 25)
    for k ∈ 2:size(M_plot, 2)
        lineplot!(plt, 1:x_max, M_plot[:, k])
    end
    display(plt)
    println()

    # Data to be saved
    data = Dict{String,Any}()
    data["Params"] = params
    data["M_ts"] = M_ts

    # Save data
    save(data_filepath, data)
end
