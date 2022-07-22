@doc raw"""
    Generate several runs of the magnetization time series matrices for Brass cellular automaton
"""

using DrWatson

@quickactivate "phd"

using Logging, JLD2, UnicodePlots

include("../../../src/DataIO.jl")
include("../../../src/CellularAutomata.jl")

using .DataIO
using .CellularAutomata

@doc raw"""
    magnet_ts_matrix!(ca::BrassCA, p::Real, r::Real, n_steps::Integer, n_samples::Integer)

Generate a magnetization time series matrix for the Brass CA `ca` starting from a randomized initial state
and using the model probabilities `p` and `r`.

The resulting matrix has `n_samples` columns, each representing a single run of `n_steps` CA steps.
Final matrix dimensions: `n_steps × n_samples`.
"""
magnet_ts_matrix!(ca::AbstractCellularAutomaton, n_steps::Integer, n_samples::Integer) = hcat(map(1:n_samples) do _
    randomize_state!(state(ca))
    return advance_measure!(magnet_total, ca, n_steps)
end...)

# Parameters to be run
const parameters_combi = Dict(
    "dim" => 2,
    "L" => 100,
    "p" => parse(Float64, ARGS[1]),
    "r" => parse(Float64, ARGS[2]),
    "n_steps" => 300,
    "n_samples" => 100,
    "n_runs" => 1000
)

# Serialize parameters
const parameters_list = dict_list(parameters_combi)
@info "Running $(length(parameters_list)) simulations"

# Output data path
output_data_path = datadir("sims", "brass_ca", "magnet_ts", "mult_mat", "rand_start")
mkpath(output_data_path)

# Loop on simulation parameters
for params in parameters_list

    @info "Parameters:" params

    # Parameters
    dim = params["dim"]
    L = params["L"]
    p = params["p"]
    r = params["r"]
    n_steps = params["n_steps"]
    n_samples = params["n_samples"]
    n_runs = params["n_runs"]

    # Brass CA system
    ca = BrassCellularAutomaton(SquareLatticeFiniteState(Val(dim), L, BrassState.TH0), p, r)

    # Generate magnetization time series matrices
    @info "Generating magnetization time series matrices..."
    M_ts_samples = [map(1:n_runs) do _
        magnet_ts_matrix!(ca, n_steps, n_samples)
    end...]

    # # Data to be saved
    # data = Dict{String,Any}()
    # data["Params"] = params
    # data["M_ts_samples"] = M_ts_samples

    # # Output data file
    # output_data_filename = savename("BrassCATSMatrix", params, "jld2")
    # output_data_filepath = joinpath(output_data_path, output_data_filename)
    # @info "Saving data:" output_data_filepath
    # save(output_data_filepath, data)

    # Plot demo matrix
    display(heatmap(hcat(M_ts_samples[1:3]...),
        title="Magnet time series matrix (p = $p, r = $r)",
        xlabel="i", ylabel="t", zlabel="mᵢ(t)",
        width=125))
    println()

    # Plot demo series
    M_plot = M_ts_samples[begin][:, 1:10]
    x_max = params["n_steps"] + 1
    plt = lineplot(1:x_max, M_plot[:, 1],
        xlim=(0, x_max), ylim=extrema(M_plot),
        xlabel="t", ylabel="m",
        width=125, height=25)
    for k ∈ 2:size(M_plot, 2)
        lineplot!(plt, 1:x_max, M_plot[:, k])
    end
    display(plt)
    println()

end
