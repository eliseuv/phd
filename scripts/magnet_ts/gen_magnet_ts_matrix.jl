@doc raw"""
    Generate several runs of the magnetization time series matrices for Brass cellular automaton
"""

using DrWatson

@quickactivate "phd"

using Logging, JLD2, UnicodePlots

include("../../src/DataIO.jl")
include("../../src/SpinModels.jl")
include("../../src/CellularAutomata.jl")

using .DataIO
using .SpinModels
using .CellularAutomata

@inline magnet_ts!(ising::AbstractIsingModel, β::Real, n_steps::Integer) = metropolis_measure!(SpinModels.magnet_total, ising, β, n_steps)
@inline magnet_ts!(blumecapel::AbstractBlumeCapelModel, β::Real, n_steps::Integer) = heatbath_measure!(SpinModels.magnet_total, blumecapel, β, n_steps)
@inline magnet_ts!(ca::AbstractCellularAutomaton, n_steps::Integer) = advance_measure!(CellularAutomata.magnet_total, ca, n_steps)

@inline magnet_ts_matrix!(ising::AbstractIsingModel, β::Real, n_steps::Integer, n_samples::Integer) = hcat(map(1:n_samples) do _
    SpinModels.set_state!(ising, SpinHalfState.up)
    return magnet_ts!(ising, β, n_steps)
end...)

@inline magnet_ts_matrix!(ca::BrassCellularAutomaton, n_steps::Integer, n_samples::Integer) = hcat(map(1:n_samples) do _
    CellularAutomata.set_state!(ca, BrassState.TH1)
    return magnet_ts!(ca, n_steps)
end...)

# Parameters to be run
const parameters_combi = Dict(
    "dim" => 2,
    "L" => 100,
    "p" => parse(Float64, ARGS[1]),
    "r" => parse(Float64, ARGS[2]),
    "n_steps" => 300,
    "n_samples" => 100,
)

# Output data path
output_data_path = datadir("sims", "brass_ca", "magnet_ts", "single_mat", "up_start")
mkpath(output_data_path)

system_prefix = "BrassCA"

# Serialize parameters
const parameters_list = dict_list(parameters_combi)
@info "Running $(length(parameters_list)) simulations"

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

    # Construct system
    ca = BrassCellularAutomaton(CellularAutomata.SquareLatticeFiniteState(Val(dim), L, BrassState.TH0), p, r)

    # Generate magnetization time series matrices
    @info "Generating magnetization time series matrix..."
    M_ts = magnet_ts_matrix!(ca, n_steps, n_samples)

    # # Data to be saved
    # data = Dict{String,Any}()
    # data["Params"] = params
    # data["M_ts_samples"] = M_ts_samples

    # # Output data file
    # output_data_filename = savename(system_prefix * "TSMatrix", params, "jld2")
    # output_data_filepath = joinpath(output_data_path, output_data_filename)
    # @info "Saving data:" output_data_filepath
    # save(output_data_filepath, data)

    # Plot demo matrix
    display(heatmap(M_ts[end:-1:1, :],
        title="Magnet time series matrix",
        xlabel="i", ylabel="t", zlabel="mᵢ(t)",
        width=110))
    println()

    # Plot demo series
    M_plot = abs.(M_ts[:, 1:10])
    x_max = params["n_steps"] + 1
    plt = lineplot(1:x_max, M_plot[:, 1],
        xlim=(0, x_max), ylim=extrema(M_plot),
        xlabel="t", ylabel="m",
        yscale=:log10,
        width=110, height=25)
    for k ∈ 2:size(M_plot, 2)
        lineplot!(plt, 1:x_max, M_plot[:, k])
    end
    display(plt)
    println()

end
