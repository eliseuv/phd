@doc raw"""
    Generate several runs of the magnetization time series matrices for Brass cellular automaton
"""

# Dr Watson helper
using DrWatson
@quickactivate "phd"

# External libraries
using Logging, JLD2, UnicodePlots

# Custom modules
include("../../src/Thesis.jl")
using .Thesis.DataIO
using .Thesis.FiniteStates
using .Thesis.SpinModels
using .Thesis.CellularAutomata
using .Thesis.Names

# Magnetization time series
@inline magnet_ts!(ising::AbstractIsingModel, β::Real, n_steps::Integer) = metropolis_measure!(SpinModels.magnet_total, ising, β, n_steps)
@inline magnet_ts!(blumecapel::AbstractBlumeCapelModel, β::Real, n_steps::Integer) = heatbath_measure!(SpinModels.magnet_total, blumecapel, β, n_steps)
@inline magnet_ts!(ca::AbstractCellularAutomaton, n_steps::Integer) = advance_measure!(CellularAutomata.magnet_total, ca, n_steps)

# Magnetization time series matrix
@inline magnet_ts_matrix!(ising::AbstractIsingModel, β::Real, n_steps::Integer, n_samples::Integer) = hcat(map(1:n_samples) do _
    set_state!(ising.state, SpinHalfState.up)
    return magnet_ts!(ising, β, n_steps)
end...)
@inline magnet_ts_matrix!(ca::BrassCellularAutomaton, n_steps::Integer, n_samples::Integer) = hcat(map(1:n_samples) do _
    set_state!(ca, BrassState.TH1)
    return magnet_ts!(ca, n_steps)
end...)

# Output data path
output_data_path = datadir("sims", "ising", "magnet_ts", "mult_mat", "rand_start")
mkpath(output_data_path)

# Parameters to be run
const parameters_combi::Dict{String} = Dict(
    # "dim" => 2,
    "N" => parse(UInt64, ARGS[1]),
    # "L" => parse(Int64, ARGS[1]),
    # "p" => parse(Float64, ARGS[2]),
    # "r" => parse(Float64, ARGS[3]),
    "beta" => parse(Float64, ARGS[2]),
    "n_steps" => 300,
    "n_samples" => 100,
    "n_runs" => 1000
)

# Serialize parameters
const parameters_list = dict_list(parameters_combi)
@info "Running $(length(parameters_list)) simulations"

# Loop on simulation parameters
for params in parameters_list

    @info "Parameters:" params

    # Construct system
    system = IsingModel(MeanFieldFiniteState(params["N"], SpinHalfState.up))

    # Generate magnetization time series matrices
    @info "Generating magnetization time series matrices..."
    M_ts_samples = [map(1:params["n_runs"]) do _
                        magnet_ts_matrix!(system, params["beta"], params["n_steps"], params["n_samples"])
    end...]

    # Data to be saved
    data = Dict{String,Any}()
    data["Params"] = params
    data["M_ts_samples"] = M_ts_samples

    # Output data file
    output_data_filename = savename(name(system) * "TSMatrix", params, "jld2")
    output_data_filepath = joinpath(output_data_path, output_data_filename)
    @info "Saving data:" output_data_filepath
    save(output_data_filepath, data)

    # # Plot demo matrix
    # display(heatmap(hcat(M_ts_samples[1:3]...),
    #     title="Magnet time series matrix (p = $p, r = $r)",
    #     xlabel="i", ylabel="t", zlabel="mᵢ(t)",
    #     width=125))
    # println()

    # # Plot demo series
    # M_plot = M_ts_samples[begin][:, 1:10]
    # x_max = params["n_steps"] + 1
    # plt = lineplot(1:x_max, M_plot[:, 1],
    #     xlim=(0, x_max), ylim=extrema(M_plot),
    #     xlabel="t", ylabel="m",
    #     width=125, height=25)
    # for k ∈ 2:size(M_plot, 2)
    #     lineplot!(plt, 1:x_max, M_plot[:, k])
    # end
    # display(plt)
    # println()

end
