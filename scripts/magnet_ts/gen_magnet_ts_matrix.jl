@doc raw"""
    Generate several runs of the magnetization time series matrices for Brass cellular automaton
"""

# Dr Watson helper
using DrWatson
@quickactivate "phd"

# External libraries
using Logging, Statistics, JLD2, UnicodePlots

# Custom modules
include("../../src/Thesis.jl")
using .Thesis.DataIO
using .Thesis.FiniteStates
using .Thesis.SpinModels
using .Thesis.CellularAutomata
using .Thesis.Names
using .Thesis.Measurements

# Output data path
output_data_path = datadir("sims", "brass_ca", "magnet_ts", "time_series")
mkpath(output_data_path)

# Parameters to be run
const parameters_combi::Dict{String} = Dict(
    "dim" => 2,
    # "N" => parse(Int64, ARGS[1]),
    # "z" => parse(Int64, ARGS[2]),
    "L" => parse(Int64, ARGS[1]),
    "p" => parse(Float64, ARGS[2]),
    "r" => parse(Float64, ARGS[3]),
    # "beta" => parse(Float64, ARGS[3]),
    "n_steps" => 300,
    "n_samples" => 128
)

# Serialize parameters
const parameters_list = dict_list(parameters_combi)
@info "Running $(length(parameters_list)) simulations"

# Loop on simulation parameters
for params in parameters_list

    @info "Parameters:" params

    # Construct system
    system = BrassCellularAutomaton(SquareLatticeFiniteState(Val(params["dim"]), params["L"], BrassState.TH1), params["p"], params["r"])

    # Generate magnetization time series matrices
    @info "Generating magnetization time series..."
    M_ts_avg = vec(magnet_ts_avg!(system, params["n_steps"], params["n_samples"]))
    M_ts_var = vec(magnet_ts_var!(system, params["n_steps"], params["n_samples"]))

    @show M_ts_avg
    @show M_ts_var

    # Data to be saved
    data = Dict{String,Any}()
    data["Params"] = params
    data["M_ts_avg"] = M_ts_avg
    data["M_ts_var"] = M_ts_var

    # Output data file
    output_data_filename = savename(name(system) * "_MagnetTS", params, "jld2")
    output_data_filepath = joinpath(output_data_path, output_data_filename)
    @info "Saving data:" output_data_filepath
    save(output_data_filepath, data)

    # Plot demo time series average
    # display(lineplot(M_ts_avg,
    #     title="Magnet time series average",
    #     xlabel="t", ylabel="⟨Mᵢ⟩(t)",
    #     width=110))
    # println()

end
