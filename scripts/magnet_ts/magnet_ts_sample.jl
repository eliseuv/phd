@doc raw"""
    Generate several runs of the magnetization time series matrices for Brass cellular automaton
"""

# Dr Watson helper
using DrWatson
@quickactivate "phd"

# External libraries
using Logging, Statistics, DataFrames, JLD2, UnicodePlots

# Custom modules
include("../../src/Thesis.jl")
using .Thesis.DataIO
using .Thesis.FiniteStates
using .Thesis.SpinModels
using .Thesis.CellularAutomata
using .Thesis.Names
using .Thesis.Measurements

# Parameters to be run
const parameters_combi::Dict{String} = Dict(
    "dim" => 2,
    "L" => parse(Int64, ARGS[1]),
    "p" => parse(Float64, ARGS[2]),
    "r" => parse(Float64, ARGS[3]),
    "n_steps" => 300,
    "n_samples" => 10000,
    "n_runs" => 5
)

# Output data path
output_data_path = datadir("sims", "brass_ca", "magnet_ts", "sampling")
mkpath(output_data_path)

# Loop on simulation parameters
const parameters_list = dict_list(parameters_combi)
@info "Running $(length(parameters_list)) simulations"
for params in parameters_list

    @info "Parameters:" params

    # Construct system
    system = BrassCellularAutomaton(SquareLatticeFiniteState(Val(params["dim"]), params["L"], BrassState.TH0), params["p"], params["r"])

    # Generate magnetization time series matrices
    @info "Generating magnetization time series..."
    M_ts = [magnet_ts_avg!(system, BrassState.TH1, params["n_steps"], params["n_samples"]) for _ ∈ 1:params["n_runs"]]
    @show M_ts

    @info "Calculating average and variance..."
    M_ts_avg = mean(M_ts)
    M_ts_var = varm(M_ts, M_ts_avg)
    @show M_ts_avg
    @show M_ts_var

    # # Plot demo time series average
    # display(lineplot(M_ts_avg,
    #     title="Magnet time series average",
    #     xlabel="t", ylabel="⟨Mᵢ⟩(t)",
    #     width=110))
    # println()

    # Data to be saved
    data = Dict{String,Any}()
    data["Params"] = params
    data["M_ts_avg"] = M_ts_avg
    data["M_ts_var"] = M_ts_var

    # Output data file
    output_data_prefix = name(system) * "MagnetTS"
    output_data_filename = savename(output_data_prefix, params, "jld2")
    output_data_filepath = joinpath(output_data_path, output_data_filename)
    @info "Saving data:" output_data_filepath
    save(output_data_filepath, data)

end
