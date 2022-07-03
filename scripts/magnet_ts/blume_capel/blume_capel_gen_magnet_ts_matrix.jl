@doc raw"""
    Generate several runs of the magnetization time series matrices for Blume Capel model.
"""

using DrWatson

@quickactivate "phd"

using Logging, JLD2, UnicodePlots

include("../../../src/DataIO.jl")
include("../../../src/BlumeCapelModel.jl")

using .DataIO
using .BlumeCapelModel

@doc raw"""
"""
magnet_ts_matrix!(bc::BlumeCapel, β::Real, n_steps::Integer, n_samples::Integer) = hcat(map(1:n_samples) do _
    randomize_state!(bc)
    return heatbath_and_measure_total_magnet!(bc, β, n_steps)
end...)

# Parameters to be run
const parameters_combi = Dict(
    "L" => 100,
    "n_steps" => 300,
    "n_samples" => 100,
    "n_runs" => 1000,
    "beta" => parse(Float64, ARGS[1]),
    "D" => parse(Float64, ARGS[2])
)

# Serialize parameters
const parameters_list = dict_list(parameters_combi)
@info "Running $(length(parameters_list)) simulations"

# Output data path
output_data_path = datadir("sims", "blume_capel", "magnet_ts", "mult_mat", "rand_start")
mkpath(output_data_path)

# Loop on simulation parameters
for params in parameters_list

    @info "Parameters:" params

    # Parameters
    β = params["beta"]
    D = params["D"]
    L = params["L"]
    n_steps = params["n_steps"]
    n_samples = params["n_samples"]
    n_runs = params["n_runs"]

    # Blume-Capel system
    @info "Generating system..."
    bc = BlumeCapelSquareLattice(Val(2), L, Val(:rand), D)

    # Generate magnetization time series matrices
    @info "Generating magnetization time series matrices..."
    M_ts_samples = [map(1:n_runs) do _
        magnet_ts_matrix!(bc, β, n_steps, n_samples)
    end...]

    # Data to be saved
    data = Dict{String,Any}()
    data["Params"] = params
    data["M_ts_samples"] = M_ts_samples

    # Output data file
    output_data_filename = savename("BlumeCapel2DMagnetTSMatrix", params, "jld2")
    output_data_filepath = joinpath(output_data_path, output_data_filename)
    @info "Saving data:" output_data_filepath
    save(output_data_filepath, data)

    # Plot demo matrix
    # display(heatmap(hcat(M_ts_samples[1:3]...),
    #     title="Blume Capel magnet time series matrix (L = $L, β = $β)",
    #     xlabel="i", ylabel="t", zlabel="mᵢ(t)",
    #     width=125))
    # println()

    # Plot demo series
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
