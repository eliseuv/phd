using DrWatson

@quickactivate "phd"

using JLD2, UnicodePlots

include("../src/DataIO.jl")
include("../src/IsingModel.jl")

using .DataIO
using .IsingModel

magnet_ts_matrix!(ising::Ising, β::Real, n_steps::Integer, n_samples::Integer) = hcat(map(1:n_samples) do _
    set_state!(ising, Val(:rand))
    return metropolis_and_measure_total_magnet!(ising, β, n_steps) ./ length(ising)
end...)

# All parameters to be run
const parameters_combi = Dict(
    "L" => 100,
    "n_samples" => 100,
    "n_steps" => 512,
    "n_runs" => 1000,
    "tau" => [0.1, 0.5, 0.95, 1, 1.05, 1.1, 1.5, 3.5, 4.5, 6.5]
)

const parameters_list = dict_list(parameters_combi)
println("Running $(length(parameters_list)) simulations.")

for params in parameters_list

    println(params)

    # Ising system
    ising = IsingSquareLattice(Val(2), params["L"], Val(:rand))

    # Temperature
    τ = params["tau"]
    β = ISING_SQ_LAT_2D_BETA_CRIT / τ
    @show τ β

    # Generate magnetization time series matrices
    M_ts = [map(1:params["n_runs"]) do _
        magnet_ts_matrix!(ising, β, params["n_steps"], params["n_samples"])
    end...]
    script_show(M_ts[begin])
    println()
    M_plot = M_ts[begin][:, 1:10]
    x_max = params["n_steps"] + 1
    plt = lineplot(1:x_max, M_plot[:, 1], xlim = (0, x_max), ylim = extrema(M_plot), width = 100, height = 25)
    for k ∈ 2:size(M_plot, 2)
        lineplot!(plt, 1:x_max, M_plot[:, k])
    end
    display(plt)

    # Data to be saved
    data_filepath = datadir("ising_ts_matrix", savename("Ising2DMagnetTSMatrix", params, "jld2"))
    println(data_filepath)

    data = Dict{String,Any}()
    data["Params"] = params
    data["M_ts"] = M_ts

    #save(data_filepath, data)
    #safesave(data_filepath, data)
end
