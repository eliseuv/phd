using DrWatson

@quickactivate "phd"

using JLD2, UnicodePlots

include("../src/DataIO.jl")
include("../src/BrassCellularAutomaton.jl")

using .DataIO
using .BrassCellularAutomaton

magnet_ts_matrix!(ca::BrassCA, p::Real, r::Real, n_steps::Integer, n_samples::Integer) = hcat(map(1:n_samples) do _
    randomize_state!(ca)
    return advance_parallel_and_measure!(magnet, ca, p, r, n_steps)
end...)

# All parameters to be run
const parameters_combi = Dict(
    "L" => 100,
    "n_steps" => 300,
    "n_samples" => 100,
    "n_runs" => 1000,
    "p" => 0.3,
    "r" => collect(range(0, 1, length = 11))
)

const parameters_list = dict_list(parameters_combi)
println("Running $(length(parameters_list)) simulations.")

for params in parameters_list

    println(params)

    # Parameters
    p = params["p"]
    r = params["r"]
    L = params["L"]
    n_steps = params["n_steps"]
    n_samples = params["n_samples"]
    n_runs = params["n_runs"]
    @show p r

    # Brass CA system
    ca = BrassCASquareLattice(Val(2), L, Val(:rand))

    # Generate magnetization time series matrices
    M_ts_samples = [map(1:n_runs) do _
        magnet_ts_matrix!(ca, p, r, n_steps, n_samples)
    end...]
    display(heatmap(hcat(M_ts_samples[1:3]...),
        title = "Magnet time series matrix (p = $p, r = $r)",
        xlabel = "i", ylabel = "t", zlabel = "mᵢ(t)",
        width = 125))
    println()

    # Plot demo
    M_plot = M_ts_samples[begin][:, 1:10]
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
    data_filepath = datadir("brass_ca_ts_matrix", savename("BrassCA2DMagnetTSMatrix", params, "jld2"))
    println(data_filepath)

    data = Dict{String,Any}()
    data["Params"] = params
    data["M_ts_samples"] = M_ts_samples

    save(data_filepath, data)
    #safesave(data_filepath, data)
end
