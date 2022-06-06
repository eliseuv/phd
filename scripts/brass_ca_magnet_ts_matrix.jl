using DrWatson

@quickactivate "phd"

include("../src/BrassCAMagnetTS.jl")

using .BrassCAMagnetTS
using .BrassCAMagnetTS.BrassCellularAutomaton

# All parameters to be run
const parameters_combi = Dict(
    :L => [32, 64, 128, 256],
    :n_samples => 512,
    :n_steps => 1024,
    :p => parse(Float64, ARGS[1]),
    :r => collect(range(0.0, 1.0, step = 0.01))
)

const parameters_list = dict_list(parameters_combi)
println("Running $(length(parameters_list)) simulations.")

for params in parameters_list

    println(params)

    ca = BrassCASquareLattice(Val(2), params[:L], TH1)
    M_ts = magnet_ts_matrix!(ca, params[:p], params[:r], params[:n_steps], params[:n_samples])

    data = Dict{Symbol,Any}()
    merge!(data, params)
    data[:M_ts] = M_ts

    filename = datadir(savename("BrassCA2DMagnetTSMatrix", params, "jld2"))
    println(filename)
    safesave(filename, data)
end
