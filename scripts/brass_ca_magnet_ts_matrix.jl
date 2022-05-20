using DrWatson
@quickactivate "phd"

include("../src/brass_ca/BrassCellularAutomaton.jl")
include("../src/brass_ca/BrassCAMagnetTS.jl")

using .BrassCAMagnetTS
using .BrassCAMagnetTS.BrassCellularAutomaton

# All parameters to be run
const parameters_combi = Dict(
    :L => [32, 64, 128, 256],
    :n_samples => 512,
    :n_steps => 1024,
    :p => collect(range(0.0, 1.0, step = 0.1)),
    :r => collect(range(0.0, 1.0, step = 0.1))
)

const parameters_list = dict_list(parameters_combi)
println("Running $(length(parameters_list)) simulations.")

for params in parameters_list

    ca = BrassCASquareLattice(Val(2), params[:L], Int8(+1))
    M_ts = magnet_ts_matrix!(ca, params[:p], params[:r], params[:n_steps], params[:n_samples])

    data = Dict{Symbol,Any}()
    merge!(data, params)
    data[:M_ts] = M_ts

    filename = savename("BrassCA2DMagnetTSMatrix", params, "jld2")
    println(filename)
    @tagsave filename data
end
