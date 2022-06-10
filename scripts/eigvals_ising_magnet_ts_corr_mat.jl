using DrWatson

@quickactivate "phd"

using JLD2, LinearAlgebra

include("../src/DataIO.jl")
include("../src/Matrices.jl")
using .DataIO
using .Matrices

data_dirpath = datadir("ising_ts_matrix")

for data_filename in readdir(data_dirpath)

    data_filepath = joinpath(data_dirpath, data_filename)
    println(data_filepath)
    data = load(data_filepath)

    λs = sort(vcat(map(data["M_ts"]) do M
        eigvals(cross_correlation_matrix(normalize_ts_matrix(M)))
    end...))
    data["eigvals"] = λs
    script_show(λs)

    script_show(data)

    save(data_filepath, data)

end
