# Dr Watson helper
using DrWatson
@quickactivate "phd"

# External libraries
using Logging, JLD2, DataFrames, CairoMakie

# Custom modules
include("../../src/Thesis.jl")
using .Thesis.DataIO

# Required parameters
const params_req = Dict("gamma" => 1, "dist" => "Distances.euclidean")

# Measurement to plot (`costs` or `variance`)
const measure = "costs"

# Color key
const color_key = "sigma"

# Find datafiles
datafiles =
    find_datafiles(
        datadir(),
        "GenUniformCorrDistSA",
        params_req,
        ext="jld2")

fig = Figure()
ax = Axis(fig[1, 1])

key_values = Vector()
for datafile ∈ datafiles

    @show datafile

    key_value = datafile.params[color_key]
    push!(key_values, key_value)

    df = load(datafile.path, "df")
    df[!, "norm_"*measure] = df[!, measure] ./ df[!, measure][begin]

    lines!(ax, df[!, "norm_"*measure],)
end

@inline params_str(params::Dict{String,T}) where {T} = join([string(name) * " = " * string(value) for (name, value) ∈ params], ", ")

const output_plotname = plotsdir(filename("GenUniformCorrDistSA_norm_" * measure, params_req, "var" => color_key, ext="png"))
@show output_plotname
save(output_plotname, fig)
