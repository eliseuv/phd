# Dr Watson helper
using DrWatson
@quickactivate "phd"

# External libraries
using Logging, JLD2, DataFrames, CairoMakie, ColorSchemes

# Custom modules
include("../../src/Thesis.jl")
using .Thesis.DataIO

# Required parameters
const params_req = Dict("gamma" => 1, "dist" => "Distances.euclidean")

# Measurement to plot (`costs` or `variance`)
const measure = "costs"

# Color key
const key_name = "sigma"

# Find datafiles
datafiles =
    find_datafiles(
        datadir(),
        "GenUniformCorrDistSA",
        params_req,
        ext="jld2")

key_values = Set{Float64}()
for datafile ∈ datafiles
    push!(key_values, datafile.params[key_name])
end
key_extrema = extrema(key_values)
@inline normalize(x) = (x - key_extrema[1]) / (key_extrema[2] - key_extrema[1])

@show key_values

fig = Figure(resolution=(1024, 768))
ax = Axis(fig[1, 1],
    title="Generate Uniform Correlation Distribution using Simulated Annealing " * join([name * " = " * string(value) for (name, value) ∈ params_req], ", "),
    xlabel="iter",
    ylabel="Normalied cost")

for datafile ∈ datafiles

    @show datafile

    key_value = datafile.params[key_name]

    df = load(datafile.path, "df")
    df[!, "norm_"*measure] = df[!, measure] ./ df[!, measure][begin]

    lines!(ax, df[!, "norm_"*measure],
        label=L"%$(key_value)",
        color=get(ColorSchemes.viridis, normalize(key_value)))
end

# Add legend
Legend(fig[1, 2], ax, label=L"\sigma")

@inline params_str(params::Dict{String,T}) where {T} = join([string(name) * " = " * string(value) for (name, value) ∈ params], ", ")

const output_plotname = plotsdir(filename("GenUniformCorrDistSA_norm_" * measure, params_req, "var" => key_name, ext="png"))
@show output_plotname
save(output_plotname, fig)
