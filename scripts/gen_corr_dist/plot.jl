@doc raw"""
    Generate a magnetization time series
"""

# Dr Watson helper
using DrWatson
@quickactivate "phd"

# External libraries
using Logging, DataFrames, CSV, Gadfly, Cairo

# Custom modules
include("../../src/Thesis.jl")
using .Thesis.DataIO

csv_files = map(datadir, keep_extension("csv", readdir(datadir())))

distances = Vector{String}()
df = DataFrame()

# Cost or variance
measure = "cost"

for csv_file ∈ csv_files
    prefix, params = parse_filename(csv_file)
    dist = params["dist"]
    push!(distances, dist)
    df_csv = DataFrame(CSV.File(csv_file))
    rename!(df_csv, ["beta_" * dist, "mean_" * dist, "variance_" * dist, "cost_" * dist])
    df_csv[!, "norm_"*measure*"_"*dist] = df_csv[!, measure*"_"*dist] ./ df_csv[!, measure*"_"*dist][begin]
    global df = hcat(df, df_csv)
end

cols = map(dist -> Symbol("norm_" * measure * "_" * dist), distances)

plt = plot(stack(df, cols), x=:beta_bhattacharyya, y=:value, color=:variable,
    Geom.line,
    Coord.cartesian(xmin=minimum(df[!, "beta_bhattacharyya"]), xmax=maximum(df[!, "beta_bhattacharyya"])),
    Guide.title("Generate Uniform Correlation Distribution using Simulated Annealing"),
    Guide.xlabel("β"), Guide.ylabel("Normalized " * measure))

draw(PNG(plotsdir("GenUniformCorrDistSA_norm_" * measure * ".png"), 30cm, 18cm), plt)
