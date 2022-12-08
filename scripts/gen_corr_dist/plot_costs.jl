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

for csv_file ∈ csv_files
    prefix, params = parse_filename(csv_file)
    dist = params["dist"]
    push!(distances, dist)
    df_csv = DataFrame(CSV.File(csv_file))
    rename!(df_csv, ["beta_" * dist, "mean_" * dist, "variance_" * dist, "cost_" * dist])
    df_csv[!, "norm_variance_"*dist] = df_csv[!, "variance_"*dist] ./ df_csv[!, "variance_"*dist][begin]
    global df = hcat(df, df_csv)
end

cols = map(dist -> Symbol("norm_variance_" * dist), distances)

plt = plot(stack(df, cols), x=:beta_bhattacharyya, y=:value, color=:variable,
    Geom.line,
    Coord.cartesian(xmin=minimum(df[!, "beta_bhattacharyya"]), xmax=maximum(df[!, "beta_bhattacharyya"])),
    Guide.title("Generate Uniform Correlation Distribution using Simulated Annealing"),
    Guide.xlabel("β"), Guide.ylabel("Normalized Variance"))

draw(PNG(plotsdir("GenUniformCorrDistSA_norm_variance.png"), 25cm, 15cm), plt)
