@doc raw"""
    Generate a magnetization time series
"""

# Dr Watson helper
using DrWatson
@quickactivate "phd"

# External libraries
using Logging, DataFrames, JLD2, Gadfly, Cairo

# Custom modules
include("../../src/Thesis.jl")
using .Thesis.DataIO

const params_req = Dict("gamma" => 1, "dist" => "Distances.euclidean")

# Get datafiles
datafiles =
    find_datafiles(
        datadir(),
        "GenUniformCorrDistSA",
        params_req,
        ext="jld2")

@inline params_str(params::Dict{String,T}) where {T} = join([string(name) * " = " * string(value) for (name, value) ∈ params], ", ")

@show datafiles

# Cost or variance
const measure = "cost"
const var_name = "sigma"

var_values = Vector()
df = DataFrame()
for datafile ∈ datafiles

    var_value = datafile.params[var_name]
    push!(var_values, var_value)

    df_it = load(datafile.path, "df")
    rename!(df_it, ["beta_" * string(var_value), "mean_" * string(var_value), "variance_" * string(var_value), "cost_" * string(var_value)])
    df_it[!, var_name*"="*string(var_value)] = df_it[!, measure*"_"*string(var_value)] ./ df_it[!, measure*"_"*string(var_value)][begin]
    global df = hcat(df, df_it)
end

df[!, :iter] = 0:(nrow(df)-1)

var_cols = map(val -> Symbol(var_name * "=" * string(val)), var_values)
beta_col = Symbol(filter(x -> startswith(x, "beta_"), names(df))[begin])

plt = plot(stack(df, var_cols), x=:iter, y=:value, color=:variable,
    Geom.line,
    Guide.title("Generate Uniform Correlation Distribution using Simulated Annealing " * params_str(params_req)),
    # Coord.cartesian(xmin=minimum(df[!, beta_col]), xmax=maximum(df[!, beta_col])),
    # Guide.xlabel("β"),
    # Scale.x_log10,
    Scale.color_continuous(),
    Coord.cartesian(xmin=minimum(df[!, :iter]), xmax=maximum(df[!, :iter])),
    Guide.xlabel("iter"),
    Guide.ylabel("Normalized " * measure),
    Guide.colorkey(title=var_name, labels=map(string, var_values)))

const output_plotname = filename("GenUniformCorrDistSA_norm_" * measure, params_req, "var" => var_name, ext="png")
@show output_plotname
draw(PNG(plotsdir(output_plotname), 30cm, 18cm), plt)
