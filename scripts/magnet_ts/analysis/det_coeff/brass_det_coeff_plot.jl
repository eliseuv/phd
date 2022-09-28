@doc raw"""

"""

using DrWatson

@quickactivate "phd"

using Logging, CSV, DataFrames, Gadfly, Cairo

include("../../../../src/Thesis.jl")
using .Thesis.DataIO

# Path of datafiles
data_filepath = datadir("sims", "brass_ca", "magnet_ts", "time_series", "BrassCASquareLattice_L=128_dim=2_n_samples=128_n_steps=300.csv")
@info data_filepath

# Parse parameters
(prefix, params) = parse_filename(data_filepath)
@info "Filename parsing:" prefix params

# Read data
@info "Reading data..."
df = CSV.File(data_filepath) |> DataFrame
filter!(:r2 => x -> !isnan(x), df)
df[!, :H] = 1 ./ (1 .- df[!, :r2])
@show df

p_vals = unique(df[!, :p])
df_filtered = similar(df, 0)
for p in p_vals
    df′ = df[df.p.==p, :]
    push!(df_filtered, df′[partialsortperm(df′.r2, 1, rev=true), :])
end
@show df_filtered

@info "Plotting..."
plot_filepath = plotsdir(filename(prefix, params, ext=".png"))
plot_title = "$prefix ($(reduce(*, [string(key, " = ", val, ", ") for (key, val) in params])))"
plt = plot(df, x=:r, y=:H, color=:p,
    Geom.point, Geom.line,
    Guide.title(plot_title),
    Guide.xlabel("r"), Guide.ylabel("G"),
    Guide.colorkey(title="p"),
    Scale.color_discrete_hue())
# Coord.cartesian(xmin=0, xmax=1, ymin=0.9, ymax=1))
draw(PNG(plot_filepath, 50cm, 30cm), plt)
