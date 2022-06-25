using DrWatson

@quickactivate "phd"

using Logging, JLD2, DataFrames, Gadfly, Cairo

include("../src/DataIO.jl")
using .DataIO

# Path for datafiles
data_filepath = joinpath(datadir("ada-lovelace", "brass_ca_ts_matrix_eigvals"), "BrassCA2DMagnetTSMatrixEigvalsStats_L=100_n_runs=1000_n_samples=100_n_steps=300.jld2")

@info "Loading data:" data_filepath
data = load(data_filepath)

df = data["eigvals_stats"]
display(df)
println()

# Plot magnet ts mean
plot_filepath = plotsdir("lambda_mean.png")
plt = plot(df, x = :r, y = :lambda_mean, color = :p, Geom.line,
    Guide.xlabel("r"), Guide.ylabel("mean(λ)"),
    Guide.colorkey(title = "p"))
draw(PNG(plot_filepath, 25cm, 15cm), plt)

# Plot magnet ts var
plot_filepath = plotsdir("lambda_var.png")
plt = plot(df, x = :r, y = :lambda_var, color = :p, Geom.line,
    Guide.xlabel("r"), Guide.ylabel("var(λ)"),
    Guide.colorkey(title = "p"))
draw(PNG(plot_filepath, 25cm, 15cm), plt)
