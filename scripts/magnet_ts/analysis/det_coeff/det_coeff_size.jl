@doc raw"""
    Calculate the eigenvalues for the normalized correlation matrices of the magnetization time series matrices.
"""

using DrWatson

@quickactivate "phd"

using Logging, JLD2, Statistics, DataFrames, GLM, Gadfly, Cairo

include("../../../../src/DataIO.jl")
using .DataIO

# Path for datafiles
datadir_path = datadir("sims", "brass_ca", "magnet_ts", "single_mat", "up_start")

# Desired parameters
prefix = "BrassCATSMatrix"
const params_req = Dict(
    "dim" => 2,
    "p" => 0.3
)
const n_steps = 300

# Filter datafiles
datafile_paths = filter_datadir(datadir_path, "prefix" => prefix, params_req)

# Dataframe to store results
df = DataFrame(L=Int64[],
    r=Float64[],
    z=Float64[], r2=Float64[])

# Loop on selected datafiles
for datafile_path in datafile_paths

    # Load data
    @info "Loading file..." basename(datafile_path)
    data = load(datafile_path)
    @show keys(data) data["Params"]

    # Fetch parameters
    params = data["Params"]
    dim = params["dim"]
    L = params["L"]
    r = params["r"]

    # Fetch matrix
    M_ts = data["M_ts"][begin:(n_steps+1), :]

    # Calculate F₂
    # F₂(t) = <M²>(t) / <M>²(t) ∼ t^{dim/z}
    F₂ = vec(mean(M_ts .^ 2, dims=2) ./ (mean(M_ts, dims=2) .^ 2))

    # Linear regression
    local df_fit = DataFrame(logTime=log.(collect(1:n_steps)),
        logF2=log.(F₂[2:end]))
    lr = lm(@formula(logF2 ~ logTime), df_fit)

    # Add data to dataframe
    push!(df, (L=L, r=r, z=(dim / coef(lr)[2]), r2=r2(lr)))

end

# Drop NaN's
filter!(row -> all(x -> !(x isa Number && isnan(x)), row), df)
sort!(df, [:L])
@show df

# # Plot data
plot_prefix = prefix * "SystemSizeDetCoeff"
plot_params = deepcopy(params_req)
plot_filename = filename(plot_prefix, plot_params, ext=".png")
plot_filepath = plotsdir(plot_filename)
p = plot_params["p"]
plot_title = "Brass CA (p = $p, n_steps = $n_steps) magnetization time series determination coefficient"

@info "Plotting..."
plt = plot(df, x=:r, y=:r2, color=:L,
    Geom.line,
    Scale.color_discrete,
    Guide.title(plot_title),
    Guide.xlabel("r"), Guide.ylabel("Goodness of fit"),
    Guide.colorkey(title="System size"))
draw(PNG(plot_filepath, 25cm, 15cm), plt)
