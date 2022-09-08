@doc raw"""
    Calculate the eigenvalues for the normalized correlation matrices of the magnetization time series matrices.
"""

using DrWatson

@quickactivate "phd"

using Logging, JLD2, Statistics, DataFrames, GLM, Gadfly, Cairo

include("../../../../src/Thesis.jl")
using .Thesis.DataIO

# Path for datafiles
data_dirpath = datadir("sims", "brass_ca", "magnet_ts", "single_mat", "up_start")

# Desired parameters
prefix = "BrassCATSMatrix"
const params_req = Dict(
    "dim" => 2,
    "L" => 128,
    "n_steps" => 300,
    "n_samples" => 1024
)

df = DataFrame(p=Float64[], r=Float64[],
    z=Float64[], r2=Float64[])

for data_filename in readdir(data_dirpath)

    @info data_filename
    filename_params = parse_filename(data_filename)
    # script_show(filename_params)

    # Ignore unrelated data files
    if !check_params(filename_params, "prefix" => prefix, params_req)
        @info "Skipping unrelated file..."
        continue
    end

    # Load data
    data_filepath = joinpath(data_dirpath, data_filename)
    @info "Loading file..."
    data = load(data_filepath)

    # Fetch parameters
    params = data["Params"]
    print_dict(params)
    dim = params["dim"]
    L = params["L"]
    p = params["p"]
    r = params["r"]
    println("L = $L, p = $p, r = $r")

    # Fetch matrix
    M_ts = data["M_ts"]
    n_steps = size(M_ts, 1) - 1

    # Calculate F₂
    # F₂(t) = <M²>(t) / <M>²(t) ∼ t^{dim/z}
    F₂ = vec(mean(M_ts .^ 2, dims=2) ./ (mean(M_ts, dims=2) .^ 2))

    # Linear regression
    local df_fit = DataFrame(logTime=log.(collect(1:n_steps)),
        logF2=log.(F₂[2:end]))
    lr = lm(@formula(logF2 ~ logTime), df_fit)

    # Add data to dataframe
    push!(df, (p=p, r=r, z=(dim / coef(lr)[2]), r2=r2(lr)))

end

# Drop NaN's
filter!(row -> all(x -> !(x isa Number && isnan(x)), row), df)
@show df

# Plot data
plot_prefix = prefix * "DetCoeff"
plot_params = deepcopy(params_req)
plot_filename = filename(plot_prefix, plot_params, ext=".png")
plot_filepath = plotsdir(plot_filename)
L = plot_params["L"]
n_steps = plot_params["n_steps"]
n_samples = plot_params["n_samples"]
plot_title = "Brass CA (L = $L, n_steps = $n_steps, n_samples = $n_samples) magnetization time series determination coefficient"

@info "Plotting..."
plt = plot(df, x=:p, y=:r, color=:r2,
    Geom.point,
    Guide.title(plot_title),
    Guide.xlabel("p"), Guide.ylabel("r"),
    Guide.colorkey(title="r²"))
draw(PNG(plot_filepath, 25cm, 15cm), plt)
