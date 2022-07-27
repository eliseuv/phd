@doc raw"""
    Calculate the eigenvalues for the normalized correlation matrices of the magnetization time series matrices.
"""

using DrWatson

@quickactivate "phd"

using Logging, JLD2, Statistics, DataFrames, GLM, Gadfly, Cairo

include("../../../../src/DataIO.jl")
using .DataIO

# Path for datafiles
data_dirpath = datadir("sims", "brass_ca", "magnet_ts", "single_mat", "up_start", "old_data")

# Desired parameters
prefix = "BrassCA2DMagnetTS"
const params_req = Dict(
    "p" => 0.3
)
const dim = 2
const n_steps = 300

df = DataFrame(L=Int64[],
    r=Float64[],
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
