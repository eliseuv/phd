@doc raw"""
    Calculate the eigenvalues for the normalized correlation matrices of the magnetization time series matrices.
"""

using DrWatson

@quickactivate "phd"

using Logging, JLD2, Statistics, DataFrames, GLM, Gadfly, Cairo

include("../../../../src/Thesis.jl")
using .Thesis.DataIO

# Path for datafiles
data_dirpath = datadir("sims", "brass_ca", "magnet_ts", "single_mat", "up_start", "p-r_map")

# Desired parameters
prefix = "BrassCATSMatrixPRMap"
const params_req = Dict(
    "dim" => 2,
    "L" => 128,
    "n_steps" => 300,
    "n_samples" => 1024
)

for data_filename in readdir(data_dirpath)

    @info data_filename
    (filename_prefix, filename_params) = parse_filename(data_filename)
    # script_show(filename_params)

    # Ignore unrelated data files
    if filename_prefix != prefix || !check_params(filename_params, params_req)
        @info "Skipping unrelated file..."
        continue
    end

    # Load data
    data_filepath = joinpath(data_dirpath, data_filename)
    @info "Loading file..."
    df = load_object(data_filepath)
    @show df

    p_vals = unique(df[!, :p])
    @show p_vals

    df_filtered = similar(df, 0)
    @show df_filtered

    for p in p_vals
        df′ = df[df.p.==p, :]
        push!(df_filtered, df′[partialsortperm(df′.r2, 1, rev=true), :])
    end

    @show df_filtered

    # # Plot data
    # plot_prefix = prefix * "DetCoeff"
    # plot_params = deepcopy(params_req)
    # plot_filename = filename(plot_prefix, plot_params, ext=".png")
    # plot_filepath = plotsdir(plot_filename)
    # L = plot_params["L"]
    # n_steps = plot_params["n_steps"]
    # n_samples = plot_params["n_samples"]
    # plot_title = "Brass CA (L = $L, n_steps = $n_steps, n_samples = $n_samples) magnetization time series determination coefficient"

    # @info "Plotting..."
    # plt = plot(df, x=:p, y=:r, color=:r2,
    #     Geom.point,
    #     Guide.title(plot_title),
    #     Guide.xlabel("p"), Guide.ylabel("r"),
    #     Guide.colorkey(title="r²"))
    # draw(PNG(plot_filepath, 25cm, 15cm), plt)

end
