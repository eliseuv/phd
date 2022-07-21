@doc raw"""
    Plot the histogram of the eigenvalues for the normalized correlation matrices of the magnetization time series matrices.
"""

using DrWatson

@quickactivate "phd"

using Logging, JLD2, StatsBase, DataFrames, UnicodePlots, Gadfly, Cairo

include("../../../../src/DataIO.jl")
using .DataIO

# Path for datafiles
data_dirpath = datadir("sims", "ising", "magnet_ts", "mult_mat", "rand_start")

# Selected parameters
prefix = "IsingMagnetTSMatrix"
const params_req = Dict(
    "dim" => 2,
    "L" => 100,
    "n_runs" => 1000,
    "n_samples" => 100,
    "n_steps" => 300
)

# Resulting dataframe
# df = DataFrame(beta=Float64[],
#     hist_bins=Float64[],
#     hist_weights=Float64[])

# Loop on datafiles
for data_filename in readdir(data_dirpath)

    filename_params = parse_filename(data_filename)
    @info data_filename filename_params

    # Ignore unrelated data files
    if !check_params(filename_params, "prefix" => prefix, params_req)
        @info "Skipping unrelated file..."
        continue
    end

    # Load data
    data_filepath = joinpath(data_dirpath, data_filename)
    @info "Loading data file..."
    data = load(data_filepath)

    # Skip files without eigenvalues calculated
    if !haskey(data, "eigvals")
        @info "Skipping file without eigenvalues..." keys(data)
        continue
    end

    # Fetch parameters
    params = data["Params"]
    print_dict(params)
    β = params["beta"]

    # Fetch eigenvalues
    λs = sort(reduce(vcat, data["eigvals"]))
    # script_show(λs)
    # println()

    # Histogram
    @info "Calculating histogram..."
    n_bins = 128
    hist = fit(Histogram, λs, range(extrema(λs)..., length=n_bins))

    hist_bins = (x -> x[1:end-1] - (diff(x) ./ 2))(collect(hist.edges[1]))
    hist_weights = (x -> (x ./ sum(x)))(hist.weights)

    # push!(df, Dict(:beta => β,
    #     :hist_bins => hist_bins,
    #     :hist_weights => hist_weights))
    df = DataFrame(hist_bins=hist_bins,
        hist_weights=hist_weights)

    # Plot filepath
    plot_prefix = prefix * "EigvalsHist"
    plot_params = deepcopy(params_req)
    ising_2d_temp_crit = 2 / log1p(sqrt(2))
    τ = round(1 / (β * ising_2d_temp_crit), digits=3)
    plot_params["tau"] = τ
    L = plot_params["L"]
    plot_filename = filename(plot_prefix, plot_params, ext=".png")
    plot_filepath = plotsdir(plot_filename)
    plot_title = "Ising (L = $L, τ = $τ) normalized correlation matrix eigenvalues distribution"

    @info "Plotting..." plot_filename
    plt = plot(df, x=:hist_bins, y=:hist_weights,
        Geom.line,
        Guide.title(plot_title),
        Guide.xlabel("λ"), Guide.ylabel("ρ(λ)"),
        Scale.y_log10,
        Coord.cartesian(xmin=0))
    draw(PNG(plot_filepath, 25cm, 15cm), plt)

end

# script_show(df)
# println()

# plot_filepath = plotsdir("lambda_dist.png")
# plt = plot(df, x=:hist_bins, y=:hist_weights,
#     color=:r,
#     alpha=[0.1],
#     Geom.line,
#     Scale.y_log10,
#     Guide.title("Eigenvalues distribution"),
#     Guide.xlabel("λ"), Guide.ylabel("ρ(λ)"),
#     Guide.colorkey(title="r"),
#     Coord.cartesian(xmin=minimum(df[!, :hist_bins]), xmax=maximum(df[!, :hist_bins])))
# draw(PNG(plot_filepath, 25cm, 15cm), plt)
