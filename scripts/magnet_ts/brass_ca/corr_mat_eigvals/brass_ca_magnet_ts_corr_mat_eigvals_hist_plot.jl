using DrWatson

@quickactivate "phd"

using Logging, JLD2, StatsBase, DataFrames, Gadfly, Cairo

include("../src/DataIO.jl")
using .DataIO

# Path for datafiles
data_dirpath = datadir("ada-lovelace", "brass_ca_ts_matrix_eigvals")

# Select parameters
const params_req = Dict(
    "prefix" => "BrassCA2DMagnetTSMatrix",
    "p" => 0.3,
    "L" => 100,
    "n_runs" => 1000,
    "n_samples" => 100,
    "n_steps" => 300
)

global df = DataFrame(r = Float64[], hist_bins = Float64[], hist_weights = Float64[])

for data_filename in readdir(data_dirpath)

    @info "Data file:" data_filename
    filename_params = parse_filename(data_filename)
    # script_show(filename_params)

    # Ignore unrelated data files
    if !check_params(parse_filename(data_filename), params_req)
        @info "Skipping data file..."
        continue
    end

    # Load data
    @info "Loading data file..."
    data_filepath = joinpath(data_dirpath, data_filename)
    data = load(data_filepath)

    # Fetch parameters
    params = data["Params"]
    print_dict(params)
    r = round(params["r"], digits = 2)

    # Retrieve eigenvalues
    λs = sort(vcat(data["eigvals"]...))
    # script_show(λs)
    # println()
    #
    # # Histogram
    n_bins = 128
    hist = fit(Histogram, λs, range(extrema(λs)..., length = n_bins))

    hist_bins = (x -> x[1:end-1] - (diff(x) ./ 2))(collect(hist.edges[1]))
    hist_weights = (x -> (x ./ sum(x)))(hist.weights)

    global df = vcat(df, DataFrame(r = r, hist_bins = hist_bins, hist_weights = hist_weights))

    # # Plot filepath
    # plot_filename = filename("BrassCA2DEigvalsHist", data["Params"], ext = "svg")
    # plot_filepath = plotsdir(plot_filename)
    # @info plot_filepath

    # fig = Figure()
    # ax = Axis(fig[1, 1], title = L"p = %$(p), r = %$(r)",
    #     xlabel = L"\lambda", ylabel = L"\log_{10}(\rho(\lambda))",
    #     yminorticksvisible = true, yminorgridvisible = true,
    #     yminorticks = IntervalsBetween(8))
    # scatter!(ax, hist_bins, hist_weights)
    # save(plot_filepath, fig)
end

script_show(df)
println()

plot_filepath = plotsdir("lambda_dist.png")
plt = plot(df, x = :hist_bins, y = :hist_weights,
    color = :r,
    alpha = [0.1],
    Geom.line,
    Scale.y_log10,
    Guide.title("Eigenvalues distribution"),
    Guide.xlabel("λ"), Guide.ylabel("ρ(λ)"),
    Guide.colorkey(title = "r"),
    Coord.cartesian(xmin = minimum(df[!, :hist_bins]), xmax = maximum(df[!, :hist_bins])))
draw(PNG(plot_filepath, 25cm, 15cm), plt)
