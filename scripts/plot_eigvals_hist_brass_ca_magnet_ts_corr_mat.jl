using DrWatson

@quickactivate "phd"

using Logging, JLD2, StatsBase, CairoMakie

CairoMakie.activate!()

include("../src/DataIO.jl")
using .DataIO

data_dirpath = datadir("ada-lovelace", "brass_ca_ts_matrix_eigvals")

# Select parameters
const p = 0.3
const n_runs = 1000

for data_filename in readdir(data_dirpath)

    filename_params = parse_filename(data_filename)
    # script_show(filename_params)

    # Ignore unrelated data files
    if filename_params["prefix"] != "BrassCA2DMagnetTSMatrix" ||
       !haskey(filename_params, "p") || filename_params["p"] != p ||
       !haskey(filename_params, "n_runs") || filename_params["n_runs"] != n_runs
        continue
    end

    # Load data
    data_filepath = joinpath(data_dirpath, data_filename)
    @info "Loading data:" data_filepath
    data = load(data_filepath)

    # Fetch parameters
    params = data["Params"]
    print_dict(params)
    r = round(params["r"], digits = 2)

    # Retrieve eigenvalues
    λs = data["eigvals"]
    script_show(λs)
    println()

    # Histogram
    n_bins = 256
    hist = fit(Histogram, λs, range(extrema(λs)..., length = n_bins))

    # Plot filepath
    plot_filename = filename("BrassCA2DEigvalsHist", data["Params"], ext = "svg")
    plot_filepath = plotsdir(plot_filename)
    @info plot_filepath

    hist_bins = (x -> x[1:end-1] - (diff(x) ./ 2))(collect(hist.edges[1]))
    hist_weigths = (x -> log10.(x ./ sum(x)))(hist.weights)
    script_show(hist_bins)
    println()
    script_show(hist_weigths)
    println()

    fig = Figure()
    ax = Axis(fig[1, 1], title = L"r = %$(r)",
        xlabel = L"\lambda", ylabel = L"\log_{10}(\rho(\lambda))",
        yminorticksvisible = true, yminorgridvisible = true,
        yminorticks = IntervalsBetween(8))
    scatter!(ax, hist_bins, hist_weigths)
    save(plot_filepath, fig)

end
