using DrWatson

@quickactivate "phd"

using JLD2, StatsBase, CairoMakie

CairoMakie.activate!()

include("../src/DataIO.jl")
using .DataIO

data_dirpath = datadir("ising_ts_matrix")

for data_filename in readdir(data_dirpath)

    # Data filepath
    data_filepath = joinpath(data_dirpath, data_filename)
    println(data_filepath)

    # Retrieve eigenvalues
    data = load(data_filepath)
    λs = data["eigvals"]
    script_show(λs)

    # Histogram
    n_bins = 128
    hist = fit(Histogram, λs, range(extrema(λs)..., length = n_bins))
    script_show(hist)

    # Plot filepath
    plot_filename = filename("Ising2DEigvalsHist", data["Params"], ext = "svg")
    plot_filepath = plotsdir(plot_filename)
    println(plot_filepath)

    τ = data["Params"]["tau"]
    hist_bins = (x -> x[1:end-1] - (diff(x) ./ 2))(collect(hist.edges[1]))
    hist_weigths = (x -> log10.(x ./ sum(x)))(hist.weights)

    fig = Figure()
    ax = Axis(fig[1, 1], title = "\$T = $(τ) T_c\$", xlabel = "λ", ylabel = "ρ(λ)")
    scatter!(ax, hist_bins, hist_weigths)
    save(plot_filepath, fig)

end
