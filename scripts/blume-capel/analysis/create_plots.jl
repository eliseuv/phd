# Dr Watson helper
using DrWatson
@quickactivate "phd"

# External libraries
using Logging, PyCall, CSV, DataFrames, StatsBase, LinearAlgebra, LaTeXStrings, CairoMakie

# My libs
include("../../../src/Thesis.jl")
using .Thesis.Metaprogramming
using .Thesis.DataIO

# PyCall to load pickle file
py"""
import pickle

def load_pickle(fpath):
    with open(fpath, "rb") as f:
        data = pickle.load(f)
    return data
"""
@inline load_pickle = py"load_pickle"

# Calculate fluctuations using histograms
@inline function hist_fluctuations(vals::AbstractVector; nbins::Integer=100, closed=:left)
    hist = fit(Histogram, vals, nbins=nbins, closed=closed)
    # x = hist.edges[begin][begin:end-1] + (diff(hist.edges[begin]) ./ 2)
    x = hist.edges[begin][begin:end-1]
    y = normalize(hist.weights)
    mean = x ⋅ y
    mean_2 = (x .* x) ⋅ y
    return (mean, mean_2 - mean^2)
end

@inline make_ticks_log(powers::AbstractVector{<:Real}, base::Integer=10) = (Float64(base) .^ powers, (map(x -> latexstring("$(base)^{$(x)}"), powers)))

# Where to take the data from
# data_dirpath = datadir("blume_capel_pickles")
data_dirpath = joinpath(homedir(), "dados")
@show data_dirpath

const global_prefix = "BlumeCapelSq2D"

# Parse datafiles into dicts
@info "Parsing datafiles..."
eigvals_datafiles = Dict()
correlations_datafiles = Dict()
for (root, _, filenames) in walkdir(data_dirpath)
    for filename in filenames
        path = joinpath(root, filename)
        datafile = DataIO.DataFile(path)
        D = Float64(datafile.params["D"])
        T = Float64(datafile.params["T"])
        if datafile.prefix == global_prefix * "Eigvals"
            if haskey(eigvals_datafiles, D)
                eigvals_datafiles[D][T] = datafile
            else
                eigvals_datafiles[D] = Dict(T => datafile)
            end
            # elseif datafile.prefix == global_prefix * "Correlations"
            #     if haskey(correlations_datafiles, D)
            #         correlations_datafiles[D][T] = datafile
            #     else
            #         correlations_datafiles[D] = Dict(T => datafile)
            #     end
        end
    end
end

# Load temperatures table
@info "Loading temperatures table..."
df_temperatures = DataFrame(CSV.File(projectdir("tables", "butera_and_pernici_2018", "blume-capel_square_lattice.csv")))

# Plot output root directory
output_root = plotsdir("blume_capel_pickles_test")

# Makie theme
# my_theme = Theme(fontsize=24)
# set_theme!(my_theme)

# Loop on anisotropy values
for (D, D_dict) in sort(collect(eigvals_datafiles), by=x -> x[1])

    T_vec = sort(collect(keys(D_dict)))
    # Fetch critical temperature info
    df_D_row = df_temperatures[only(findall(==(D), df_temperatures.anisotropy_field)), 2:end]
    transition_order = lowercase(string(df_D_row[:transition_order]))
    transition_order_str = replace(string(df_D_row[:transition_order]), "First" => "First order", "Second" => "Second order")
    crit_temp_source = findfirst(!ismissing, df_D_row)
    crit_temp_source_str = replace(string(crit_temp_source), "_" => " ")
    T_c = df_D_row[crit_temp_source]
    tau_vec = map(T_vec ./ T_c) do x
        round(x; digits=3)
    end

    println("D = $D ($(transition_order_str))")

    # Create dir
    output_dir_D = joinpath(output_root, "D=$D($(transition_order))")
    mkpath(output_dir_D)

    @info "Plotting fluctuations..."
    # Loop on temperatures
    mean_vec = similar(T_vec)
    var_vec = similar(T_vec)
    for (i, T) in enumerate(T_vec)
        datafile = D_dict[T]
        eigvals = vec(load_pickle(datafile.path))
        mean_vec[i], var_vec[i] = hist_fluctuations(eigvals, nbins=100, closed=:left)
        # mean_vec[i] = mean(eigvals)
        # var_vec[i] = var(eigvals)
    end

    # Plot eigenvalues mean
    fig = Figure()
    ax = Axis(fig[1, 1],
        title=L"Eigenvalues mean $D = %$(D)$",
        xlabel=L"\tau", ylabel=L"\langle \lambda \rangle",
        xticks=0:0.5:6.5)
    scatter!(ax, tau_vec, mean_vec)
    save(joinpath(output_dir_D, filename(global_prefix * "EigvalsMean", @varsdict(D); ext="svg")), fig)

    # Plot eigenvalues variance
    fig = Figure()
    ax = Axis(fig[1, 1],
        title=L"Eigenvalues variance $D = %$(D)$",
        xlabel=L"\tau", ylabel=L"\langle \lambda^2 \rangle - \langle \lambda \rangle^2",
        xticks=0:0.5:6.5)
    scatter!(ax, tau_vec, var_vec)
    save(joinpath(output_dir_D, filename(global_prefix * "EigvalsVar", @varsdict(D); ext="svg")), fig)

    # Plot eigenvalues spacing fluctuations
    # Loop on temperatures
    mean_vec = similar(T_vec)
    var_vec = similar(T_vec)
    for (i, T) in enumerate(T_vec)
        datafile = D_dict[T]
        eigvals_spacings = vcat(map(diff, eachcol(load_pickle(datafile.path)))...)
        mean_vec[i] = mean(eigvals_spacings)
        var_vec[i] = var(eigvals_spacings, mean=mean_vec[i])
    end

    # Plot eigenvalues gaps mean
    fig = Figure()
    ax = Axis(fig[1, 1],
        title=L"Eigenvalues spacing mean $D = %$(D)$",
        xlabel=L"\tau", ylabel=L"\langle\Delta\lambda\rangle",
        xticks=0:0.5:6.5)
    scatter!(ax, tau_vec, mean_vec)
    save(joinpath(output_dir_D, filename(global_prefix * "EigvalsSpacingMean", @varsdict(D); ext="svg")), fig)

    # Plot eigenvalues gaps variance
    fig = Figure()
    ax = Axis(fig[1, 1],
        title=L"Eigenvalues spacing variance $D = %$(D)$",
        xlabel=L"\tau", ylabel=L"\langle \Delta\lambda^2 \rangle - \langle \Delta\lambda \rangle^2",
        xticks=0:0.5:6.5)
    scatter!(ax, tau_vec, var_vec)
    save(joinpath(output_dir_D, filename(global_prefix * "EigvalsSpacingVar", @varsdict(D); ext="svg")), fig)

    @info "Plotting combined histograms..."
    idxs = [1, 5, 6, 7, 8, 16, 18, 19, 21]
    println("tau = ", getindex(tau_vec, idxs))

    # Eigenvalues distribution
    fig = Figure()
    axs = [Axis(fig[i, j],
        limits=((0, nothing), (0, nothing)),
        yticks=make_ticks_log(0:5),
        yscale=Makie.pseudolog10)
           for i ∈ 1:3 for j ∈ 1:3]
    for (ax, idx) ∈ zip(axs, idxs)
        T = T_vec[idx]
        tau = round(T / T_c; digits=3)
        ax.title = L"$\tau = %$(tau)$"
        datafile = D_dict[T]
        eigvals = load_pickle(datafile.path)
        hist!(ax, vec(eigvals), bins=100)
    end
    xlabel = Label(fig[4, 1:3], L"$\lambda$")
    ylabel = Label(fig[1:3, 0], L"$\rho(\lambda)$", rotation=pi / 2)
    save(joinpath(output_dir_D, filename(global_prefix * "EigvalsHist", @varsdict(D); ext="svg")), fig)

    # Eigenvalues spacing distributions
    fig = Figure()
    axs = [Axis(fig[i, j],
        limits=((0, nothing), (0, nothing)),
        yticks=make_ticks_log(0:5),
        yscale=Makie.pseudolog10)
           for i ∈ 1:3 for j ∈ 1:3]
    for (ax, idx) ∈ zip(axs, idxs)
        T = T_vec[idx]
        tau = round(T / T_c; digits=3)
        ax.title = L"$\tau = %$(tau)$"
        datafile = D_dict[T]
        eigvals = load_pickle(datafile.path)
        eigvals_spacings = vcat(map(eachcol(eigvals)) do col
            spacings = diff(col)
            return spacings ./ mean(spacings)
        end...)
        hist!(ax, eigvals_spacings ./ mean(eigvals_spacings), bins=100)
    end
    xlabel = Label(fig[4, 1:3], L"$\lambda$")
    ylabel = Label(fig[1:3, 0], L"$\rho(\Delta\lambda)$", rotation=pi / 2)
    save(joinpath(output_dir_D, filename(global_prefix * "EigvalsSpacingHist", @varsdict(D); ext="svg")), fig)

    # @info "Plotting histograms..."
    # for (T, tau) in zip(T_vec, tau_vec)
    #     datafile = D_dict[T]
    #     eigvals = load_pickle(datafile.path)
    #     @show tau T

    #     # Create dir
    #     output_dir_D_T = joinpath(output_dir_D, "tau=$tau")
    #     mkpath(output_dir_D_T)

    #     # Plot eigenvalues distribution
    #     @info "Plotting eigenvalues distributions..."
    #     fig = Figure()
    #     ax = Axis(fig[1, 1],
    #         title=L"Eigenvalues distribution $D = %$(D)$, $\tau = %$(tau)$",
    #         xlabel=L"\lambda",
    #         ylabel=L"\rho(\lambda)",
    #         limits=((0, nothing), (0, nothing)),
    #         yticks=make_ticks_log(0:5),
    #         yscale=Makie.pseudolog10)
    #     hist!(ax, vec(eigvals), bins=100)
    #     save(joinpath(output_dir_D_T, filename(global_prefix * "EigvalsHist", @varsdict(D, tau); ext="svg")), fig)

    #     # Plot eigenvalues gaps distribution
    #     @info "Plotting eigenvalues spacing distributions..."
    #     fig = Figure()
    #     ax = Axis(fig[1, 1],
    #         title=L"Eigenvalues gap distribution $D = %$(D)$, $\tau = %$(tau)$",
    #         xlabel=L"\lambda",
    #         ylabel=L"\rho(\Delta\lambda)",
    #         limits=((0, nothing), (0, nothing)),
    #         yticks=make_ticks_log(0:5),
    #         yscale=Makie.pseudolog10)
    #     hist!(ax, vcat(map(diff, eachcol(eigvals))...), bins=100)
    #     save(joinpath(output_dir_D_T, filename(global_prefix * "EigvalsSpacingHist", @varsdict(D, tau); ext="svg")), fig)


    #     # Plot min eigenvalue distribution
    #     @info "Plotting min eigenvalue distributions..."
    #     fig = Figure()
    #     ax = Axis(fig[1, 1],
    #         title=L"Minimum eigenvalue distribution $D = %$(D)$, $\tau = %$(tau)$",
    #         xlabel=L"\lambda",
    #         ylabel=L"min(\lambda)",
    #         limits=((0, nothing), (0, nothing)),
    #         yticks=make_ticks_log(0:5),
    #         yscale=Makie.pseudolog10)
    #     hist!(ax, vcat(map(first, eachcol(eigvals))...), bins=100)

    #     save(joinpath(output_dir_D_T, filename(global_prefix * "EigvalsMin", @varsdict(D, tau); ext="svg")), fig)

    #     # Plot min eigenvalue distribution
    #     @info "Plotting max eigenvalue distributions..."
    #     fig = Figure()
    #     ax = Axis(fig[1, 1],
    #         title=L"Maximum eigenvalue distribution $D = %$(D)$, $\tau = %$(tau)$",
    #         xlabel=L"\lambda",
    #         ylabel=L"max(\lambda)",
    #         limits=((0, nothing), (0, nothing)),
    #         yticks=make_ticks_log(0:5),
    #         yscale=Makie.pseudolog10)
    #     hist!(ax, vcat(map(last, eachcol(eigvals))...), bins=100)

    #     save(joinpath(output_dir_D_T, filename(global_prefix * "EigvalsMax", @varsdict(D, tau); ext="svg")), fig)
    # end

end
