# Dr Watson helper
using DrWatson
@quickactivate "phd"

# External libraries
using Logging, PyCall, CSV, DataFrames, StatsBase, LinearAlgebra, LaTeXStrings, CairoMakie

# My libs
include("../../../src/Thesis.jl")
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
@inline function hist_fluctuations(vals::AbstractVector; nbins::Integer=128, closed=:left)
    hist = fit(Histogram, vals, nbins=nbins, closed=closed)
    x = hist.edges[begin][begin:end-1] + (diff(hist.edges[begin]) ./ 2)
    y = normalize(hist.weights)
    mean = x ⋅ y
    mean_2 = (x .* x) ⋅ y
    return (mean, mean_2 - mean^2)
end

@inline make_ticks_log(powers::AbstractVector{<:Real}, base::Integer=10) = (Float64(base) .^ powers, (map(x -> latexstring("$(base)^{$(x)}"), powers)))

# Where to take the data from
data_dirpath = datadir("blume_capel_pickles")
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
        elseif datafile.prefix == global_prefix * "Correlations"
            if haskey(correlations_datafiles, D)
                correlations_datafiles[D][T] = datafile
            else
                correlations_datafiles[D] = Dict(T => datafile)
            end
        end
    end
end

# Create output directories
@info "Creating directories..."
output_dir = plotsdir("blume_capel_pickles")
for (D, D_dict) in eigvals_datafiles
    for (T, datafile) in D_dict
        new_dir = joinpath(output_dir, "D=$D", "T=$T")
        mkpath(new_dir)
    end
end
return

# Load temperatures table
@info "Loading temperatures table..."
df_temperatures = DataFrame(CSV.File(projectdir("tables", "butera_and_pernici_2018", "blume-capel_square_lattice.csv")))

# Loop on anisotropy values
for (D, D_dict) in sort(collect(eigvals_datafiles), by=x -> x[1])

    @show D

    T_vec = sort(collect(keys(D_dict)))

    df_D_row = df_temperatures[only(findall(==(D), df_temperatures.anisotropy_field)), 2:end]
    transition_order = replace(string(df_D_row[:transition_order]), "First" => "First order", "Second" => "Second order")
    crit_temp_source = findfirst(!ismissing, df_D_row)
    T_c = df_D_row[crit_temp_source]
    crit_temp_source_str = replace(string(crit_temp_source), "_" => " ")

    # @info "Plotting fluctuations..."
    # # Loop on temperatures
    # mean_vec = similar(T_vec)
    # var_vec = similar(T_vec)
    # for (i, T) in enumerate(T_vec)
    #     datafile = D_dict[T]
    #     eigvals = vec(load_pickle(datafile.path))
    #     mean_vec[i], var_vec[i] = hist_fluctuations(eigvals, nbins=128, closed=:left)
    # end

    # # Plot eigenvalues mean
    # fig = Figure()
    # ax = Axis(fig[1, 1],
    #     title=L"Eigenvalues mean $D = %$(D)$",
    #     xlabel=L"\tau", ylabel=L"\langle \lambda \rangle",
    #     xticks=0:0.5:6.5)
    # scatter!(ax, T_vec ./ T_c, mean_vec)

    # output_filename = global_prefix * "EigvalsMean_D=$D.svg"
    # output_path = joinpath(output_dir, "D=$D", output_filename)
    # save(output_path, fig)

    # # Plot eigenvalues variance
    # fig = Figure()
    # ax = Axis(fig[1, 1],
    #     title=L"Eigenvalues variance $D = %$(D)$",
    #     xlabel=L"\tau", ylabel=L"\langle \lambda^2 \rangle - \langle \lambda \rangle^2",
    #     xticks=0:0.5:6.5)
    # scatter!(ax, T_vec ./ T_c, var_vec)

    # output_filename = global_prefix * "EigvalsVar_D=$D.svg"
    # output_path = joinpath(output_dir, "D=$D", output_filename)
    # save(output_path, fig)

    # Plot eigenvalues gaps mean
    # Loop on temperatures
    mean_vec = similar(T_vec)
    var_vec = similar(T_vec)
    for (i, T) in enumerate(T_vec)
        datafile = D_dict[T]
        eigvals_gaps = vcat(map(diff, eachcol(load_pickle(datafile.path)))...)
        mean_vec[i] = mean(eigvals_gaps)
        var_vec[i] = var(eigvals_gaps, mean=mean_vec[i])
    end

    # # Plot eigenvalues gaps mean
    # fig = Figure()
    # ax = Axis(fig[1, 1],
    #     title=L"Eigenvalue gaps mean $D = %$(D)$",
    #     xlabel=L"\tau", ylabel=L"\langle\Delta\lambda\rangle",
    #     xticks=0:0.5:6.5)
    # scatter!(ax, T_vec ./ T_c, mean_vec)

    # output_filename = global_prefix * "EigvalGapsMean_D=$D.svg"
    # output_path = joinpath(output_dir, "D=$D", output_filename)
    # save(output_path, fig)

    # Plot eigenvalues gaps variance
    fig = Figure(fontsize=24)
    ax = Axis(fig[1, 1],
        title=L"Eigenvalue gaps variance $D = %$(D)$",
        xlabel=L"\tau", ylabel=L"\langle \Delta\lambda^2 \rangle - \langle \Delta\lambda \rangle^2",
        xticks=0:0.5:6.5)
    scatter!(ax, T_vec ./ T_c, var_vec)

    # output_filename = global_prefix * "EigvalGapsVar_D=$D.png"
    # output_path = plotsdir("enviar", output_filename)
    output_filename = global_prefix * "EigvalGapsVar_D=$D.svg"
    output_path = joinpath(output_dir, "D=$D", output_filename)
    save(output_path, fig, pt_per_unit=1)

    # @info "Plotting combined histograms..."
    # idxs = [1, 5, 6, 7, 8, 16, 18, 19, 21]

    # # Eigenvalues distribution
    # fig = Figure()
    # axs = [Axis(fig[i, j],
    #     limits=((0, nothing), (0, nothing)),
    #     yticks=make_ticks_log(0:5),
    #     yscale=Makie.pseudolog10)
    #        for i ∈ 1:3 for j ∈ 1:3]
    # for (ax, idx) ∈ zip(axs, idxs)
    #     T = T_vec[idx]
    #     tau = round(T / T_c; digits=3)
    #     ax.title = L"$\tau = %$(tau)$"
    #     datafile = D_dict[T]
    #     eigvals = load_pickle(datafile.path)
    #     hist!(ax, vec(eigvals), bins=100)
    # end
    # xlabel = Label(fig[4, 1:3], L"$\lambda$")
    # ylabel = Label(fig[1:3, 0], L"$\rho(\lambda)$", rotation=pi / 2)

    # output_filename = global_prefix * "EigvalsDistribution_D=$(D).svg"
    # output_path = joinpath(output_dir, "D=$D", output_filename)
    # save(output_path, fig)

    # # Eigenvalues gap distributions
    # fig = Figure()
    # axs = [Axis(fig[i, j],
    #     limits=((0, nothing), (0, nothing)),
    #     yticks=make_ticks_log(0:5),
    #     yscale=Makie.pseudolog10)
    #        for i ∈ 1:3 for j ∈ 1:3]
    # for (ax, idx) ∈ zip(axs, idxs)
    #     T = T_vec[idx]
    #     tau = round(T / T_c; digits=3)
    #     ax.title = L"$\tau = %$(tau)$"
    #     datafile = D_dict[T]
    #     eigvals = load_pickle(datafile.path)
    #     hist!(ax, vcat(map(diff, eachcol(eigvals))...), bins=100)
    # end
    # xlabel = Label(fig[4, 1:3], L"$\lambda$")
    # ylabel = Label(fig[1:3, 0], L"$\rho(\Delta\lambda)$", rotation=pi / 2)

    # output_filename = global_prefix * "EigvalsGaps_D=$(D).svg"
    # output_path = joinpath(output_dir, "D=$D", output_filename)
    # save(output_path, fig)

    # @info "Plotting histograms..."
    # for (i, T) in enumerate(T_vec)
    #     datafile = D_dict[T]
    #     eigvals = load_pickle(datafile.path)
    #     tau = round(T / T_c; digits=3)
    #     @show tau T

    #     # Plot eigenvalues distribution
    #     #= @info "Plotting eigenvalues distributions..." =#
    #     fig = Figure()
    #     ax = Axis(fig[1, 1],
    #         title=L"Eigenvalues distribution $D = %$(D)$, $\tau = %$(tau)$",
    #         xlabel=L"\lambda",
    #         ylabel=L"\rho(\lambda)",
    #         limits=((0, nothing), (0, nothing)),
    #         yticks=make_ticks_log(0:5),
    #         yscale=Makie.pseudolog10)
    #     hist!(ax, vec(eigvals), bins=100)

    #     output_filename = global_prefix * "EigvalsDistribution_D=$(D)_T=$(T).svg"
    #     output_path = joinpath(output_dir, "D=$D", "T=$T", output_filename)
    #     save(output_path, fig)

    #     # Plot eigenvalues gaps distribution
    #     #= @info "Plotting eigenvalues gaps distributions..." =#
    #     fig = Figure()
    #     ax = Axis(fig[1, 1],
    #         title=L"Eigenvalues gap distribution $D = %$(D)$, $\tau = %$(tau)$",
    #         xlabel=L"\lambda",
    #         ylabel=L"\rho(\Delta\lambda)",
    #         limits=((0, nothing), (0, nothing)),
    #         yticks=make_ticks_log(0:5),
    #         yscale=Makie.pseudolog10)
    #     hist!(ax, vcat(map(diff, eachcol(eigvals))...), bins=100)

    #     output_filename = global_prefix * "EigvalsGaps_D=$(D)_T=$(T).svg"
    #     output_path = joinpath(output_dir, "D=$D", "T=$T", output_filename)
    #     save(output_path, fig)

    #     # Plot min eigenvalue distribution
    #     #= @info "Plotting min eigenvalue distributions..." =#
    #     fig = Figure()
    #     ax = Axis(fig[1, 1],
    #         title=L"Minimum eigenvalue distribution $D = %$(D)$, $\tau = %$(tau)$",
    #         xlabel=L"\lambda",
    #         ylabel=L"min(\lambda)",
    #         limits=((0, nothing), (0, nothing)),
    #         yticks=make_ticks_log(0:5),
    #         yscale=Makie.pseudolog10)
    #     hist!(ax, vcat(map(first, eachcol(eigvals))...), bins=100)

    #     output_filename = global_prefix * "EigvalsMin_D=$(D)_T=$(T).svg"
    #     output_path = joinpath(output_dir, "D=$D", "T=$T", output_filename)
    #     save(output_path, fig)

    #     # Plot min eigenvalue distribution
    #     #= @info "Plotting max eigenvalue distributions..." =#
    #     fig = Figure()
    #     ax = Axis(fig[1, 1],
    #         title=L"Maximum eigenvalue distribution $D = %$(D)$, $\tau = %$(tau)$",
    #         xlabel=L"\lambda",
    #         ylabel=L"max(\lambda)",
    #         limits=((0, nothing), (0, nothing)),
    #         yticks=make_ticks_log(0:5),
    #         yscale=Makie.pseudolog10)
    #     hist!(ax, vcat(map(last, eachcol(eigvals))...), bins=100)

    #     output_filename = global_prefix * "EigvalsMax_D=$(D)_T=$(T).svg"
    #     output_path = joinpath(output_dir, "D=$D", "T=$T", output_filename)
    #     save(output_path, fig)
    # end


end
