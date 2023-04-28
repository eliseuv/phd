# Dr Watson helper
using DrWatson
@quickactivate "phd"

# External libraries
using Logging, Statistics, PyCall, CSV, DataFrames, LinearAlgebra, LaTeXStrings, CairoMakie

# My libs
include("../../../src/Thesis.jl")
using .Thesis.Metaprogramming
using .Thesis.DataIO
using .Thesis.Stats

# Calculate fluctuations using histograms
@inline function hist_fluctuations(values::AbstractVector, nbins::Integer)
    hist = Histogram(values, nbins)
    mean_value = mean(hist)
    return (mean_value, var(hist, mean=mean_value))
end

@inline make_ticks_log(powers::AbstractVector{<:Real}, base::Integer=10) = (Float64(base) .^ powers, (map(x -> latexstring("$(base)^{$(x)}"), powers)))

py"""
import pickle

def load_pickle(fpath):
    with open(fpath, "rb") as f:
        data = pickle.load(f)
    return data
"""

@inline load_pickle = py"load_pickle"

@inline function get_critical_temperature(df_temperatures, D::Real)
    df_D_row = df_temperatures[only(findall(==(D), df_temperatures.anisotropy_field)), 2:end]
    transition_order = lowercase(string(df_D_row[:transition_order]))
    crit_temp_source = findfirst(!ismissing, df_D_row)
    T_c = df_D_row[crit_temp_source]
    return (T_c, transition_order, crit_temp_source)
end

# Where to take the data from
data_dirpath = datadir("blume_capel_pickles", "eigvals")
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

D_values = sort(collect(keys(eigvals_datafiles)))

# Load temperatures table
@info "Loading temperatures table..."
df_temperatures = DataFrame(CSV.File(projectdir("tables", "butera_and_pernici_2018", "blume-capel_square_lattice.csv")))

# Plot output root directory
output_root = plotsdir("blume_capel_pickles")

# Makie theme
my_theme = Theme(fontsize=24)
set_theme!(my_theme)

# # Figure size
# size_inches = (4, 3)
# size_pt = 72 .* size_inches

# D values considered
D_vals_second = [0.0, 0.5, 1.0, 1.5, 1.75, 1.8028, 1.9, 1.9501]
D_vals_tcp = [1.9658149, 1.96582, 1.96604]

#####################################
# Multiplot Eigenvalue Fluctuations #
#####################################

# fig_mean = Figure(resolution=(600, 800))
# axs_mean = [Axis(fig_mean[i, j],
#     limits=((nothing, nothing), (0.9, 1)),
#     xticks=[1, 2, 4, 6],
#     yticks=[0.9, 0.95, 1])
#             for i ∈ 1:4 for j ∈ 1:2]
# fig_var = Figure(resolution=(600, 800))
# axs_var = [Axis(fig_var[i, j],
#     limits=((nothing, nothing), (nothing, nothing)),
#     xticks=[1, 2, 4, 6],
#     yticks=[0, 20, 40, 60])
#            for i ∈ 1:4 for j ∈ 1:2]
# for (ax_mean, ax_var, D) ∈ zip(axs_mean, axs_var, D_vals_second)
#     D_dict = eigvals_datafiles[D]
#     T_c, transition_order, crit_temp_source = get_critical_temperature(df_temperatures, D)
#     @assert transition_order == "second"
#     transition_order_str = replace(transition_order, "first" => "First order", "second" => "Second order", "tcp" => "TCP")
#     crit_temp_source_str = replace(string(crit_temp_source), "_" => " ")
#     T_vec = sort(collect(keys(D_dict)))
#     tau_vec = map(T_vec ./ T_c) do x
#         round(x; digits=3)
#     end
#     map(ax -> ax.title = L"$D = %$(D)$", [ax_mean, ax_var])
#     # Loop on temperatures
#     mean_vec = similar(T_vec)
#     var_vec = similar(T_vec)
#     for (i, T) in enumerate(T_vec)
#         datafile = D_dict[T]
#         eigvals = vec(load_pickle(datafile.path))
#         mean_vec[i], var_vec[i] = hist_fluctuations(eigvals, 100)
#         # mean_vec[i] = mean(eigvals)
#         # var_vec[i] = var(eigvals)
#     end
#     scatter!(ax_mean, tau_vec, mean_vec)
#     scatter!(ax_var, tau_vec, var_vec)
# end
# xlabel = Label(fig_mean[5, 1:2], L"$\tau$")
# ylabel = Label(fig_mean[1:4, 0], L"$\langle \lambda \rangle$", rotation=pi / 2)
# save(joinpath(output_root, filename(global_prefix * "EigvalsMeans", "order" => "second"; ext="svg")), fig_mean)
# xlabel = Label(fig_var[5, 1:2], L"$\tau$")
# ylabel = Label(fig_var[1:4, 0], L"\langle \lambda^2 \rangle - \langle \lambda \rangle^2", rotation=pi / 2)
# save(joinpath(output_root, filename(global_prefix * "EigvalsVars", "order" => "second"; ext="svg")), fig_var)

#############################################
# Multiplot Eigenvalue Spacing Fluctuations #
#############################################

# fig_mean = Figure(resolution=(600, 800))
# axs_mean = [Axis(fig_mean[i, j],
#     limits=((nothing, nothing), (nothing, nothing)))
#             for i ∈ 1:4 for j ∈ 1:2]
# fig_var = Figure(resolution=(600, 800))
# axs_var = [Axis(fig_var[i, j],
#     limits=((nothing, nothing), (nothing, nothing)),
#     xticks=[1, 2, 4, 6])
#            for i ∈ 1:4 for j ∈ 1:2]
# for (ax_mean, ax_var, D) ∈ zip(axs_mean, axs_var, D_vals_second)
#     D_dict = eigvals_datafiles[D]
#     T_c, transition_order, crit_temp_source = get_critical_temperature(df_temperatures, D)
#     @assert transition_order == "second"
#     transition_order_str = replace(transition_order, "first" => "First order", "second" => "Second order", "tcp" => "TCP")
#     crit_temp_source_str = replace(string(crit_temp_source), "_" => " ")
#     T_vec = sort(collect(keys(D_dict)))
#     tau_vec = map(T_vec ./ T_c) do x
#         round(x; digits=3)
#     end
#     map(ax -> ax.title = L"$D = %$(D)$", [ax_mean, ax_var])
#     # Loop on temperatures
#     mean_vec = similar(T_vec)
#     var_vec = similar(T_vec)
#     for (i, T) in enumerate(T_vec)
#         datafile = D_dict[T]
#         eigvals = load_pickle(datafile.path)
#         eigvals_max_spacings = maximum(diff(eigvals, dims=1), dims=1)
#         # mean_vec[i], var_vec[i] = hist_fluctuations(eigvals_spacings, 100)
#         mean_vec[i] = mean(eigvals_max_spacings)
#         var_vec[i] = var(eigvals_max_spacings)
#     end
#     scatter!(ax_mean, tau_vec, mean_vec)
#     scatter!(ax_var, tau_vec, var_vec)
# end
# xlabel = Label(fig_mean[5, 1:2], L"$\tau$")
# ylabel = Label(fig_mean[1:4, 0], L"$\langle \max(\Delta\lambda) \rangle$", rotation=pi / 2)
# save(joinpath(output_root, filename(global_prefix * "EigvalMaxSpacingMeans", "order" => "second"; ext="svg")), fig_mean)
# xlabel = Label(fig_var[5, 1:2], L"$\tau$")
# ylabel = Label(fig_var[1:4, 0], L"\langle \lambda^2 \rangle - \langle \lambda \rangle^2", rotation=pi / 2)
# save(joinpath(output_root, filename(global_prefix * "EigvalMaxSpacingVars", "order" => "second"; ext="svg")), fig_var)

# Loop on anisotropy values
for (D, D_dict) ∈ sort(collect(eigvals_datafiles), by=x -> x[1])

    # Fetch critical temperature info
    T_c, transition_order, crit_temp_source = get_critical_temperature(df_temperatures, D)
    transition_order_str = replace(transition_order, "first" => "First order", "second" => "Second order", "tcp" => "TCP")
    crit_temp_source_str = replace(string(crit_temp_source), "_" => " ")
    T_vec = sort(collect(keys(D_dict)))
    tau_vec = map(T_vec ./ T_c) do x
        round(x; digits=3)
    end

    println("D = $D ($(transition_order_str))")

    # Create dir
    output_dir_D = joinpath(output_root, "D=$D($(transition_order))")
    mkpath(output_dir_D)

    #     @info "Plotting fluctuations..."
    #     # Loop on temperatures
    #     mean_vec = similar(T_vec)
    #     var_vec = similar(T_vec)
    #     for (i, T) in enumerate(T_vec)
    #         datafile = D_dict[T]
    #         eigvals = vec(load_pickle(datafile.path))
    #         mean_vec[i], var_vec[i] = hist_fluctuations(eigvals, 100)
    #         # mean_vec[i] = mean(eigvals)
    #         # var_vec[i] = var(eigvals)
    #     end

    #     # Plot eigenvalues mean
    #     fig = Figure()
    #     ax = Axis(fig[1, 1],
    #         title=L"Eigenvalues mean $D = %$(D)$",
    #         xlabel=L"\tau", ylabel=L"\langle \lambda \rangle",
    #         xticks=0:0.5:6.5)
    #     scatter!(ax, tau_vec, mean_vec)
    #     save(joinpath(output_dir_D, filename(global_prefix * "EigvalsMean", @varsdict(D); ext="svg")), fig)

    #     # Plot eigenvalues variance
    #     fig = Figure()
    #     ax = Axis(fig[1, 1],
    #         title=L"Eigenvalues variance $D = %$(D)$",
    #         xlabel=L"\tau", ylabel=L"\langle \lambda^2 \rangle - \langle \lambda \rangle^2",
    #         xticks=0:0.5:6.5)
    #     scatter!(ax, tau_vec, var_vec)
    #     save(joinpath(output_dir_D, filename(global_prefix * "EigvalsVar", @varsdict(D); ext="svg")), fig)

    # # Plot eigenvalues spacing fluctuations
    # # Loop on temperatures
    # mean_vec = similar(T_vec)
    # var_vec = similar(T_vec)
    # for (i, T) in enumerate(T_vec)
    #     datafile = D_dict[T]
    #     eigvals_spacings = diff(load_pickle(datafile.path), dims=1)
    #     # col_means_inv = mean(eigvals_spacings, dims=1)
    #     # eigvals_spacings = col_means_inv .\ eigvals_spacings
    #     mean_vec[i], var_vec[i] = hist_fluctuations(vcat(eigvals_spacings...), 100)
    #     # mean_vec[i] = mean(eigvals_spacings)
    #     # var_vec[i] = var(eigvals_spacings, mean=mean_vec[i])
    # end

    # # Plot eigenvalues spacing mean
    # fig = Figure()
    # ax = Axis(fig[1, 1],
    #     title=L"Eigenvalues spacing mean $D = %$(D)$",
    #     xlabel=L"\tau", ylabel=L"\langle\Delta\lambda\rangle",
    #     xticks=0:0.5:6.5)
    # scatter!(ax, tau_vec, mean_vec)
    # save(joinpath(output_dir_D, filename(global_prefix * "EigvalsSpacingMean", @varsdict(D); ext="svg")), fig)

    # # Plot eigenvalues spacing variance
    # fig = Figure()
    # ax = Axis(fig[1, 1],
    #     title=L"Eigenvalues spacing variance $D = %$(D)$",
    #     xlabel=L"\tau", ylabel=L"\langle \Delta\lambda^2 \rangle - \langle \Delta\lambda \rangle^2",
    #     xticks=0:0.5:6.5)
    # scatter!(ax, tau_vec, var_vec)
    # save(joinpath(output_dir_D, filename(global_prefix * "EigvalsSpacingVar", @varsdict(D); ext="svg")), fig)

    # Plot eigenvalues mean max spacing
    # Loop on temperatures
    mean_vec = similar(T_vec)
    var_vec = similar(T_vec)
    for (i, T) in enumerate(T_vec)
        datafile = D_dict[T]
        eigvals_spacings = diff(load_pickle(datafile.path), dims=1)
        # col_means = mean(eigvals_spacings, dims=1)
        # eigvals_spacings = col_means .\ eigvals_spacings
        eigvals_max_spacings = maximum(eigvals_spacings, dims=1)
        # mean_vec[i], var_vec[i] = hist_fluctuations(vcat(eigvals_max_spacings...), 100)
        mean_vec[i] = mean(eigvals_max_spacings)
        var_vec[i] = var(eigvals_max_spacings)
    end

    # Plot mean maximum eigenvalues spacing mean
    fig = Figure()
    ax = Axis(fig[1, 1],
        title=L"Eigenvalues mean maximum spacing $D = %$(D)$",
        xlabel=L"\tau", ylabel=L"\langle \max(\Delta\lambda) \rangle",
        xticks=0:0.5:6.5)
    scatter!(ax, tau_vec, mean_vec)
    save(joinpath(output_dir_D, filename(global_prefix * "EigvalsMaxSpacingMean", @varsdict(D); ext="svg")), fig)

    # Plot var maximum eigenvalues spacing mean
    fig = Figure()
    ax = Axis(fig[1, 1],
        title=L"Eigenvalues maximum spacing variance $D = %$(D)$",
        xlabel=L"\tau", ylabel=L"\langle \max(\Delta\lambda) \rangle",
        xticks=0:0.5:6.5)
    scatter!(ax, tau_vec, var_vec)
    save(joinpath(output_dir_D, filename(global_prefix * "EigvalsMaxSpacingVar", @varsdict(D); ext="svg")), fig)

    #     @info "Plotting combined histograms..."
    #     idxs = [1, 4, 5, 6, 8, 10, 11, 19, 21]
    #     println("tau = ", getindex(tau_vec, idxs))

    #     # Eigenvalues distribution
    #     fig = Figure()
    #     axs = [Axis(fig[i, j],
    #         limits=((0, nothing), (0, nothing)),
    #         yticks=make_ticks_log(0:5),
    #         yscale=Makie.pseudolog10)
    #            for i ∈ 1:3 for j ∈ 1:3]
    #     for (ax, idx) ∈ zip(axs, idxs)
    #         T = T_vec[idx]
    #         tau = round(T / T_c; digits=3)
    #         ax.title = L"$\tau = %$(tau)$"
    #         datafile = D_dict[T]
    #         eigvals = load_pickle(datafile.path)
    #         hist!(ax, vec(eigvals), bins=100)
    #     end
    #     xlabel = Label(fig[4, 1:3], L"$\lambda$")
    #     ylabel = Label(fig[1:3, 0], L"$\rho(\lambda)$", rotation=pi / 2)
    #     save(joinpath(output_dir_D, filename(global_prefix * "EigvalsHist", @varsdict(D); ext="svg")), fig)

    #     # Eigenvalues spacing distributions
    #     fig = Figure()
    #     axs = [Axis(fig[i, j],
    #         limits=((0, nothing), (0, nothing)),
    #         yticks=make_ticks_log(0:5),
    #         yscale=Makie.pseudolog10)
    #            for i ∈ 1:3 for j ∈ 1:3]
    #     for (ax, idx) ∈ zip(axs, idxs)
    #         T = T_vec[idx]
    #         tau = round(T / T_c; digits=3)
    #         ax.title = L"$\tau = %$(tau)$"
    #         datafile = D_dict[T]
    #         eigvals = load_pickle(datafile.path)
    #         eigvals_spacings = vcat(map(eachcol(eigvals)) do col
    #             spacings = diff(col)
    #             return spacings ./ mean(spacings)
    #         end...)
    #         hist!(ax, eigvals_spacings ./ mean(eigvals_spacings), bins=100)
    #     end
    #     xlabel = Label(fig[4, 1:3], L"$\lambda$")
    #     ylabel = Label(fig[1:3, 0], L"$\rho(\Delta\lambda)$", rotation=pi / 2)
    #     save(joinpath(output_dir_D, filename(global_prefix * "EigvalsSpacingHist", @varsdict(D); ext="svg")), fig)

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
