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

# @inline make_ticks_log(powers::AbstractVector{<:Real}, base::Integer=10) = (Float64(base) .^ powers, (map(x -> latexstring("$(base)^{$(x)}"), powers)))
@inline make_ticks_log(powers::AbstractVector{<:Real}, base::Integer=10) = (powers, (map(x -> latexstring("$(base)^{$(x)}"), powers)))
@inline make_ticks(powers::AbstractVector{<:Real}, base::Integer=10) = (Float64(base) .^ powers, (map(x -> latexstring("$(base)^{$(x)}"), powers)))

@inline function axis_ticks_int_range(low::Real, high::Real, length::Integer)
    x_final = floor(Int64, high)
    tick_vals = collect(ceil(Int64, low):ceil(Int64, (high - low) / length):floor(Int64, high))
    if tick_vals[end] != x_final
        push!(tick_vals, x_final)
    end
    return (tick_vals, map(tck -> latexstring("$(tck)"), tick_vals))
end

@inline function axis_ticks_range(low::Real, high::Real, length::Integer)
    tick_vals = map(x -> round(x, digits=5), range(low, high, length=length))
    return (tick_vals, map(tck -> latexstring("$(tck)"), tick_vals))
end

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

@inline get_spacings(eigvals_matrix::AbstractMatrix{<:Real}) = diff(eigvals_matrix, dims=2)
@inline function get_normalized_spacings(eigvals_matrix::AbstractMatrix{<:Real})
    eigvals_spacings = get_spacings(eigvals_matrix)
    eigvals_spacings_means = mean(eigvals_spacings, dims=2)
    return eigvals_spacings_means .\ eigvals_spacings
end

# Where to take the data from
data_dirpath = datadir("blume_capel_pickles", "eigvals")
@show data_dirpath

const global_prefix = "BlumeCapelSq2D"

# Parse datafiles into dicts
# Eigenvalues pickle stores a (n_runs x n_samples) matrix each row containing a set of eigenvalues
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
D_vals_first = [1.96820, 1.97308, 1.9777, 1.98142, 1.98490, 1.994232, 1.99681357, 1.99842103, 1.99932488]

# #####################################
# # Multiplot Eigenvalue Fluctuations #
# #####################################

# # Second order
# @info "Plotting eigenvalues fluctuations (Second order)..."
# fig_mean = Figure(resolution=(600, 900))
# axs_mean = [Axis(fig_mean[i, j],
#     limits=((nothing, nothing), (0.83, 1)),
#     xticks=[1, 2, 4, 6],
#     yticks=[0.83, 0.9, 1])
#             for i ∈ 1:4 for j ∈ 1:2]
# fig_var = Figure(resolution=(600, 900))
# axs_var = [Axis(fig_var[i, j],
#     limits=((nothing, nothing), (0, nothing)),
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
#     scatterlines!(ax_mean, tau_vec, mean_vec)
#     scatterlines!(ax_var, tau_vec, var_vec)
# end
# Label(fig_mean[0, :], "Mean eigenvalue (second order)")
# Label(fig_mean[5, 1:2], L"$T/T_c$")
# Label(fig_mean[1:4, 0], L"$\langle \lambda \rangle$", rotation=pi / 2)
# save(joinpath(output_root, filename(global_prefix * "EigvalsMeans", "order" => "second"; ext="svg")), fig_mean)
# Label(fig_var[0, :], "Eigenvalues variance (second order)")
# Label(fig_var[5, 1:2], L"$T/T_c$")
# Label(fig_var[1:4, 0], L"\langle \lambda^2 \rangle - \langle \lambda \rangle^2", rotation=pi / 2)
# save(joinpath(output_root, filename(global_prefix * "EigvalsVars", "order" => "second"; ext="svg")), fig_var)

# # First order
# @info "Plotting eigenvalues fluctuations (First order)..."
# fig_mean = Figure(resolution=(600, 900))
# axs_mean = [Axis(fig_mean[i, j],
#     limits=((nothing, nothing), (0.8, 1)),
#     xticks=[1, 2, 4, 6],
#     yticks=[0.8, 0.9, 1])
#             for i ∈ 1:4 for j ∈ 1:2]
# fig_var = Figure(resolution=(600, 900))
# axs_var = [Axis(fig_var[i, j],
#     limits=((nothing, nothing), (0, nothing)),
#     xticks=[1, 2, 4, 6],
#     yticks=[0, 10, 20])
#            for i ∈ 1:4 for j ∈ 1:2]
# for (ax_mean, ax_var, D) ∈ zip(axs_mean, axs_var, D_vals_first)
#     D_dict = eigvals_datafiles[D]
#     T_c, transition_order, crit_temp_source = get_critical_temperature(df_temperatures, D)
#     @assert transition_order == "first"
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
#     scatterlines!(ax_mean, tau_vec, mean_vec)
#     scatterlines!(ax_var, tau_vec, var_vec)
# end
# Label(fig_mean[0, :], "Mean eigenvalue (first order)")
# Label(fig_mean[5, 1:2], L"$T/T_c$")
# Label(fig_mean[1:4, 0], L"$\langle \lambda \rangle$", rotation=pi / 2)
# save(joinpath(output_root, filename(global_prefix * "EigvalsMeans", "order" => "first"; ext="svg")), fig_mean)
# Label(fig_var[0, :], "Eigenvalues variance (first order)")
# Label(fig_var[5, 1:2], L"$T/T_c$")
# Label(fig_var[1:4, 0], L"\langle \lambda^2 \rangle - \langle \lambda \rangle^2", rotation=pi / 2)
# save(joinpath(output_root, filename(global_prefix * "EigvalsVars", "order" => "first"; ext="svg")), fig_var)

# #############################################
# # Multiplot Extreme Eigenvalue Fluctuations #
# #############################################

# # Second order
# @info "Plotting minimum eigenvalues fluctuations (Second order)..."
# fig_mean = Figure(resolution=(600, 900))
# axs_mean = [Axis(fig_mean[i, j],
#     limits=((nothing, nothing), (nothing, nothing)),
#     xticks=[1, 2, 4, 6],
#     yticks=make_ticks(-5:0),
#     yscale=log10)
#             for i ∈ 1:4 for j ∈ 1:2]
# fig_var = Figure(resolution=(600, 900))
# axs_var = [Axis(fig_var[i, j],
#     limits=((nothing, nothing), (nothing, nothing)),
#     xticks=[1, 2, 4, 6],
#     yticks=make_ticks(-10:2:0),
#     yscale=log10)
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
#         eigvals_matrix = load_pickle(datafile.path)
#         eigvals_min = vec(eigvals_matrix[:, begin])
#         # mean_vec[i], var_vec[i] = hist_fluctuations(eigvals, 100)
#         mean_vec[i] = mean(eigvals_min)
#         var_vec[i] = var(eigvals_min)
#     end
#     scatterlines!(ax_mean, tau_vec, mean_vec)
#     scatterlines!(ax_var, tau_vec, var_vec)
# end
# Label(fig_mean[0, :], "Mean minimum eigenvalue (second order)")
# Label(fig_mean[5, 1:2], L"$T/T_c$")
# Label(fig_mean[1:4, 0], L"$\langle \lambda_{min} \rangle$", rotation=pi / 2)
# save(joinpath(output_root, filename(global_prefix * "MinEigvalMean", "order" => "second"; ext="svg")), fig_mean)
# Label(fig_var[0, :], "Minimum eigenvalue variance (second order)")
# Label(fig_var[5, 1:2], L"$T/T_c$")
# Label(fig_var[1:4, 0], L"\langle \lambda_{min}^2 \rangle - \langle \lambda_{min} \rangle^2", rotation=pi / 2)
# save(joinpath(output_root, filename(global_prefix * "MinEigvalVar", "order" => "second"; ext="svg")), fig_var)

# # First order
# @info "Plotting minimum eigenvalues fluctuations (First order)..."
# fig_mean = Figure(resolution=(600, 900))
# axs_mean = [Axis(fig_mean[i, j],
#     limits=((nothing, nothing), (nothing, nothing)),
#     xticks=[1, 2, 4, 6],
#     yticks=make_ticks(-5:0),
#     yscale=log10)
#             for i ∈ 1:4 for j ∈ 1:2]
# fig_var = Figure(resolution=(600, 900))
# axs_var = [Axis(fig_var[i, j],
#     limits=((nothing, nothing), (nothing, nothing)),
#     xticks=[1, 2, 4, 6],
#     yticks=make_ticks(-10:2:0),
#     yscale=log10)
#            for i ∈ 1:4 for j ∈ 1:2]
# for (ax_mean, ax_var, D) ∈ zip(axs_mean, axs_var, D_vals_first)
#     D_dict = eigvals_datafiles[D]
#     T_c, transition_order, crit_temp_source = get_critical_temperature(df_temperatures, D)
#     @assert transition_order == "first"
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
#         eigvals_matrix = load_pickle(datafile.path)
#         eigvals_min = vec(eigvals_matrix[:, begin])
#         # mean_vec[i], var_vec[i] = hist_fluctuations(eigvals, 100)
#         mean_vec[i] = mean(eigvals_min)
#         var_vec[i] = var(eigvals_min)
#     end
#     scatterlines!(ax_mean, tau_vec, mean_vec)
#     scatterlines!(ax_var, tau_vec, var_vec)
# end
# Label(fig_mean[0, :], "Mean minimum eigenvalue (first order)")
# Label(fig_mean[5, 1:2], L"$T/T_c$")
# Label(fig_mean[1:4, 0], L"$\langle \lambda_{min} \rangle$", rotation=pi / 2)
# save(joinpath(output_root, filename(global_prefix * "MinEigvalMean", "order" => "first"; ext="svg")), fig_mean)
# Label(fig_var[0, :], "Minimum eigenvalues variance (first order)")
# Label(fig_var[5, 1:2], L"$T/T_c$")
# Label(fig_var[1:4, 0], L"\langle \lambda_{min}^2 \rangle - \langle \lambda_{min} \rangle^2", rotation=pi / 2)
# save(joinpath(output_root, filename(global_prefix * "MinEigvalVar", "order" => "first"; ext="svg")), fig_var)

# # Second order
# @info "Plotting maximum eigenvalues fluctuations (Second order)..."
# fig_mean = Figure(resolution=(600, 900))
# axs_mean = [Axis(fig_mean[i, j],
#     limits=((nothing, nothing), (nothing, nothing)),
#     xticks=[1, 2, 4, 6],
#     yticks=make_ticks(-5:5),
#     yscale=log10)
#             for i ∈ 1:4 for j ∈ 1:2]
# fig_var = Figure(resolution=(600, 900))
# axs_var = [Axis(fig_var[i, j],
#     limits=((nothing, nothing), (nothing, nothing)),
#     xticks=[1, 2, 4, 6],
#     yticks=make_ticks(-5:5),
#     yscale=log10)
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
#         eigvals_matrix = load_pickle(datafile.path)
#         eigvals_max = vec(eigvals_matrix[:, end])
#         # mean_vec[i], var_vec[i] = hist_fluctuations(eigvals, 100)
#         mean_vec[i] = mean(eigvals_max)
#         var_vec[i] = var(eigvals_max)
#     end
#     scatterlines!(ax_mean, tau_vec, mean_vec)
#     scatterlines!(ax_var, tau_vec, var_vec)
# end
# Label(fig_mean[0, :], "Mean maximum eigenvalue (second order)")
# Label(fig_mean[5, 1:2], L"$T/T_c$")
# Label(fig_mean[1:4, 0], L"$\langle \lambda_{max} \rangle$", rotation=pi / 2)
# save(joinpath(output_root, filename(global_prefix * "MaxEigvalMean", "order" => "second"; ext="svg")), fig_mean)
# Label(fig_var[0, :], "Maximum eigenvalue variance (second order)")
# Label(fig_var[5, 1:2], L"$T/T_c$")
# Label(fig_var[1:4, 0], L"\langle \lambda_{max}^2 \rangle - \langle \lambda_{max} \rangle^2", rotation=pi / 2)
# save(joinpath(output_root, filename(global_prefix * "MaxEigvalVar", "order" => "second"; ext="svg")), fig_var)

# # First order
# @info "Plotting maximum eigenvalues fluctuations (First order)..."
# fig_mean = Figure(resolution=(600, 900))
# axs_mean = [Axis(fig_mean[i, j],
#     limits=((nothing, nothing), (nothing, nothing)),
#     xticks=[1, 2, 4, 6],
#     yticks=make_ticks(-5:5),
#     yscale=log10)
#             for i ∈ 1:4 for j ∈ 1:2]
# fig_var = Figure(resolution=(600, 900))
# axs_var = [Axis(fig_var[i, j],
#     limits=((nothing, nothing), (nothing, nothing)),
#     xticks=[1, 2, 4, 6],
#     yticks=make_ticks(-5:5),
#     yscale=log10)
#            for i ∈ 1:4 for j ∈ 1:2]
# for (ax_mean, ax_var, D) ∈ zip(axs_mean, axs_var, D_vals_first)
#     D_dict = eigvals_datafiles[D]
#     T_c, transition_order, crit_temp_source = get_critical_temperature(df_temperatures, D)
#     @assert transition_order == "first"
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
#         eigvals_matrix = load_pickle(datafile.path)
#         eigvals_max = vec(eigvals_matrix[:, end])
#         # mean_vec[i], var_vec[i] = hist_fluctuations(eigvals, 100)
#         mean_vec[i] = mean(eigvals_max)
#         var_vec[i] = var(eigvals_max)
#     end
#     scatterlines!(ax_mean, tau_vec, mean_vec)
#     scatterlines!(ax_var, tau_vec, var_vec)
# end
# Label(fig_mean[0, :], "Mean maximum eigenvalue (first order)")
# Label(fig_mean[5, 1:2], L"$T/T_c$")
# Label(fig_mean[1:4, 0], L"$\langle \lambda_{max} \rangle$", rotation=pi / 2)
# save(joinpath(output_root, filename(global_prefix * "MaxEigvalMean", "order" => "first"; ext="svg")), fig_mean)
# Label(fig_var[0, :], "Maximum eigenvalues variance (first order)")
# Label(fig_var[5, 1:2], L"$T/T_c$")
# Label(fig_var[1:4, 0], L"\langle \lambda_{max}^2 \rangle - \langle \lambda_{max} \rangle^2", rotation=pi / 2)
# save(joinpath(output_root, filename(global_prefix * "MaxEigvalVar", "order" => "first"; ext="svg")), fig_var)

# #############################################
# # Multiplot Eigenvalue Spacing Fluctuations #
# #############################################

# # Second order
# @info "Plotting eigenvalue spacings fluctuations (Second order)..."
# fig_mean = Figure(resolution=(600, 900),
#     title="Mean eigenvalue spacings (second order)")
# axs_mean = [Axis(fig_mean[i, j],
#     limits=((nothing, nothing), (0, 0.8)),
#     xticks=[1, 2, 4, 6],
#     yticks=[0, 0.4, 0.8])
#             for i ∈ 1:4 for j ∈ 1:2]
# fig_var = Figure(resolution=(600, 900),
#     title="Eigenvalue spacings variance (second order)")
# axs_var = [Axis(fig_var[i, j],
#     limits=((nothing, nothing), (0, 60)),
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
#         eigvals_matrix = load_pickle(datafile.path)
#         eigvals_spacings = vec(get_spacings(eigvals_matrix))
#         mean_vec[i], var_vec[i] = hist_fluctuations(eigvals_spacings, 100)
#         # mean_vec[i] = mean(eigvals)
#         # var_vec[i] = var(eigvals)
#     end
#     scatterlines!(ax_mean, tau_vec, mean_vec)
#     scatterlines!(ax_var, tau_vec, var_vec)
# end
# Label(fig_mean[0, :], "Mean eigenvalue spacing (second order)")
# Label(fig_mean[5, 1:2], L"$T/T_c$")
# Label(fig_mean[1:4, 0], L"$\langle \Delta\lambda \rangle$", rotation=pi / 2)
# save(joinpath(output_root, filename(global_prefix * "EigvalSpacingMeans", "order" => "second"; ext="svg")), fig_mean)
# Label(fig_var[0, :], "Eigenvalue spacing variance (second order)")
# Label(fig_var[5, 1:2], L"$T/T_c$")
# Label(fig_var[1:4, 0], L"\langle ( \Delta\lambda )^2 \rangle - \langle \Delta\lambda \rangle^2", rotation=pi / 2)
# save(joinpath(output_root, filename(global_prefix * "EigvalSpacingVars", "order" => "second"; ext="svg")), fig_var)

# # First order
# @info "Plotting eigenvalue spacings fluctuations (First order)..."
# fig_mean = Figure(resolution=(600, 900),
#     title="Mean eigenvalue spacings (first order)")
# axs_mean = [Axis(fig_mean[i, j],
#     limits=((nothing, nothing), (0, 0.6)),
#     xticks=[1, 2, 4, 6],
#     yticks=[0, 0.3, 0.6])
#             for i ∈ 1:4 for j ∈ 1:2]
# fig_var = Figure(resolution=(600, 900),
#     title="Eigenvalue spacings variance (first order)")
# axs_var = [Axis(fig_var[i, j],
#     limits=((nothing, nothing), (0, 15)),
#     xticks=[1, 2, 4, 6],
#     yticks=[0, 7, 15])
#            for i ∈ 1:4 for j ∈ 1:2]
# for (ax_mean, ax_var, D) ∈ zip(axs_mean, axs_var, D_vals_first)
#     D_dict = eigvals_datafiles[D]
#     T_c, transition_order, crit_temp_source = get_critical_temperature(df_temperatures, D)
#     @assert transition_order == "first"
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
#         eigvals_matrix = load_pickle(datafile.path)
#         eigvals_spacings = vec(get_spacings(eigvals_matrix))
#         mean_vec[i], var_vec[i] = hist_fluctuations(eigvals_spacings, 100)
#         # mean_vec[i] = mean(eigvals)
#         # var_vec[i] = var(eigvals)
#     end
#     scatterlines!(ax_mean, tau_vec, mean_vec)
#     scatterlines!(ax_var, tau_vec, var_vec)
# end
# Label(fig_mean[0, :], "Mean eigenvalue spacing (first order)")
# Label(fig_mean[5, 1:2], L"$T/T_c$")
# Label(fig_mean[1:4, 0], L"$\langle \Delta\lambda \rangle$", rotation=pi / 2)
# save(joinpath(output_root, filename(global_prefix * "EigvalSpacingMeans", "order" => "first"; ext="svg")), fig_mean)
# Label(fig_var[0, :], "Eigenvalue spacing variance (first order)")
# Label(fig_var[5, 1:2], L"$T/T_c$")
# Label(fig_var[1:4, 0], L"\langle ( \Delta\lambda )^2 \rangle - \langle \Delta\lambda \rangle^2", rotation=pi / 2)
# save(joinpath(output_root, filename(global_prefix * "EigvalSpacingVars", "order" => "first"; ext="svg")), fig_var)

# #####################################################
# # Multiplot Maximum Eigenvalue Spacing Fluctuations #
# #####################################################

# # Second order
# @info "Plotting eigenvalues maximum spacing fluctuations (Second order)..."
# fig_mean = Figure(resolution=(600, 900))
# axs_mean = [Axis(fig_mean[i, j],
#     limits=((nothing, nothing), (0, nothing)),
#     xticks=[1, 2, 4, 6])
#             for i ∈ 1:4 for j ∈ 1:2]
# fig_var = Figure(resolution=(600, 900))
# axs_var = [Axis(fig_var[i, j],
#     limits=((nothing, nothing), (0, nothing)),
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
#         eigvals_matrix = load_pickle(datafile.path)
#         eigvals_spacings = get_normalized_spacings(eigvals_matrix)
#         eigvals_spacings_max = maximum(eigvals_spacings, dims=2)
#         # mean_vec[i], var_vec[i] = hist_fluctuations(eigvals_spacings, 100)
#         mean_vec[i] = mean(eigvals_spacings_max)
#         var_vec[i] = var(eigvals_spacings_max)
#     end
#     scatterlines!(ax_mean, tau_vec, mean_vec)
#     scatterlines!(ax_var, tau_vec, var_vec)
# end

# Label(fig_mean[0, :], text="Mean maximum eigenvalue spacing (second order)", fontsize=25)
# Label(fig_mean[5, 1:2], L"$T/T_c$")
# Label(fig_mean[1:4, 0], L"$\langle \max(\Delta\lambda) \rangle$", rotation=pi / 2)
# save(joinpath(output_root, filename(global_prefix * "EigvalMaxSpacingMeans", "order" => "second"; ext="svg")), fig_mean)
# Label(fig_var[0, :], text="Maximum eigenvalue spacings variance (second order)", fontsize=25)
# Label(fig_var[5, 1:2], L"$T/T_c$")
# Label(fig_var[1:4, 0], L"\langle \max(\Delta\lambda)^2 \rangle - \langle \max(\Delta\lambda) \rangle^2", rotation=pi / 2)
# save(joinpath(output_root, filename(global_prefix * "EigvalMaxSpacingVars", "order" => "second"; ext="svg")), fig_var)

# # First order
# @info "Plotting eigenvalues maximum spacing fluctuations (First order)..."
# fig_mean = Figure(resolution=(600, 900))
# axs_mean = [Axis(fig_mean[i, j],
#     limits=((nothing, nothing), (0, nothing)),
#     xticks=[1, 2, 4, 6])
#             for i ∈ 1:4 for j ∈ 1:2]
# fig_var = Figure(resolution=(600, 900))
# axs_var = [Axis(fig_var[i, j],
#     limits=((nothing, nothing), (0, nothing)),
#     xticks=[1, 2, 4, 6])
#            for i ∈ 1:4 for j ∈ 1:2]
# for (ax_mean, ax_var, D) ∈ zip(axs_mean, axs_var, D_vals_first)
#     D_dict = eigvals_datafiles[D]
#     T_c, transition_order, crit_temp_source = get_critical_temperature(df_temperatures, D)
#     @assert transition_order == "first"
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
#         eigvals_matrix = load_pickle(datafile.path)
#         eigvals_spacings = get_normalized_spacings(eigvals_matrix)
#         eigvals_spacings_max = maximum(eigvals_spacings, dims=2)
#         # mean_vec[i], var_vec[i] = hist_fluctuations(eigvals_spacings, 100)
#         mean_vec[i] = mean(eigvals_spacings_max)
#         var_vec[i] = var(eigvals_spacings_max)
#     end
#     scatterlines!(ax_mean, tau_vec, mean_vec)
#     scatterlines!(ax_var, tau_vec, var_vec)
# end
# Label(fig_mean[0, :], text="Mean maximum eigenvalue spacing (first order)", fontsize=25)
# Label(fig_mean[5, 1:2], L"$T/T_c$")
# Label(fig_mean[1:4, 0], L"$\langle \max(\Delta\lambda) \rangle$", rotation=pi / 2)
# save(joinpath(output_root, filename(global_prefix * "EigvalMaxSpacingMeans", "order" => "first"; ext="svg")), fig_mean)
# Label(fig_var[0, :], text="Maximum eigenvalue spacings variance (first order)", fontsize=25)
# Label(fig_var[5, 1:2], L"$T/T_c$")
# Label(fig_var[1:4, 0], L"\langle \max(\Delta\lambda)^2 \rangle - \langle \max(\Delta\lambda) \rangle^2", rotation=pi / 2)
# save(joinpath(output_root, filename(global_prefix * "EigvalMaxSpacingVars", "order" => "first"; ext="svg")), fig_var)

#################################
# Single Anisotropy Value Plots #
#################################

# Loop on anisotropy values
for (D, D_dict) ∈ sort(collect(eigvals_datafiles), by=x -> x[1])

    # if D ∉ [D_vals_second..., D_vals_first..., D_vals_tcp[2]]
    #     continue
    # end

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

    # #
    # T_max = T_vec[end]
    # tau_max = tau_vec[end]
    # fig = Figure(resolution=(800, 600))
    # ax = Axis(fig[1, 1],
    #     title=L"$D = %$(D)$, $T/T_c = %$(tau_max)$",
    #     xlabel=L"s",
    #     ylabel=L"\ln{\frac{p(s)}{s}}")
    # datafile = D_dict[T_max]
    # eigvals_matrix = load_pickle(datafile.path)
    # eigvals_spacings = vec(get_normalized_spacings(eigvals_matrix))
    # hist = Histogram(eigvals_spacings, 100)
    # s, counts = hist_coords(hist)
    # const_log = log(sum(counts))
    # z = map(zip(s, counts)) do (x, y)
    #     if y == 0
    #         missing
    #     else
    #         log(y) - const_log
    #     end
    # end
    # # lines!(ax, 0:0.1:15, x -> log(π / 2) - (π / 4) * (x^2))
    # scatterlines!(ax, s, z)
    # save(joinpath(output_dir_D, filename(global_prefix * "EigvalsSpacingHist", @varsdict(D, tau_max); ext="svg")), fig)

    # @info "Plotting fluctuations..."
    # # Loop on temperatures
    # mean_vec = similar(T_vec)
    # var_vec = similar(T_vec)
    # for (i, T) in enumerate(T_vec)
    #     datafile = D_dict[T]
    #     eigvals = vec(load_pickle(datafile.path))
    #     mean_vec[i], var_vec[i] = hist_fluctuations(eigvals, 100)
    #     # mean_vec[i] = mean(eigvals)
    #     # var_vec[i] = var(eigvals)
    # end
    # # Plot eigenvalues mean
    # fig = Figure()
    # ax = Axis(fig[1, 1],
    #     title=L"Eigenvalues mean $D = %$(D)$",
    #     xlabel=L"$T/T_c$", ylabel=L"\langle \lambda \rangle",
    #     xticks=0:0.5:6.5)
    # scatterlines!(ax, tau_vec, mean_vec)
    # save(joinpath(output_dir_D, filename(global_prefix * "EigvalsMean", @varsdict(D); ext="svg")), fig)
    # # Plot eigenvalues variance
    # fig = Figure()
    # ax = Axis(fig[1, 1],
    #     title=L"Eigenvalues variance $D = %$(D)$",
    #     xlabel=L"$T/T_c$", ylabel=L"\langle \lambda^2 \rangle - \langle \lambda \rangle^2",
    #     xticks=0:0.5:6.5)
    # scatterlines!(ax, tau_vec, var_vec)
    # save(joinpath(output_dir_D, filename(global_prefix * "EigvalsVar", @varsdict(D); ext="svg")), fig)

    # @info "Plotting eigenvalue extrema fluctuations..."
    # # Loop on temperatures
    # mean_vec = similar(T_vec)
    # var_vec = similar(T_vec)
    # for (i, T) in enumerate(T_vec)
    #     datafile = D_dict[T]
    #     eigvals_matrix = load_pickle(datafile.path)
    #     eigvals_min = vec(eigvals_matrix[:, begin])
    #     # mean_vec[i], var_vec[i] = hist_fluctuations(eigvals, 100)
    #     mean_vec[i] = mean(eigvals_min)
    #     var_vec[i] = var(eigvals_min)
    # end
    # # Plot eigenvalues mean
    # fig = Figure()
    # ax = Axis(fig[1, 1],
    #     title=L"Mean minimum eigenvalue $D = %$(D)$",
    #     xlabel=L"$T/T_c$", ylabel=L"\langle \lambda_{min} \rangle",
    #     xticks=0:0.5:6.5,
    #     yticks=make_ticks(-5:0),
    #     yscale=log10)
    # scatterlines!(ax, tau_vec, mean_vec)
    # save(joinpath(output_dir_D, filename(global_prefix * "MinEigvalsMean", @varsdict(D); ext="svg")), fig)
    # # Plot eigenvalues variance
    # fig = Figure()
    # ax = Axis(fig[1, 1],
    #     title=L"Minimum eigenvalue variance $D = %$(D)$",
    #     xlabel=L"$T/T_c$", ylabel=L"\langle \lambda_{min}^2 \rangle - \langle \lambda_{min} \rangle^2",
    #     xticks=0:0.5:6.5,
    #     yticks=make_ticks(-5:0),
    #     yscale=log10)
    # scatterlines!(ax, tau_vec, var_vec)
    # save(joinpath(output_dir_D, filename(global_prefix * "MinEigvalVar", @varsdict(D); ext="svg")), fig)

    # # Loop on temperatures
    # mean_vec = similar(T_vec)
    # var_vec = similar(T_vec)
    # for (i, T) in enumerate(T_vec)
    #     datafile = D_dict[T]
    #     eigvals_matrix = load_pickle(datafile.path)
    #     eigvals_max = vec(eigvals_matrix[:, end])
    #     # mean_vec[i], var_vec[i] = hist_fluctuations(eigvals, 100)
    #     mean_vec[i] = mean(eigvals_max)
    #     var_vec[i] = var(eigvals_max)
    # end
    # # Plot eigenvalues mean
    # fig = Figure()
    # ax = Axis(fig[1, 1],
    #     title=L"Mean maximum eigenvalue $D = %$(D)$",
    #     xlabel=L"$T/T_c$", ylabel=L"\langle \lambda_{max} \rangle",
    #     xticks=0:0.5:6.5,
    #     yticks=make_ticks(-5:5),
    #     yscale=log10)
    # scatterlines!(ax, tau_vec, mean_vec)
    # save(joinpath(output_dir_D, filename(global_prefix * "MaxEigvalsMean", @varsdict(D); ext="svg")), fig)
    # # Plot eigenvalues variance
    # fig = Figure()
    # ax = Axis(fig[1, 1],
    #     title=L"Maximum eigenvalue variance $D = %$(D)$",
    #     xlabel=L"$T/T_c$", ylabel=L"\langle \lambda_{max}^2 \rangle - \langle \lambda_{max} \rangle^2",
    #     xticks=0:0.5:6.5,
    #     yticks=make_ticks(-5:5),
    #     yscale=log10)
    # scatterlines!(ax, tau_vec, var_vec)
    # save(joinpath(output_dir_D, filename(global_prefix * "MaxEigvalVar", @varsdict(D); ext="svg")), fig)

    # # Plot eigenvalues spacing fluctuations
    # # Loop on temperatures
    # mean_vec = similar(T_vec)
    # var_vec = similar(T_vec)
    # for (i, T) in enumerate(T_vec)
    #     datafile = D_dict[T]
    #     eigvals_matrix = load_pickle(datafile.path)
    #     eigvals_spacings = vec(get_spacings(eigvals_matrix))
    #     mean_vec[i], var_vec[i] = hist_fluctuations(eigvals_spacings, 100)
    #     # mean_vec[i] = mean(eigvals_spacings)
    #     # var_vec[i] = var(eigvals_spacings)
    # end
    # # Plot eigenvalues spacing mean
    # fig = Figure()
    # ax = Axis(fig[1, 1],
    #     title=L"Eigenvalues spacing mean $D = %$(D)$",
    #     xlabel=L"$T/T_c$", ylabel=L"\langle\Delta\lambda\rangle",
    #     xticks=0:0.5:6.5)
    # scatterlines!(ax, tau_vec, mean_vec)
    # save(joinpath(output_dir_D, filename(global_prefix * "EigvalsSpacingMean", @varsdict(D); ext="svg")), fig)
    # # Plot eigenvalues spacing variance
    # fig = Figure()
    # ax = Axis(fig[1, 1],
    #     title=L"Eigenvalues spacing variance $D = %$(D)$",
    #     xlabel=L"$T/T_c$", ylabel=L"\langle ( \Delta\lambda )^2 \rangle - \langle \Delta\lambda \rangle^2",
    #     xticks=0:0.5:6.5)
    # scatterlines!(ax, tau_vec, var_vec)
    # save(joinpath(output_dir_D, filename(global_prefix * "EigvalsSpacingVar", @varsdict(D); ext="svg")), fig)

    # # Plot eigenvalues max spacing fluctuations
    # # Loop on temperatures
    # mean_vec = similar(T_vec)
    # var_vec = similar(T_vec)
    # for (i, T) in enumerate(T_vec)
    #     datafile = D_dict[T]
    #     eigvals_matrix = load_pickle(datafile.path)
    #     eigvals_spacings = get_spacings(eigvals_matrix)
    #     eigvals_spacings_max = vec(maximum(eigvals_spacings, dims=2))
    #     # mean_vec[i], var_vec[i] = hist_fluctuations(eigvals_spacings, 100)
    #     mean_vec[i] = mean(eigvals_spacings_max)
    #     var_vec[i] = var(eigvals_spacings_max)
    # end
    # # Plot mean maximum eigenvalues spacing
    # fig = Figure()
    # ax = Axis(fig[1, 1],
    #     title=L"Mean eigenvalue maximum spacing $D = %$(D)$",
    #     xlabel=L"$T/T_c$", ylabel=L"\langle \max(\Delta\lambda) \rangle",
    #     xticks=0:0.5:6.5)
    # scatterlines!(ax, tau_vec, mean_vec)
    # save(joinpath(output_dir_D, filename(global_prefix * "EigvalsMaxSpacingMean", @varsdict(D); ext="svg")), fig)
    # # Plot maximum eigenvalues spacing variance
    # fig = Figure()
    # ax = Axis(fig[1, 1],
    #     title=L"Eigenvalues maximum spacing variance $D = %$(D)$",
    #     xlabel=L"$T/T_c$", ylabel=L"\langle \max\left((\Delta\lambda)^2\right) \rangle - \langle \max(\Delta\lambda) \rangle^2",
    #     xticks=0:0.5:6.5)
    # scatterlines!(ax, tau_vec, var_vec)
    # save(joinpath(output_dir_D, filename(global_prefix * "EigvalsMaxSpacingVar", @varsdict(D); ext="svg")), fig)

    @info "Plotting combined histograms..."
    idxs = [1, 4, 5, 6, 8, 10, 11, 19, 21]
    println("tau = ", getindex(tau_vec, idxs))

    # # Eigenvalues distribution
    # fig = Figure()
    # axs = [Axis(fig[i, j],
    #     yticks=make_ticks_log(-5:2:0))
    #        for i ∈ 1:3 for j ∈ 1:3]
    # for (ax, idx) ∈ zip(axs, idxs)
    #     T = T_vec[idx]
    #     tau = round(T / T_c; digits=3)
    #     ax.title = L"$T/T_c = %$(tau)$"
    #     datafile = D_dict[T]
    #     eigvals = load_pickle(datafile.path)
    #     # hist!(ax, vec(eigvals), bins=100, normalization=:probability)
    #     hist = Histogram(vec(eigvals), 100)
    #     x, y = hist_coords(hist)
    #     const_log = log10(sum(y))
    #     ax.limits = ((0, x[end]), (-const_log, 0))
    #     x_max = x[end]
    #     ax.xticks = axis_ticks_range(0, x_max, 4)
    #     y = log10.(y)
    #     barplot!(ax, x, y, gap=0, offset=-const_log)
    # end
    # Label(fig[0, :], text=L"Eigenvalues ($D = %$(D)$)", fontsize=30)
    # Label(fig[4, 1:3], L"$\lambda$")
    # Label(fig[1:3, 0], L"$\rho(\lambda)$", rotation=pi / 2)
    # save(joinpath(output_dir_D, filename(global_prefix * "EigvalsHist", @varsdict(D); ext="svg")), fig)

    # # Eigenvalue minimum
    # fig = Figure()
    # axs = [Axis(fig[i, j],
    #     yticks=make_ticks_log(-5:2:0))
    #        for i ∈ 1:3 for j ∈ 1:3]
    # for (ax, idx) ∈ zip(axs, idxs)
    #     T = T_vec[idx]
    #     tau = round(T / T_c; digits=3)
    #     ax.title = L"$T/T_c = %$(tau)$"
    #     datafile = D_dict[T]
    #     eigvals_matrix = load_pickle(datafile.path)
    #     eigvals_min = vec(eigvals_matrix[:, begin])
    #     hist = Histogram(eigvals_min, 50)
    #     x, y = hist_coords(hist)
    #     const_log = log10(sum(y))
    #     ax.limits = ((0, x[end]), (-const_log, 0))
    #     x_max = x[end]
    #     ax.xticks = axis_ticks_range(0, x_max, 3)
    #     y = log10.(y)
    #     barplot!(ax, x, y, gap=0, offset=-const_log)
    # end
    # Label(fig[0, :], text=L"Minimum eigenvalue distribution ($D = %$(D)$)", fontsize=30)
    # Label(fig[4, 1:3], L"$\lambda_{min}$")
    # Label(fig[1:3, 0], L"$\rho(\lambda_{min})$", rotation=pi / 2)
    # save(joinpath(output_dir_D, filename(global_prefix * "EigvalsMinHist", @varsdict(D); ext="svg")), fig)

    # # Eigenvalue maximum
    # fig = Figure()
    # axs = [Axis(fig[i, j],
    #     yticks=make_ticks_log(-5:2:0))
    #        for i ∈ 1:3 for j ∈ 1:3]
    # for (ax, idx) ∈ zip(axs, idxs)
    #     T = T_vec[idx]
    #     tau = round(T / T_c; digits=3)
    #     ax.title = L"$T/T_c = %$(tau)$"
    #     datafile = D_dict[T]
    #     eigvals_matrix = load_pickle(datafile.path)
    #     eigvals_max = vec(eigvals_matrix[:, end])
    #     hist = Histogram(eigvals_max, 50)
    #     x, y = hist_coords(hist)
    #     const_log = log10(sum(y))
    #     ax.limits = ((0, x[end]), (-const_log, 0))
    #     x_max = x[end]
    #     ax.xticks = axis_ticks_int_range(0, x_max, 4)
    #     y = log10.(y)
    #     barplot!(ax, x, y, gap=0, offset=-const_log)
    # end
    # Label(fig[0, :], text=L"Maximum eigenvalue distribution ($D = %$(D)$)", fontsize=30)
    # Label(fig[4, 1:3], L"$\lambda_{max}$")
    # Label(fig[1:3, 0], L"$\rho(\lambda_{max})$", rotation=pi / 2)
    # save(joinpath(output_dir_D, filename(global_prefix * "EigvalsMaxHist", @varsdict(D); ext="svg")), fig)

    # Eigenvalues spacing distributions
    fig = Figure()
    axs = [Axis(fig[i, j],
        yticks=make_ticks_log(-4:2:0))
           for i ∈ 1:3 for j ∈ 1:3]
    for (ax, idx) ∈ zip(axs, idxs)
        T = T_vec[idx]
        tau = round(T / T_c; digits=3)
        ax.title = L"$T/T_c = %$(tau)$"
        datafile = D_dict[T]
        eigvals_matrix = load_pickle(datafile.path)
        # script_show(eigvals_matrix)
        eigvals_normalized_spacings = get_normalized_spacings(eigvals_matrix)
        # script_show(eigvals_normalized_spacings)
        hist = Histogram(vec(eigvals_normalized_spacings), 100)
        x, y = hist_coords(hist)
        const_log = log10(sum(y))
        ax.limits = ((0, x[end]), (-const_log, nothing))
        ax.xticks = axis_ticks_int_range(0, x[end], 4)
        y = log10.(y)
        barplot!(ax, x, y, gap=0, offset=-const_log)
    end
    Label(fig[0, :], text=L"Eigenvalues spacing ($D = %$(D)$)", fontsize=30)
    Label(fig[4, 1:3], L"$s$")
    Label(fig[1:3, 0], L"$\rho(s)$", rotation=pi / 2)
    save(joinpath(output_dir_D, filename(global_prefix * "EigvalsSpacingHist", @varsdict(D); ext="svg")), fig)

    # @info "Plotting histograms..."
    # for (T, tau) in zip(T_vec, tau_vec)
    #     datafile = D_dict[T]
    #     eigvals_matrix = load_pickle(datafile.path)
    #     @show tau T

    #     # Create dir
    #     output_dir_D_T = joinpath(output_dir_D, "tau=$tau")
    #     mkpath(output_dir_D_T)

    #     # Plot eigenvalues distribution
    #     @info "Plotting eigenvalues distributions..."
    #     fig = Figure()
    #     ax = Axis(fig[1, 1],
    #         title=L"Eigenvalues distribution $D = %$(D)$, $T/T_c = %$(tau)$",
    #         xlabel=L"\lambda",
    #         ylabel=L"\rho(\lambda)",
    #         limits=((0, nothing), (0, nothing)),
    #         yticks=make_ticks_log(0:5),
    #         yscale=Makie.pseudolog10)
    #     hist!(ax, vec(eigvals_matrix), bins=100)
    #     save(joinpath(output_dir_D_T, filename(global_prefix * "EigvalsHist", @varsdict(D, tau); ext="svg")), fig)

    #     # Plot eigenvalues gaps distribution
    #     @info "Plotting eigenvalues spacing distributions..."
    #     fig = Figure()
    #     ax = Axis(fig[1, 1],
    #         title=L"Eigenvalues gap distribution $D = %$(D)$, $T/T_c = %$(tau)$",
    #         xlabel=L"\lambda",
    #         ylabel=L"\rho(\Delta\lambda)",
    #         limits=((0, nothing), (0, nothing)),
    #         yticks=make_ticks_log(0:5),
    #         yscale=Makie.pseudolog10)
    #     eigvals_normalized_spacings = get_normalized_spacings(eigvals_matrix)
    #     hist!(ax, vec(eigvals_normalized_spacings), bins=100)
    #     save(joinpath(output_dir_D_T, filename(global_prefix * "EigvalsSpacingHist", @varsdict(D, tau); ext="svg")), fig)

    #     # Plot min eigenvalue distribution
    #     @info "Plotting min eigenvalue distributions..."
    #     fig = Figure()
    #     ax = Axis(fig[1, 1],
    #         title=L"Minimum eigenvalue distribution $D = %$(D)$, $T/T_c = %$(tau)$",
    #         xlabel=L"\lambda",
    #         ylabel=L"min(\lambda)",
    #         limits=((0, nothing), (0, nothing)),
    #         yticks=make_ticks_log(0:5),
    #         yscale=Makie.pseudolog10)
    #     eigvals_min = vec(eigvals_matrix[:, begin])
    #     hist!(ax, eigvals_min, bins=100)
    #     save(joinpath(output_dir_D_T, filename(global_prefix * "EigvalsMin", @varsdict(D, tau); ext="svg")), fig)

    #     # Plot max eigenvalue distribution
    #     @info "Plotting max eigenvalue distributions..."
    #     fig = Figure()
    #     ax = Axis(fig[1, 1],
    #         title=L"Maximum eigenvalue distribution $D = %$(D)$, $T/T_c = %$(tau)$",
    #         xlabel=L"\lambda",
    #         ylabel=L"max(\lambda)",
    #         limits=((0, nothing), (0, nothing)),
    #         yticks=make_ticks_log(0:5),
    #         yscale=Makie.pseudolog10)
    #     eigvals_max = vec(eigvals_matrix[:, end])
    #     hist!(ax, eigvals_max, bins=100)
    #     save(joinpath(output_dir_D_T, filename(global_prefix * "EigvalsMax", @varsdict(D, tau); ext="svg")), fig)
    # end

end
