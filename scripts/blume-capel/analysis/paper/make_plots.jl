# Dr Watson helper
using DrWatson
@quickactivate "phd"

using DataFrames, CairoMakie

include("plot_utils.jl")

function make_param_dict(datafiles::AbstractVector{DataFile})
    datafiles_dict = Dict()
    for datafile ∈ datafiles
        D = Float64(datafile.params["D"])
        T = Float64(datafile.params["T"])
        if haskey(datafiles_dict, D)
            datafiles_dict[D][T] = datafile
        else
            datafiles_dict[D] = Dict(T => datafile)
        end
    end
    return datafiles_dict
end

@info "Loading datafiles..."
const datafiles_dir = datadir("magnet_ts_wishart", "blume-capel_3d", "eigvals")
const global_prefix = "BlumeCapelSq3D"
const datafiles = find_datafiles(datafiles_dir,
    global_prefix * "Eigvals",
    "L" => 20;
    ext=".pickle")

const D_vals_available = map(x -> x.params["D"], datafiles) |> unique |> sort
@show D_vals_available

const output_root = plotsdir("blume-capel_3d")
const df_temperatures = DataFrame(CSV.File(projectdir("tables", "butera_and_pernici_2018", "blume-capel_s=1_cubic_lattice.csv")))

# D values considered
# const D_vals_2order = [1.0, 1.75, 1.9]
# const D_val_tcp = 1.96582
# const D_vals_1order = [1.9777, 1.99932488]
const D_vals_2order = [0, 1, 2]
const D_val_tcp = 2.8446
const D_vals_1order = [2.8502, 2.998]
const D_vals_plot = [D_vals_2order..., D_val_tcp, D_vals_1order...]


set_theme!(Theme(fontsize=24))
fig_size = (700, 900)

@info "Plotting eigenvalues fluctuations..."
fig_mean = Figure(resolution=fig_size)
axs_mean = [Axis(fig_mean[i, j],
    limits=((nothing, nothing), (nothing, nothing)),
    xticks=axis_ticks([1, 2, 4, 6]))
            for i ∈ 1:3 for j ∈ 1:2]
fig_var = Figure(resolution=fig_size)
axs_var = [Axis(fig_var[i, j],
    limits=((nothing, nothing), (0, nothing)),
    xticks=axis_ticks([1, 2, 4, 6]))
           for i ∈ 1:3 for j ∈ 1:2]
for (ax_mean, ax_var, D) ∈ zip(axs_mean, axs_var, D_vals_plot)
    datafiles_D = filter(x -> x.params["D"] == D, datafiles)
    T_vec = map(x -> x.params["T"], datafiles_D) |> sort
    T_c, transition_order_str, _ = get_critical_temperature_info(df_temperatures, D)
    foreach(ax -> ax.title = L"$D = %$(D)$ (%$(transition_order_str))", [ax_mean, ax_var])
    # Loop on temperatures
    df = DataFrame(tau=Float64[], mean=Float64[], var=Float64[])
    for (i, T) in enumerate(T_vec)
        datafile = only(filter(x -> x.params["T"] == T, datafiles_D))
        eigvals = vec(load_pickle(datafile.path))
        val_mean, val_var = hist_fluctuations(eigvals, 100)
        push!(df, (tau=T / T_c, mean=val_mean, var=val_var))
    end
    scatterlines!(ax_mean, df.tau, df.mean)
    ax_mean.yticks = axis_ticks_range(extrema(df.mean)..., 5)
    scatterlines!(ax_var, df.tau, df.var)
    ax_var.yticks = axis_ticks_range(0, maximum(df.var), 5)
end
Label(fig_mean[0, :], "Mean eigenvalue")
Label(fig_mean[4, 1:2], L"$T/T_c$")
Label(fig_mean[1:3, 0], L"$\langle \lambda \rangle$", rotation=pi / 2)
save(joinpath(output_root, filename(global_prefix * "EigvalsMean"; ext="svg")), fig_mean)
Label(fig_var[0, :], "Eigenvalues variance")
Label(fig_var[4, 1:2], L"$T/T_c$")
Label(fig_var[1:3, 0], L"\langle \lambda^2 \rangle - \langle \lambda \rangle^2", rotation=pi / 2)
save(joinpath(output_root, filename(global_prefix * "EigvalsVar"; ext="svg")), fig_var)

@info "Plotting minimum eigenvalue fluctuations..."
fig_mean = Figure(resolution=fig_size)
axs_mean = [Axis(fig_mean[i, j],
    limits=((nothing, nothing), (nothing, nothing)),
    xticks=axis_ticks([1, 2, 4, 6]),
    yticks=make_ticks(-4:1:0),
    yscale=log10)
            for i ∈ 1:3 for j ∈ 1:2]
fig_var = Figure(resolution=fig_size)
axs_var = [Axis(fig_var[i, j],
    limits=((nothing, nothing), (nothing, nothing)),
    xticks=axis_ticks([1, 2, 4, 6]),
    yticks=make_ticks(-10:1:0),
    yscale=log10)
           for i ∈ 1:3 for j ∈ 1:2]
for (ax_mean, ax_var, D) ∈ zip(axs_mean, axs_var, D_vals_plot)
    datafiles_D = filter(x -> x.params["D"] == D, datafiles)
    T_vec = map(x -> x.params["T"], datafiles_D) |> sort
    T_c, transition_order_str, _ = get_critical_temperature_info(df_temperatures, D)
    foreach(ax -> ax.title = L"$D = %$(D)$ (%$(transition_order_str))", [ax_mean, ax_var])
    # Loop on temperatures
    df = DataFrame(tau=Float64[], mean=Float64[], var=Float64[])
    for (i, T) in enumerate(T_vec)
        datafile = only(filter(x -> x.params["T"] == T, datafiles_D))
        eigvals_matrix = load_pickle(datafile.path)
        eigvals_min = vec(eigvals_matrix[:, begin])
        val_mean = mean(eigvals_min)
        val_var = var(eigvals_min)
        if val_mean > 0 && val_var > 0
            push!(df, (tau=T / T_c, mean=val_mean, var=val_var))
        end
    end
    scatterlines!(ax_mean, df.tau, df.mean)
    scatterlines!(ax_var, df.tau, df.var)
end
Label(fig_mean[0, :], "Mean smallest eigenvalue")
Label(fig_mean[4, 1:2], L"$T/T_c$")
Label(fig_mean[1:3, 0], L"$\langle \lambda_{min} \rangle$", rotation=pi / 2)
save(joinpath(output_root, filename(global_prefix * "MinEigvalMean"; ext="svg")), fig_mean)
Label(fig_var[0, :], "Smallest eigenvalue variance")
Label(fig_var[4, 1:2], L"$T/T_c$")
Label(fig_var[1:3, 0], L"\langle \lambda_{min}^2 \rangle - \langle \lambda_{min} \rangle^2", rotation=pi / 2)
save(joinpath(output_root, filename(global_prefix * "MinEigvalVar"; ext="svg")), fig_var)

@info "Plotting maximum eigenvalue fluctuations..."
fig_mean = Figure(resolution=fig_size)
axs_mean = [Axis(fig_mean[i, j],
    limits=((nothing, nothing), (nothing, nothing)),
    xticks=axis_ticks([1, 2, 4, 6]),
    yticks=make_ticks(0:0.5:2),
    yscale=log10)
            for i ∈ 1:3 for j ∈ 1:2]
fig_var = Figure(resolution=fig_size)
axs_var = [Axis(fig_var[i, j],
    limits=((nothing, nothing), (nothing, nothing)),
    xticks=axis_ticks([1, 2, 4, 6]),
    yticks=make_ticks(-2:2),
    yscale=log10)
           for i ∈ 1:3 for j ∈ 1:2]
for (ax_mean, ax_var, D) ∈ zip(axs_mean, axs_var, D_vals_plot)
    datafiles_D = filter(x -> x.params["D"] == D, datafiles)
    T_vec = map(x -> x.params["T"], datafiles_D) |> sort
    T_c, transition_order_str, _ = get_critical_temperature_info(df_temperatures, D)
    foreach(ax -> ax.title = L"$D = %$(D)$ (%$(transition_order_str))", [ax_mean, ax_var])
    # Loop on temperatures
    df = DataFrame(tau=Float64[], mean=Float64[], var=Float64[])
    for (i, T) in enumerate(T_vec)
        datafile = only(filter(x -> x.params["T"] == T, datafiles_D))
        eigvals_matrix = load_pickle(datafile.path)
        eigvals_max = vec(eigvals_matrix[:, end])
        val_mean = mean(eigvals_max)
        val_var = var(eigvals_max)
        push!(df, (tau=T / T_c, mean=val_mean, var=val_var))
    end
    scatterlines!(ax_mean, df.tau, df.mean)
    scatterlines!(ax_var, df.tau, df.var)
end
Label(fig_mean[0, :], "Mean largest eigenvalue")
Label(fig_mean[4, 1:2], L"$T/T_c$")
Label(fig_mean[1:3, 0], L"$\langle \lambda_{max} \rangle$", rotation=pi / 2)
save(joinpath(output_root, filename(global_prefix * "MaxEigvalMean"; ext="svg")), fig_mean)
Label(fig_var[0, :], "Largest eigenvalue variance")
Label(fig_var[4, 1:2], L"$T/T_c$")
Label(fig_var[1:3, 0], L"\langle \lambda_{max}^2 \rangle - \langle \lambda_{max} \rangle^2", rotation=pi / 2)
save(joinpath(output_root, filename(global_prefix * "MaxEigvalVar"; ext="svg")), fig_var)

@info "Plotting eigenvalue spacings fluctuations..."
fig_mean = Figure(resolution=fig_size)
axs_mean = [Axis(fig_mean[i, j],
    limits=((nothing, nothing), (0, nothing)),
    xticks=axis_ticks([1, 2, 4, 6]))
            for i ∈ 1:3 for j ∈ 1:2]
fig_var = Figure(resolution=fig_size)
axs_var = [Axis(fig_var[i, j],
    limits=((nothing, nothing), (0, nothing)),
    xticks=axis_ticks([1, 2, 4, 6]))
           for i ∈ 1:3 for j ∈ 1:2]
for (ax_mean, ax_var, D) ∈ zip(axs_mean, axs_var, D_vals_plot)
    datafiles_D = filter(x -> x.params["D"] == D, datafiles)
    T_vec = map(x -> x.params["T"], datafiles_D) |> sort
    T_c, transition_order_str, _ = get_critical_temperature_info(df_temperatures, D)
    foreach(ax -> ax.title = L"$D = %$(D)$ (%$(transition_order_str))", [ax_mean, ax_var])
    # Loop on temperatures
    df = DataFrame(tau=Float64[], mean=Float64[], var=Float64[])
    for (i, T) in enumerate(T_vec)
        datafile = only(filter(x -> x.params["T"] == T, datafiles_D))
        eigvals_matrix = load_pickle(datafile.path)
        eigvals_spacings = vec(get_spacings(eigvals_matrix))
        val_mean = mean(eigvals_spacings)
        val_var = var(eigvals_spacings)
        push!(df, (tau=T / T_c, mean=val_mean, var=val_var))
    end
    scatterlines!(ax_mean, df.tau, df.mean)
    ax_mean.yticks = axis_ticks_range(0, maximum(df.mean), 5)
    scatterlines!(ax_var, df.tau, df.var)
    ax_var.yticks = axis_ticks_range(0, maximum(df.var), 5)
end
Label(fig_mean[0, :], "Mean eigenvalue spacing")
Label(fig_mean[4, 1:2], L"$T/T_c$")
Label(fig_mean[1:3, 0], L"$\langle \Delta\lambda \rangle$", rotation=pi / 2)
save(joinpath(output_root, filename(global_prefix * "EigvalSpacingMean"; ext="svg")), fig_mean)
Label(fig_var[0, :], "Eigenvalue spacing variance")
Label(fig_var[4, 1:2], L"$T/T_c$")
Label(fig_var[1:3, 0], L"\langle (\Delta\lambda)^2 \rangle - \langle \Delta\lambda \rangle^2", rotation=pi / 2)
save(joinpath(output_root, filename(global_prefix * "EigvalSpacingVar"; ext="svg")), fig_var)
