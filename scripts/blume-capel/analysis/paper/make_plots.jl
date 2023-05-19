# Dr Watson helper
using DrWatson
@quickactivate "phd"

using CairoMakie

include("plot_utils.jl")

@info "Loading datafiles..."
const datafiles_dict = get_datafiles_dict(datadir("blume_capel_pickles", "eigvals"))

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
for (ax_mean, ax_var, D) ∈ zip(axs_mean, axs_var, D_vals)
    D_dict = datafiles_dict[D]
    T_c, transition_order, crit_temp_source = get_critical_temperature(df_temperatures, D)
    transition_order_str = replace(transition_order,
        "first" => "1st order",
        "second" => "2nd order",
        "tcp" => "TCP")
    crit_temp_source_str = replace(string(crit_temp_source), "_" => " ")
    T_vec = sort(collect(keys(D_dict)))
    tau_vec = map(T_vec ./ T_c) do x
        round(x; digits=3)
    end
    map(ax -> ax.title = L"$D = %$(D)$ (%$(transition_order_str))", [ax_mean, ax_var])
    # Loop on temperatures
    mean_vec = similar(T_vec)
    var_vec = similar(T_vec)
    for (i, T) in enumerate(T_vec)
        datafile = D_dict[T]
        eigvals = vec(load_pickle(datafile.path))
        mean_vec[i], var_vec[i] = hist_fluctuations(eigvals, 100)
        # mean_vec[i] = mean(eigvals)
        # var_vec[i] = var(eigvals)
    end
    scatterlines!(ax_mean, tau_vec, mean_vec)
    ax_mean.yticks = axis_ticks_range(extrema(mean_vec)..., 5)
    scatterlines!(ax_var, tau_vec, var_vec)
    ax_var.yticks = axis_ticks_range(0, maximum(var_vec), 5)
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
for (ax_mean, ax_var, D) ∈ zip(axs_mean, axs_var, D_vals)
    D_dict = datafiles_dict[D]
    T_c, transition_order, crit_temp_source = get_critical_temperature(df_temperatures, D)
    transition_order_str = replace(transition_order,
        "first" => "1st order",
        "second" => "2nd order",
        "tcp" => "TCP")
    crit_temp_source_str = replace(string(crit_temp_source), "_" => " ")
    T_vec = sort(collect(keys(D_dict)))
    tau_vec = map(T_vec ./ T_c) do x
        round(x; digits=3)
    end
    map(ax -> ax.title = L"$D = %$(D)$ (%$(transition_order_str))", [ax_mean, ax_var])
    # Loop on temperatures
    mean_vec = similar(T_vec)
    var_vec = similar(T_vec)
    for (i, T) in enumerate(T_vec)
        datafile = D_dict[T]
        eigvals_matrix = load_pickle(datafile.path)
        eigvals_min = vec(eigvals_matrix[:, begin])
        mean_vec[i] = mean(eigvals_min)
        var_vec[i] = var(eigvals_min)
    end
    scatterlines!(ax_mean, tau_vec, mean_vec)
    scatterlines!(ax_var, tau_vec, var_vec)
end
Label(fig_mean[0, :], "Mean minumum eigenvalue")
Label(fig_mean[4, 1:2], L"$T/T_c$")
Label(fig_mean[1:3, 0], L"$\langle \lambda_{min} \rangle$", rotation=pi / 2)
save(joinpath(output_root, filename(global_prefix * "MinEigvalMean"; ext="svg")), fig_mean)
Label(fig_var[0, :], "Minimum eigenvalue variance")
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
for (ax_mean, ax_var, D) ∈ zip(axs_mean, axs_var, D_vals)
    D_dict = datafiles_dict[D]
    T_c, transition_order, crit_temp_source = get_critical_temperature(df_temperatures, D)
    transition_order_str = replace(transition_order,
        "first" => "1st order",
        "second" => "2nd order",
        "tcp" => "TCP")
    crit_temp_source_str = replace(string(crit_temp_source), "_" => " ")
    T_vec = sort(collect(keys(D_dict)))
    tau_vec = map(T_vec ./ T_c) do x
        round(x; digits=3)
    end
    map(ax -> ax.title = L"$D = %$(D)$ (%$(transition_order_str))", [ax_mean, ax_var])
    # Loop on temperatures
    mean_vec = similar(T_vec)
    var_vec = similar(T_vec)
    for (i, T) in enumerate(T_vec)
        datafile = D_dict[T]
        eigvals_matrix = load_pickle(datafile.path)
        eigvals_max = vec(eigvals_matrix[:, end])
        mean_vec[i] = mean(eigvals_max)
        var_vec[i] = var(eigvals_max)
    end
    scatterlines!(ax_mean, tau_vec, mean_vec)
    scatterlines!(ax_var, tau_vec, var_vec)
end
Label(fig_mean[0, :], "Mean maximum eigenvalue")
Label(fig_mean[4, 1:2], L"$T/T_c$")
Label(fig_mean[1:3, 0], L"$\langle \lambda_{max} \rangle$", rotation=pi / 2)
save(joinpath(output_root, filename(global_prefix * "MaxEigvalMean"; ext="svg")), fig_mean)
Label(fig_var[0, :], "Maximum eigenvalue variance")
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
for (ax_mean, ax_var, D) ∈ zip(axs_mean, axs_var, D_vals)
    D_dict = datafiles_dict[D]
    T_c, transition_order, crit_temp_source = get_critical_temperature(df_temperatures, D)
    transition_order_str = replace(transition_order,
        "first" => "1st order",
        "second" => "2nd order",
        "tcp" => "TCP")
    crit_temp_source_str = replace(string(crit_temp_source), "_" => " ")
    T_vec = sort(collect(keys(D_dict)))
    tau_vec = map(T_vec ./ T_c) do x
        round(x; digits=3)
    end
    map(ax -> ax.title = L"$D = %$(D)$ (%$(transition_order_str))", [ax_mean, ax_var])
    # Loop on temperatures
    mean_vec = similar(T_vec)
    var_vec = similar(T_vec)
    for (i, T) in enumerate(T_vec)
        datafile = D_dict[T]
        eigvals_matrix = load_pickle(datafile.path)
        eigvals_spacings = vec(get_spacings(eigvals_matrix))
        mean_vec[i], var_vec[i] = hist_fluctuations(eigvals_spacings, 100)
        # mean_vec[i] = mean(eigvals)
        # var_vec[i] = var(eigvals)
    end
    scatterlines!(ax_mean, tau_vec, mean_vec)
    ax_mean.yticks = axis_ticks_range(0, maximum(mean_vec), 5)
    scatterlines!(ax_var, tau_vec, var_vec)
    ax_var.yticks = axis_ticks_range(0, maximum(var_vec), 5)
end
Label(fig_mean[0, :], "Mean eigenvalue spacing")
Label(fig_mean[4, 1:2], L"$T/T_c$")
Label(fig_mean[1:3, 0], L"$\langle \Delta\lambda \rangle$", rotation=pi / 2)
save(joinpath(output_root, filename(global_prefix * "EigvalSpacingMean"; ext="svg")), fig_mean)
Label(fig_var[0, :], "Eigenvalue spacing variance")
Label(fig_var[4, 1:2], L"$T/T_c$")
Label(fig_var[1:3, 0], L"\langle (\Delta\lambda)^2 \rangle - \langle \Delta\lambda \rangle^2", rotation=pi / 2)
save(joinpath(output_root, filename(global_prefix * "EigvalSpacingVar"; ext="svg")), fig_var)
