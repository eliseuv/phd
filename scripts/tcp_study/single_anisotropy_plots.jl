# Dr Watson helper
using DrWatson
@quickactivate "phd"

using DataFrames, CairoMakie

include("plot_utils.jl")

@info "Loading datafiles..."

# Blume-Capel 2D
const system_title = "Blume-Capel 2D"
const datafiles_dir = datadir("magnet_ts_wishart", "blume-capel_2d", "eigvals")
const global_prefix = "BlumeCapelSq2D"
const datafiles = find_datafiles(datafiles_dir,
    global_prefix * "Eigvals",
    "L" => 100;
    ext=".pickle")
const output_root = plotsdir("magnet_ts_wishart", "blume-capel_2d")
const df_temperatures = DataFrame(CSV.File(projectdir("tables", "butera_and_pernici_2018", "blume-capel_s=1_square_lattice.csv")))

# # Blume-Capel 3D
# const system_title = "Blume-Capel 3D"
# const datafiles_dir = datadir("magnet_ts_wishart", "blume-capel_3d", "eigvals")
# const global_prefix = "BlumeCapelSq3D"
# const datafiles = find_datafiles(datafiles_dir,
#     global_prefix * "Eigvals",
#     "L" => 20;
#     ext=".pickle")
# const output_root = plotsdir("magnet_ts_wishart", "blume-capel_3d")
# const df_temperatures = DataFrame(CSV.File(projectdir("tables", "butera_and_pernici_2018", "blume-capel_s=1_cubic_lattice.csv")))

const D_vals = map(x -> x.params["D"], datafiles) |> unique |> sort
@show D_vals
const tau_vals_plot = [0.5, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.5, 3.5]

set_theme!(Theme(fontsize=24))
fig_size = (800, 600)

for D ∈ D_vals

    @show D
    # Filter datafiles
    datafiles_D = filter(x -> x.params["D"] == D, datafiles)
    T_vec = map(x -> x.params["T"], datafiles_D) |> sort
    # Fetch critical temperature info
    T_c, transition_order_str, _ = get_critical_temperature_info(df_temperatures, D)
    transition_order = replace(transition_order_str,
        "1st order" => "first",
        "2nd order" => "second",
        "TCP" => "tcp")
    # Create dir
    output_dir_D = joinpath(output_root, "D=$D($(transition_order))")
    mkpath(output_dir_D)

    @info "Plotting eigenvalues fluctuations..."
    fig_mean = Figure(resolution=fig_size)
    fig_var = Figure(resolution=fig_size)
    ax_mean = Axis(fig_mean[1, 1],
        limits=((nothing, nothing), (nothing, nothing)),
        xticks=axis_ticks([1, 2, 4, 6]),
        title=L"%$(system_title) - Mean eigenvalue $D = %$(D)$ (%$(transition_order_str))",
        xlabel=L"T/T_c",
        ylabel=L"\langle \lambda \rangle", ylabelrotation=pi / 2)
    ax_var = Axis(fig_var[1, 1],
        limits=((nothing, nothing), (nothing, nothing)),
        xticks=axis_ticks([1, 2, 4, 6]),
        title=L"%$(system_title) - Eigenvalues variance $D = %$(D)$ (%$(transition_order_str))",
        xlabel=L"T/T_c",
        ylabel=L"\langle \lambda^2 \rangle - \langle \lambda \rangle^2", ylabelrotation=pi / 2)
    # Loop on temperatures
    df = DataFrame(tau=Float64[], mean=Float64[], var=Float64[])
    for (i, T) in enumerate(T_vec)
        datafile = only(filter(x -> x.params["T"] == T, datafiles_D))
        eigvals = vec(load_pickle(datafile.path))
        val_mean, val_var = hist_fluctuations(eigvals, 100)
        # val_var = var(eigvals)
        push!(df, (tau=T / T_c, mean=val_mean, var=val_var))
    end
    scatterlines!(ax_mean, df.tau, df.mean)
    scatterlines!(ax_var, df.tau, df.var)
    save(joinpath(output_dir_D, filename(global_prefix * "EigvalMean", @varsdict(D); ext="svg")), fig_mean)
    save(joinpath(output_dir_D, filename(global_prefix * "EigvalVar", @varsdict(D); ext="svg")), fig_var)

    @info "Plotting eigenvalues variance inset..."
    fig = Figure(resolution=fig_size)
    ax_main = Axis(fig[1, 1],
        limits=((nothing, nothing), (0, nothing)),
        xticks=axis_ticks([1, 2, 4, 6]),
        yticks=axis_ticks(0:20:60),
        backgroundcolor=:white,
        title=L"%$(system_title) - Eigenvalue spectrum variance ($D=%$(D)$)",
        xlabel=L"T/T_c",
        ylabel=L"\langle \lambda^2 \rangle - \langle \lambda \rangle^2", ylabelrotation=pi / 2)
    scatterlines!(ax_main, df.tau, df.var)
    vlines!(ax_main, [1], color=:grey)

    ax_inset = Axis(fig[1, 1],
        limits=((nothing, nothing), (nothing, nothing)),
        xticks=axis_ticks([1, 2, 4, 6]),
        yticks=axis_ticks(-120:40:0),
        width=Relative(0.5),
        height=Relative(0.6),
        halign=0.95,
        valign=0.65,
        backgroundcolor=:lightgray,
        xlabel=L"T/T_c",
        ylabel=L"\frac{d}{dT} \left[ \langle \lambda^2 \rangle - \langle \lambda \rangle^2 \right]", ylabelrotation=pi / 2)
    # hidexdecorations!(ax_inset)
    translate!(ax_inset.scene, 0, 0, 10)
    translate!(ax_inset.elements[:background], 0, 0, 9)
    x_diff, y_diff = discrete_first_derivative(df.tau, df.var)
    scatterlines!(ax_inset, x_diff, y_diff .* (1 / T_c))
    vlines!(ax_inset, [1], color=:grey)
    save(joinpath(output_dir_D, filename(global_prefix * "EigvalVarDiff", @varsdict(D); ext="svg")), fig)

    @info "Plotting minimum eigenvalues fluctuations..."
    fig_mean = Figure(resolution=fig_size)
    fig_var = Figure(resolution=fig_size)
    ax_mean = Axis(fig_mean[1, 1],
        limits=((nothing, nothing), (nothing, nothing)),
        xticks=axis_ticks([1, 2, 4, 6]),
        title=L"%$(system_title) - Mean smallest eigenvalue $D = %$(D)$ (%$(transition_order_str))",
        xlabel=L"T/T_c",
        ylabel=L"\langle \lambda_{min} \rangle", ylabelrotation=pi / 2)
    ax_var = Axis(fig_var[1, 1],
        limits=((nothing, nothing), (nothing, nothing)),
        xticks=axis_ticks([1, 2, 4, 6]),
        title=L"%$(system_title) - Smallest eigenvalue variance $D = %$(D)$ (%$(transition_order_str))",
        xlabel=L"T/T_c",
        ylabel=L"\langle \lambda_{min}^2 \rangle - \langle \lambda_{min} \rangle^2", ylabelrotation=pi / 2)
    # Loop on temperatures
    df = DataFrame(tau=Float64[], mean=Float64[], var=Float64[])
    for (i, T) in enumerate(T_vec)
        datafile = only(filter(x -> x.params["T"] == T, datafiles_D))
        eigvals_matrix = load_pickle(datafile.path)
        eigvals_min = vec(eigvals_matrix[:, begin])
        val_mean = mean(eigvals_min)
        val_var = var(eigvals_min)
        push!(df, (tau=T / T_c, mean=val_mean, var=val_var))
    end
    scatterlines!(ax_mean, df.tau, df.mean)
    scatterlines!(ax_var, df.tau, df.var)
    save(joinpath(output_dir_D, filename(global_prefix * "MinEigvalMean", @varsdict(D); ext="svg")), fig_mean)
    save(joinpath(output_dir_D, filename(global_prefix * "MinEigvalVar", @varsdict(D); ext="svg")), fig_var)

    @info "Plotting maximum eigenvalues fluctuations..."
    fig_mean = Figure(resolution=fig_size)
    fig_var = Figure(resolution=fig_size)
    ax_mean = Axis(fig_mean[1, 1],
        limits=((nothing, nothing), (nothing, nothing)),
        xticks=axis_ticks([1, 2, 4, 6]),
        title=L"%$(system_title) - Mean largest eigenvalue $D = %$(D)$ (%$(transition_order_str))",
        xlabel=L"T/T_c",
        ylabel=L"\langle \lambda_{max} \rangle", ylabelrotation=pi / 2)
    ax_var = Axis(fig_var[1, 1],
        limits=((nothing, nothing), (nothing, nothing)),
        xticks=axis_ticks([1, 2, 4, 6]),
        title=L"%$(system_title) - Largest eigenvalue variance $D = %$(D)$ (%$(transition_order_str))",
        xlabel=L"T/T_c",
        ylabel=L"\langle \lambda_{max}^2 \rangle - \langle \lambda_{max} \rangle^2", ylabelrotation=pi / 2)
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
    save(joinpath(output_dir_D, filename(global_prefix * "MaxEigvalMean", @varsdict(D); ext="svg")), fig_mean)
    save(joinpath(output_dir_D, filename(global_prefix * "MaxEigvalVar", @varsdict(D); ext="svg")), fig_var)

    @info "Plotting eigenvalues spacing fluctuations..."
    fig_mean = Figure(resolution=fig_size)
    fig_var = Figure(resolution=fig_size)
    ax_mean = Axis(fig_mean[1, 1],
        limits=((nothing, nothing), (nothing, nothing)),
        xticks=axis_ticks([1, 2, 4, 6]),
        title=L"%$(system_title) - Mean eigenvalue spacing $D = %$(D)$ (%$(transition_order_str))",
        xlabel=L"T/T_c",
        ylabel=L"\langle \Delta\lambda \rangle", ylabelrotation=pi / 2)
    ax_var = Axis(fig_var[1, 1],
        limits=((nothing, nothing), (nothing, nothing)),
        xticks=axis_ticks([1, 2, 4, 6]),
        title=L"%$(system_title) - Eigenvalue spacing variance $D = %$(D)$ (%$(transition_order_str))",
        xlabel=L"T/T_c",
        ylabel=L"\langle (\Delta\lambda)^2 \rangle - \langle \Delta\lambda \rangle^2", ylabelrotation=pi / 2)
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
    scatterlines!(ax_var, df.tau, df.var)
    save(joinpath(output_dir_D, filename(global_prefix * "EigvalSpacingMean", @varsdict(D); ext="svg")), fig_mean)
    save(joinpath(output_dir_D, filename(global_prefix * "EigvalSpacingVar", @varsdict(D); ext="svg")), fig_var)

    @info "Plotting largest eigenvalues spacing fluctuations..."
    fig_mean = Figure(resolution=fig_size)
    fig_var = Figure(resolution=fig_size)
    ax_mean = Axis(fig_mean[1, 1],
        limits=((nothing, nothing), (nothing, nothing)),
        xticks=axis_ticks([1, 2, 4, 6]),
        title=L"%$(system_title) - Mean largest eigenvalue spacing $D = %$(D)$ (%$(transition_order_str))",
        xlabel=L"T/T_c",
        ylabel=L"\langle \max(\Delta\lambda) \rangle", ylabelrotation=pi / 2)
    ax_var = Axis(fig_var[1, 1],
        limits=((nothing, nothing), (nothing, nothing)),
        xticks=axis_ticks([1, 2, 4, 6]),
        title=L"%$(system_title) - Largest eigenvalue spacing variance $D = %$(D)$ (%$(transition_order_str))",
        xlabel=L"T/T_c",
        ylabel=L"\langle \max\left((\Delta\lambda)^2\right) \rangle - \langle \max(\Delta\lambda) \rangle^2", ylabelrotation=pi / 2)
    # Loop on temperatures
    df = DataFrame(tau=Float64[], mean=Float64[], var=Float64[])
    for (i, T) in enumerate(T_vec)
        datafile = only(filter(x -> x.params["T"] == T, datafiles_D))
        eigvals_matrix = load_pickle(datafile.path)
        eigvals_spacings = vec(get_spacings(eigvals_matrix))
        eigvals_spacings_max = vec(maximum(eigvals_spacings, dims=2))
        val_mean = mean(eigvals_spacings_max)
        val_var = var(eigvals_spacings_max)
        push!(df, (tau=T / T_c, mean=val_mean, var=val_var))
    end
    scatterlines!(ax_mean, df.tau, df.mean)
    scatterlines!(ax_var, df.tau, df.var)
    save(joinpath(output_dir_D, filename(global_prefix * "EigvalMaxSpacingMean", @varsdict(D); ext="svg")), fig_mean)
    save(joinpath(output_dir_D, filename(global_prefix * "EigvalMaxSpacingVar", @varsdict(D); ext="svg")), fig_var)

    @info "Plotting eigenvalue histogram..."
    fig = Figure()
    axs = [Axis(fig[i, j],
        yticks=make_ticks_log(-5:2:0))
           for i ∈ 1:3 for j ∈ 1:3]
    for (ax, tau) ∈ zip(axs, tau_vals_plot)
        datafile = only(filter(x -> round(x.params["T"] / T_c; digits=3) == tau, datafiles_D))
        ax.title = L"$T/T_c = %$(tau)$"
        eigvals = load_pickle(datafile.path)
        hist = Histogram(vec(eigvals), 100)
        x, y = hist_coords(hist)
        const_log = log10(sum(y))
        ax.limits = ((0, x[end]), (-const_log, 0))
        x_max = x[end]
        ax.xticks = axis_ticks_range(0, x_max, 4)
        y = log10.(y)
        barplot!(ax, x, y, gap=0, offset=-const_log)
    end
    Label(fig[0, :], text=L"%$(system_title) - Eigenvalues ($D = %$(D)$)", fontsize=30)
    Label(fig[4, 1:3], L"$\lambda$")
    Label(fig[1:3, 0], L"$\rho(\lambda)$", rotation=pi / 2)
    save(joinpath(output_dir_D, filename(global_prefix * "EigvalHist", @varsdict(D); ext="svg")), fig)

    # @info "Plotting minimum eigenvalue histogram..."
    # fig = Figure()
    # axs = [Axis(fig[i, j],
    #     yticks=make_ticks_log(-5:2:0))
    #        for i ∈ 1:3 for j ∈ 1:3]
    # for (ax, tau) ∈ zip(axs, tau_vals_plot)
    #     datafile = only(filter(x -> round(x.params["T"] / T_c; digits=3) == tau, datafiles_D))
    #     ax.title = L"$T/T_c = %$(tau)$"
    #     eigvals_matrix = load_pickle(datafile.path)
    #     eigvals_min = vec(eigvals_matrix[:, begin])
    #     hist = Histogram(eigvals_min, 100)
    #     x, y = hist_coords(hist)
    #     const_log = log10(sum(y))
    #     ax.limits = ((0, x[end]), (-const_log, 0))
    #     x_max = x[end]
    #     ax.xticks = axis_ticks_range(0, x_max, 4)
    #     y = log10.(y)
    #     barplot!(ax, x, y, gap=0, offset=-const_log)
    # end
    # Label(fig[0, :], text=L"%$(system_title) - Smallest eigenvalue ($D = %$(D)$)", fontsize=30)
    # Label(fig[4, 1:3], L"$\lambda_{min}$")
    # Label(fig[1:3, 0], L"$\rho(\lambda_{min})$", rotation=pi / 2)
    # save(joinpath(output_dir_D, filename(global_prefix * "MinEigvalHist", @varsdict(D); ext="svg")), fig)

    @info "Plotting eigenvalue spacing histogram..."
    fig = Figure()
    axs = [Axis(fig[i, j],
        yticks=make_ticks_log(-5:2:0))
           for i ∈ 1:3 for j ∈ 1:3]
    for (ax, tau) ∈ zip(axs, tau_vals_plot)
        datafile = only(filter(x -> round(x.params["T"] / T_c; digits=3) == tau, datafiles_D))
        ax.title = L"$T/T_c = %$(tau)$"
        eigvals_matrix = load_pickle(datafile.path)
        eigvals_normalized_spacings = get_normalized_spacings(eigvals_matrix)
        hist = Histogram(vec(eigvals_normalized_spacings), 100)
        x, y = hist_coords(hist)
        const_log = log10(sum(y))
        ax.limits = ((0, x[end]), (-const_log, nothing))
        ax.xticks = axis_ticks_int_range(0, x[end], 4)
        y = log10.(y)
        barplot!(ax, x, y, gap=0, offset=-const_log)
    end
    Label(fig[0, :], text=L"%$(system_title) - Eigenvalue spacing ($D = %$(D)$)", fontsize=30)
    Label(fig[4, 1:3], L"$s$")
    Label(fig[1:3, 0], L"$\rho(s)$", rotation=pi / 2)
    save(joinpath(output_dir_D, filename(global_prefix * "EigvalSpacingHist", @varsdict(D); ext="svg")), fig)

end
