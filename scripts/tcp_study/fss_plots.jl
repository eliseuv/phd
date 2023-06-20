# Dr Watson helper
using DrWatson
@quickactivate "phd"

using DataFrames, CairoMakie, Makie.Colors

include("plot_utils.jl")

@inline log_interpolate(x, low, high; base=2) = (log(base, x) - log(base, low)) / (log(base, high) - log(base, low))
@inline linear_interpolate(x, low, high) = (x - low) / (high - low)

@info "Loading datafiles..."

# Blume-Capel 2D
const system_title = "Blume-Capel 2D"
const datafiles_dir = datadir("magnet_ts_wishart", "blume-capel_2d", "eigvals")
const global_prefix = "BlumeCapelSq2D"
const D = 1.96582
const L_max = 30
const datafiles = find_datafiles(datafiles_dir,
    global_prefix * "Eigvals",
    "D" => D;
    ext=".pickle")
const output_root = plotsdir("magnet_ts_wishart", "blume-capel_2d", "fss")
const df_temperatures = DataFrame(CSV.File(projectdir("tables", "butera_and_pernici_2018", "blume-capel_s=1_square_lattice.csv")))

# # Blume-Capel 3D
# const system_title = "Blume-Capel 3D"
# const datafiles_dir = datadir("magnet_ts_wishart", "blume-capel_3d", "eigvals")
# const global_prefix = "BlumeCapelSq3D"
# const D = 2.8448
# const L_max = 10
# const datafiles = find_datafiles(datafiles_dir,
#     global_prefix * "Eigvals",
#     "D" => D;
#     ext=".pickle")
# const output_root = plotsdir("magnet_ts_wishart", "blume-capel_3d", "fss")
# const df_temperatures = DataFrame(CSV.File(projectdir("tables", "butera_and_pernici_2018", "blume-capel_s=1_cubic_lattice.csv")))

const L_vals = map(x -> x.params["L"], datafiles) |> sort |> unique |> filter(L -> L <= L_max)
@show L_vals
const T_c, transition_order_str, _ = get_critical_temperature_info(df_temperatures, D)
mkpath(output_root)

set_theme!(Theme(fontsize=24))
fig_size = (800, 600)

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
mean_curves = []
var_curves = []
for (i, L) ∈ enumerate(L_vals)
    datafiles_L = filter(x -> x.params["L"] == L, datafiles)
    T_vec = map(x -> x.params["T"], datafiles_L) |> sort
    # Loop on temperatures
    df = DataFrame(tau=Float64[], mean=Float64[], var=Float64[])
    for (i, T) in enumerate(T_vec)
        datafile = only(filter(x -> x.params["T"] == T, datafiles_L))
        eigvals = vec(load_pickle(datafile.path))
        val_mean, val_var = hist_fluctuations(eigvals, 100)
        # val_var = var(eigvals)
        push!(df, (tau=T / T_c, mean=val_mean, var=val_var))
    end
    v = linear_interpolate(L, extrema(L_vals)...)
    push!(mean_curves, scatterlines!(ax_mean, df.tau, df.mean, color=RGBf(v, 0, 1 - v)))
    push!(var_curves, scatterlines!(ax_var, df.tau, df.var, color=RGBf(v, 0, 1 - v)))
end
Legend(fig_mean[1, 2], mean_curves, (LaTeXString ∘ string).(L_vals), L"L", framevisible=false)
Legend(fig_var[1, 2], var_curves, (LaTeXString ∘ string).(L_vals), L"L", framevisible=false)
save(joinpath(output_root, filename(global_prefix * "EigvalMean"; ext="svg")), fig_mean)
save(joinpath(output_root, filename(global_prefix * "EigvalVar"; ext="svg")), fig_var)

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
mean_curves = []
var_curves = []
for (i, L) ∈ enumerate(L_vals)
    datafiles_L = filter(x -> x.params["L"] == L, datafiles)
    T_vec = map(x -> x.params["T"], datafiles_L) |> sort
    # Loop on temperatures
    df = DataFrame(tau=Float64[], mean=Float64[], var=Float64[])
    for (i, T) in enumerate(T_vec)
        datafile = only(filter(x -> x.params["T"] == T, datafiles_L))
        eigvals_matrix = load_pickle(datafile.path)
        eigvals_min = vec(eigvals_matrix[:, begin])
        val_mean = mean(eigvals_min)
        val_var = var(eigvals_min)
        push!(df, (tau=T / T_c, mean=val_mean, var=val_var))
    end
    v = linear_interpolate(L, extrema(L_vals)...)
    push!(mean_curves, scatterlines!(ax_mean, df.tau, df.mean, color=RGBf(v, 0, 1 - v)))
    push!(var_curves, scatterlines!(ax_var, df.tau, df.var, color=RGBf(v, 0, 1 - v)))
end
Legend(fig_mean[1, 2], mean_curves, (LaTeXString ∘ string).(L_vals), L"L", framevisible=false)
Legend(fig_var[1, 2], var_curves, (LaTeXString ∘ string).(L_vals), L"L", framevisible=false)
save(joinpath(output_root, filename(global_prefix * "MinEigvalMean"; ext="svg")), fig_mean)
save(joinpath(output_root, filename(global_prefix * "MinEigvalVar"; ext="svg")), fig_var)

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
mean_curves = []
var_curves = []
for (i, L) ∈ enumerate(L_vals)
    datafiles_L = filter(x -> x.params["L"] == L, datafiles)
    T_vec = map(x -> x.params["T"], datafiles_L) |> sort
    # Loop on temperatures
    df = DataFrame(tau=Float64[], mean=Float64[], var=Float64[])
    for (i, T) in enumerate(T_vec)
        datafile = only(filter(x -> x.params["T"] == T, datafiles_L))
        eigvals_matrix = load_pickle(datafile.path)
        eigvals_max = vec(eigvals_matrix[:, end])
        val_mean = mean(eigvals_max)
        val_var = var(eigvals_max)
        push!(df, (tau=T / T_c, mean=val_mean, var=val_var))
    end
    v = linear_interpolate(L, extrema(L_vals)...)
    push!(mean_curves, scatterlines!(ax_mean, df.tau, df.mean, color=RGBf(v, 0, 1 - v)))
    push!(var_curves, scatterlines!(ax_var, df.tau, df.var, color=RGBf(v, 0, 1 - v)))
end
Legend(fig_mean[1, 2], mean_curves, (LaTeXString ∘ string).(L_vals), L"L", framevisible=false)
Legend(fig_var[1, 2], var_curves, (LaTeXString ∘ string).(L_vals), L"L", framevisible=false)
save(joinpath(output_root, filename(global_prefix * "MaxEigvalMean"; ext="svg")), fig_mean)
save(joinpath(output_root, filename(global_prefix * "MaxEigvalVar"; ext="svg")), fig_var)

@info "Plotting eigenvalue spacings fluctuations..."
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
mean_curves = []
var_curves = []
for (i, L) ∈ enumerate(L_vals)
    datafiles_L = filter(x -> x.params["L"] == L, datafiles)
    T_vec = map(x -> x.params["T"], datafiles_L) |> sort
    # Loop on temperatures
    df = DataFrame(tau=Float64[], mean=Float64[], var=Float64[])
    for (i, T) in enumerate(T_vec)
        datafile = only(filter(x -> x.params["T"] == T, datafiles_L))
        eigvals_matrix = load_pickle(datafile.path)
        eigvals_spacings = vec(get_spacings(eigvals_matrix))
        val_mean = mean(eigvals_spacings)
        val_var = var(eigvals_spacings)
        push!(df, (tau=T / T_c, mean=val_mean, var=val_var))
    end
    v = linear_interpolate(L, extrema(L_vals)...)
    push!(mean_curves, scatterlines!(ax_mean, df.tau, df.mean, color=RGBf(v, 0, 1 - v)))
    push!(var_curves, scatterlines!(ax_var, df.tau, df.var, color=RGBf(v, 0, 1 - v)))
end
Legend(fig_mean[1, 2], mean_curves, (LaTeXString ∘ string).(L_vals), L"L", framevisible=false)
Legend(fig_var[1, 2], var_curves, (LaTeXString ∘ string).(L_vals), L"L", framevisible=false)
save(joinpath(output_root, filename(global_prefix * "EigvalSpacingMean"; ext="svg")), fig_mean)
save(joinpath(output_root, filename(global_prefix * "EigvalSpacingVar"; ext="svg")), fig_var)
