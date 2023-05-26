# Dr Watson helper
using DrWatson
@quickactivate "phd"

using CairoMakie

include("plot_utils.jl")

@info "Loading datafiles..."
const datafiles_dict = get_datafiles_dict(datadir("blume_capel_pickles", "eigvals"))

D = 1.0

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

set_theme!(Theme(fontsize=26))

fig = Figure()

ax_main = Axis(fig[1, 1],
    limits=((nothing, nothing), (nothing, nothing)),
    xticks=axis_ticks([1, 2, 4, 6]),
    yticks=axis_ticks(0.9:0.02:1.0),
    backgroundcolor=:white,
    title=L"Eigenvalue spectrum mean and variance ($D=%$(D)$)",
    xlabel=L"T/T_c",
    ylabel=L"\langle \lambda \rangle", ylabelrotation=pi / 2)
scatterlines!(ax_main, tau_vec, mean_vec)
vlines!(ax_main, [1], color=:grey)

ax_inset = Axis(fig[1, 1],
    limits=((nothing, nothing), (0, nothing)),
    xticks=axis_ticks([1, 2, 4, 6]),
    yticks=axis_ticks(0:10:60),
    width=Relative(0.55),
    height=Relative(0.55),
    halign=0.9,
    valign=0.35,
    backgroundcolor=:lightgray,
    xlabel=L"T/T_c",
    ylabel=L"\langle \lambda^2 \rangle - \langle \lambda \rangle^2", ylabelrotation=pi / 2)
# hidexdecorations!(ax_inset)
translate!(ax_inset.scene, 0, 0, 10)
translate!(ax_inset.elements[:background], 0, 0, 9)
scatterlines!(ax_inset, tau_vec, var_vec)
vlines!(ax_inset, [1], color=:grey)
save(joinpath(output_root, filename(global_prefix * "EigvalsMeanAndVar", @varsdict(D); ext="svg")), fig)
