# Dr Watson helper
using DrWatson
@quickactivate "phd"

using CairoMakie

set_theme!(Theme(fontsize=26))

include("plot_utils.jl")

@info "Loading datafiles..."
const datafiles_dict = get_datafiles_dict(datadir("blume_capel_pickles", "correlations"))

for (D, D_dict) ∈ sort(collect(datafiles_dict), by=x -> x[1])
    if D ∈ [1.0]
        continue
    end
    @show D

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

    # Eigenvalues distribution
    fig = Figure()
    axs = [Axis(fig[i, j],
        yticks=make_ticks_log(-6:2:0))
           for i ∈ 1:3 for j ∈ 1:3]
    for (ax, idx) ∈ zip(axs, T_idxs)
        T = T_vec[idx]
        tau = round(T / T_c; digits=3)
        ax.title = L"$T/T_c = %$(tau)$"
        datafile = D_dict[T]
        eigvals = load_pickle(datafile.path)
        # hist!(ax, vec(eigvals), bins=100, normalization=:probability)
        hist = Histogram(vec(eigvals), 100)
        x, y = hist_coords(hist)
        const_log = log10(sum(y))
        ax.limits = ((-1.1, 1.1), (-const_log, 0))
        x_max = x[end]
        ax.xticks = axis_ticks_range(-1, 1, 3)
        y = log10.(y)
        barplot!(ax, x, y, gap=0, offset=-const_log)
    end
    Label(fig[0, :], text=L"Correlations ($D = %$(D)$)", fontsize=30)
    Label(fig[4, 1:3], L"$\rho$")
    Label(fig[1:3, 0], L"$p(\rho)$", rotation=pi / 2)
    save(joinpath(output_root, filename(global_prefix * "CorrHist", @varsdict(D); ext="svg")), fig)

end
