# Dr Watson helper
using DrWatson
@quickactivate "phd"

using CairoMakie

include("plot_utils.jl")

@info "Loading datafiles..."
const datafiles_dict = get_datafiles_dict(datadir("blume_capel_pickles", "eigvals"))

function discrete_first_derivative(x::AbstractVector{T}, y::AbstractVector{T}) where {T<:Real}
    @assert length(x) == length(y)
    N = length(x)
    x_out = x[2:N-1]
    y_out = Vector{T}(undef, N - 2)
    for k ∈ 2:N-1
        y_out[k-1] = (y[k+1] - y[k-1]) / (x[k+1] - x[k-1])
    end
    return (x_out, y_out)
end

function discrete_second_derivative(x::AbstractVector{T}, y::AbstractVector{T}) where {T<:Real}
    @assert length(x) == length(y)
    N = length(x)
    x_out = x[2:N-1]
    y_out = Vector{T}(undef, N - 2)
    for k ∈ 2:N-1
        y_out[k-1] = 2 * ((x[k] - x[k-1]) * (y[k+1] - y[k]) - (x[k+1] - x[k]) * (y[k] - y[k-1])) / ((x[k+1] - x[k-1]) * (x[k+1] - x[k]) * (x[k] - x[k-1]))
    end
    return (x_out, y_out)
end

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
var_vec = similar(T_vec)
for (i, T) in enumerate(T_vec)
    datafile = D_dict[T]
    eigvals = vec(load_pickle(datafile.path))
    # _, var_vec[i] = hist_fluctuations(eigvals, 100)
    var_vec[i] = var(eigvals)
end

set_theme!(Theme(fontsize=26))

fig = Figure()

ax_main = Axis(fig[1, 1],
    limits=((nothing, nothing), (0, nothing)),
    xticks=axis_ticks([1, 2, 4, 6]),
    yticks=axis_ticks(0:20:60),
    backgroundcolor=:white,
    title=L"Eigenvalue spectrum variance ($D=%$(D)$)",
    xlabel=L"T/T_c",
    ylabel=L"\langle \lambda^2 \rangle - \langle \lambda \rangle^2", ylabelrotation=pi / 2)
scatterlines!(ax_main, tau_vec, var_vec)
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
x_diff, y_diff = discrete_first_derivative(tau_vec, var_vec)
scatterlines!(ax_inset, x_diff, y_diff .* (1 / T_c))
vlines!(ax_inset, [1], color=:grey)
save(joinpath(output_root, filename(global_prefix * "EigvalsVarDiff", @varsdict(D); ext="svg")), fig)
