@doc raw"""
    Generate several runs of the magnetization time series matrices for Brass cellular automaton
"""

using DrWatson

@quickactivate "phd"

using Logging, JLD2, UnicodePlots

include("../../../src/DataIO.jl")
include("../../../src/CellularAutomata.jl")

using .DataIO
using .CellularAutomata

@doc raw"""
    magnet_ts_matrix!(ca::BrassCA, p::Real, r::Real, n_steps::Integer, n_samples::Integer)

Generate a magnetization time series matrix for the Brass CA `ca` starting from a randomized initial state
and using the model probabilities `p` and `r`.

The resulting matrix has `n_samples` columns, each representing a single run of `n_steps` CA steps.
Final matrix dimensions: `n_steps × n_samples`.
"""
magnet_ts_matrix!(ca::AbstractCellularAutomaton, n_steps::Integer, n_samples::Integer) = hcat(map(1:n_samples) do _
    randomize_state!(ca.state)
    return advance_measure!(magnet_total, ca, n_steps)
end...)

const dim = 2
const L = 100
const p = parse(Float64, ARGS[1])
const r = parse(Float64, ARGS[2])
const n_steps = 1000
const n_samples = 10

# Brass CA system
ca = BrassCellularAutomaton(SquareLatticeFiniteState(Val(dim), L, BrassState.TH0), p, r)

# Generate magnetization time series matrices
@info "Generating magnetization time series matrix..."
M_ts = magnet_ts_matrix!(ca, n_steps, n_samples)

# Plot demo matrix
# display(heatmap(M_ts,
#     title="Magnet time series matrix (p = $p, r = $r)",
#     xlabel="i", ylabel="t", zlabel="mᵢ(t)",
#     width=125))
# println()

# Plot demo series
M_plot = M_ts[:, 1:10]
x_max = n_steps + 1
plt = lineplot(1:x_max, M_plot[:, 1],
    xlim=(0, x_max), ylim=extrema(M_plot),
    xlabel="t", ylabel="m",
    width=125, height=25)
for k ∈ 2:size(M_plot, 2)
    lineplot!(plt, 1:x_max, M_plot[:, k])
end
display(plt)
println()
