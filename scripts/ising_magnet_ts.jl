using DrWatson

@quickactivate "phd"

using UnicodePlots

include("../src/DataIO.jl")
include("../src/IsingModel.jl")

using .DataIO
using .IsingModel

# Parameters
const dim = 2
const L = 100
const σ₀ = up
const n_steps = 2^16
const n_samples = 10
const τ = 0

# Ising system
ising = IsingSquareLattice(Val(dim), L, σ₀)

# Temperature
β = ising_square_lattice_2d_beta_critical(τ)

# Generate magnetization time series matrices
M = hcat(map(1:n_samples) do _
    randomize_state!(ising)
    return metropolis_and_measure_energy!(ising, β, n_steps)
end...)

script_show(M)
println()
@show τ β
println()

x_max = n_steps + 1
plt = lineplot(1:x_max, M[:, 1], xlim = (0, x_max), ylim = extrema(M), width = 100, height = 25)
for k ∈ 2:size(M, 2)
    lineplot!(plt, 1:x_max, M[:, k])
end
display(plt)
