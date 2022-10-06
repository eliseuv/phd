@doc raw"""
    Generate a magnetization time series
"""

# Dr Watson helper
using DrWatson
@quickactivate "phd"

# External libraries
using Logging, Statistics, Gadfly

# Custom modules
include("../../src/Thesis.jl")
using .Thesis.DataIO
using .Thesis.FiniteStates
using .Thesis.SpinModels
using .Thesis.CellularAutomata
using .Thesis.Names

# Magnetization time series matrix
@inline magnet_ts_matrix!(ising::AbstractIsingModel, β::Real, n_steps::Integer, n_samples::Integer)::Matrix{Float64} = hcat(map(1:n_samples) do _
    set_state!(ising.state, SpinHalfState.up)
    return metropolis_measure!(SpinModels.magnet, ising, β, n_steps)
end...)

# Parameters
const N = 10000
const z = 4
const β = 1 / 4
const n_steps = 128
const n_runs = 16

@info "Constructing system..."
system = IsingModel(MeanFieldFiniteState(N, z, SpinHalfState.up))

@info "Calculating time series..."
m_ts = vec(mean(magnet_ts_matrix!(system, β, n_steps, n_runs), dims=2))

script_show(m_ts)

@info "Plotting time series..."
time = 0:n_steps
plt = plot(x=time[begin+1:end], y=m_ts[begin+1:end],
    Geom.point,
    Scale.x_log10, Scale.y_log10)

draw(SVG(plotsdir("test_magnet_ts.svg"), 15cm, 10cm), plt)
