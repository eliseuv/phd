using DrWatson
@quickactivate "phd"

using Logging, DataFrames, Gadfly

include("../../../src/BlumeCapelModel.jl")
using .BlumeCapelModel

# Parameters
const dim = 2
const L = 256
const β = 0.1
const n_steps = 1000

# System
@info "Creating system..."
bc = BlumeCapelSquareLattice(Val(dim), L, Val(:rand))

@info "Simulating..."
M_t = heatbath_and_measure_total_magnet!(bc, β, n_steps)

df = DataFrame(t=0:n_steps,
    M=M_t)

@info "Plotting..."
display(plot(df, x=:t, y=:M,
    Geom.line,
    Guide.xlabel("t"), Guide.ylabel("Mₜ")))
