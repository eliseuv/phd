using DrWatson
@quickactivate "phd"

using Logging, DataFrames, Gadfly

include("../../../src/SpinModels.jl")
using .SpinModels

# Parameters
const dim = 2
const L = 256
# const D = 0
# const β = Inf
const D = 1.96582
const β = 1 / (0.60858)
const n_steps = 100

# System
@info "Creating system..."
blumecapel = BlumeCapelModel(SquareLatticeSpinState(Val(dim), L, SpinOneState.T, Val(:rand)), D)

H = energy(blumecapel)
println("Energy = $H")

@info "Simulating..."
M_t = heatbath_measure!(energy, blumecapel, β, n_steps)

df = DataFrame(t=0:n_steps,
    M=M_t)

@info "Plotting..."
display(plot(df, x=:t, y=:M,
    Geom.line,
    Guide.xlabel("t"), Guide.ylabel("Mₜ")))
