# Dr Watson helper
using DrWatson
@quickactivate "phd"

# External libraries
using Logging, Profile

# Custom modules
include("../../../src/Thesis.jl")
using .Thesis.FiniteStates
using .Thesis.SpinModels

# System parameters
const dim = Int64(2)
const L = parse(Int64, ARGS[1])
const D = parse(Float64, ARGS[2])

# Simulation parameters
const T = parse(Float64, ARGS[3])
const n_steps = parse(Int64, ARGS[4])

@show dim L D T n_steps

const beta = 1.0 / T

# Construct system
@info "Constructing system..."
system = BlumeCapelModel(SquareLatticeFiniteState(Val(dim), L, SpinOneState.T, Val(:rand)), Val(D))
@show typeof(system)

heatbath_measure!(magnet, system, beta, 1)

Profile.clear_malloc_data()

m = heatbath_measure!(magnet, system, beta, n_steps)
