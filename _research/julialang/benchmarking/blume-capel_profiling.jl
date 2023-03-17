include("../../../src/Thesis.jl")

using .Thesis.FiniteStates
using .Thesis.SpinModels

const dim = 2
const L = 128
const β = 1.0
const D = 1
const n_steps = 2 ^ 16

system = BlumeCapelModel(SquareLatticeFiniteState(Val(dim), L, SpinOneState.T, Val(:rand)), Val(D))

@info typeof(system)

@info "Compiling..."
metropolis_measure!(magnet, system, β, 1)

@info "Profiling..."
@profview metropolis_measure!(magnet, system, β, n_steps)
