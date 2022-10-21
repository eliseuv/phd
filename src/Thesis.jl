@doc raw"""
    Thesis

Main module for collecting all submodules necessary for the simulations.
"""
module Thesis

# Metaprogramming macros and utils
include("Metaprogramming.jl")

# Data files utils
include("DataIO.jl")

# Lattices specialized methods
include("Lattices.jl")

# Finite state
include("FiniteStates.jl")

# Cellular automata models
include("CellularAutomata.jl")

# Spin models
include("SpinModels.jl")

# Time series utils
include("RandomMatrices.jl")

# Measurements
include("Measurements.jl")

end
