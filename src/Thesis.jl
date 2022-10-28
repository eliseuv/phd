@doc raw"""
    Thesis

Main module for collecting all submodules necessary for the simulations.
"""
module Thesis

# Useful macros, quote macros and type utils
include("Metaprogramming.jl")

# Data files utils
include("DataIO.jl")

# JHU Covid-19 data
include("Covid19Data.jl")

# Correlated random variables pair
include("CorrelatedPairs.jl")

# Random matrices utils
include("RandomMatrices.jl")

# Lattice geometry utils
include("Lattices.jl")

# Finite state
include("FiniteStates.jl")

# Cellular automata models
include("CellularAutomata.jl")

# Spin models
include("SpinModels.jl")

# Measurements
include("Measurements.jl")

end
