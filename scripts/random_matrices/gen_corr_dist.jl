@doc raw"""
    Generate samples of time series with a given distribution o correlations.
"""

# Dr Watson helper
using DrWatson
@quickactivate "phd"

# External libraries
using Logging

# Custom modules
include("../../src/Thesis.jl")
using .Thesis.CorrelatedPairs
using .Thesis.RandomMatrices
