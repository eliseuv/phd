# Dr Watson helper
using DrWatson
@quickactivate "phd"

# External libraries
using Logging, Statistics, PyCall, CSV, DataFrames

# My libs
include("../../../src/Thesis.jl")
using .Thesis.Metaprogramming
using .Thesis.DataIO
using .Thesis.Stats
