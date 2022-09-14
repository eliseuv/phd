@doc raw"""

"""

using DrWatson

@quickactivate "phd"

using Logging, SHA, DataFrames, CSV

include("../../../../src/Thesis.jl")
using .Thesis.DataIO
using .Thesis.FiniteStates
using .Thesis.CellularAutomata
using .Thesis.Names
using .Thesis.Measurements

# System parameters
const dim = 2
const L = parse(Int64, ARGS[1])
const p = parse(Float64, ARGS[2])
const r = parse(Float64, ARGS[3])
const n_steps = 300
const n_samples = 128

# Create system
ca = BrassCellularAutomaton(SquareLatticeFiniteState(Val(dim), L, BrassState.TH0), p, r)

# Calculate dynamical exponent
(z, r2) = fit_dynamic_exponent!(ca, n_steps, n_samples)

# Output data path
output_data_path = datadir("sims", "brass_ca", "magnet_ts", "time_series")
mkpath(output_data_path)

output_file_path = joinpath(output_data_path, "test.csv")

CSV.write(output_file_path, DataFrame(p=p, r=r, z=z, r2=r2), append=true)
