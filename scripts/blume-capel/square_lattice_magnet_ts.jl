@doc raw"""
    Blume-Capel spin model magnetization time series analysis
"""

# Dr Watson helper
using DrWatson
@quickactivate "phd"

# External libraries
using Logging, CSV, DataFrames

# Custom modules
include("../../src/Thesis.jl")
using .Thesis.DataIO
using .Thesis.FiniteStates
using .Thesis.SpinModels

# Magnetization time series matrix
@inline magnet_ts_matrix!(spinmodel::AbstractSpinModel{<:AbstractFiniteState{SpinOneState.T}}, β::Real, n_steps::Integer, n_samples::Integer)::Matrix{Float64} =
    hcat(map(1:n_samples) do _
        set_state!(spinmodel.state, SpinOneState.up)
        return metropolis_measure!(SpinModels.magnet, spinmodel, β, n_steps)
    end...)

# System parameters
const dim = 2
const L = 512

blumecapel = BlumeCapelIsotropicModel(SquareLatticeFiniteState(Val(dim), L, SpinOneState.up))


# Load temperatures table
const df = DataFrame(CSV.File("tables/butera_and_pernici_2018/blume-capel_square_lattice.csv"))
const df_row = df[only(findall(==(0), df.anisotropy_field)), :]
script_show(df_row)

const betas = []

const n_steps = 128
const n_runs = 16
