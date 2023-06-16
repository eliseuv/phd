# Dr Watson helper
using DrWatson
@quickactivate "phd"

# External libraries
using Logging, Statistics, CSV, DataFrames, LinearAlgebra, LaTeXStrings

# My libs
include("../../src/Thesis.jl")
using .Thesis.DataIO
using .Thesis.Stats

function make_param_dict(datafiles::AbstractVector{DataFile}, param::String, ::Type{T})::Dict{T,Vector{DataFile}} where {T}
    param_vals = map(x -> x.params[param], datafiles) |> unique
    dict = Dict{T,Vector{DataFile}}()
    for val ∈ param_vals
        dict[val] = filter(x -> x.params[param] == val, datafiles)
    end
    return dict
end

@info "Loading datafiles..."
const datafiles_dir = datadir("magnet_ts_wishart", "blume-capel_3d", "eigvals")
const datafiles_prefix = "BlumeCapelSq3DEigvals"
const datafiles = find_datafiles(datafiles_dir, datafiles_prefix, "L" => 20; ext=".pickle")

const datafiles_dict = make_param_dict(datafiles, "D", Float64)
