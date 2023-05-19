# Dr Watson helper
using DrWatson
@quickactivate "phd"

# External libraries
using Logging, Statistics, PyCall, CSV, DataFrames, LinearAlgebra, LaTeXStrings

# My libs
include("../../../../src/Thesis.jl")
using .Thesis.Metaprogramming
using .Thesis.DataIO
using .Thesis.Stats

py"""
import pickle

def load_pickle(fpath):
    with open(fpath, "rb") as f:
        data = pickle.load(f)
    return data
"""

@inline load_pickle = py"load_pickle"

function get_datafiles_dict(dir_path)
    datafiles_dict = Dict()
    for (root, _, filenames) in walkdir(dir_path)
        for filename in filenames
            path = joinpath(root, filename)
            datafile = DataIO.DataFile(path)
            D = Float64(datafile.params["D"])
            T = Float64(datafile.params["T"])
            if haskey(datafiles_dict, D)
                datafiles_dict[D][T] = datafile
            else
                datafiles_dict[D] = Dict(T => datafile)
            end
        end
    end
    return datafiles_dict
end

const df_temperatures = DataFrame(CSV.File(projectdir("tables", "butera_and_pernici_2018", "blume-capel_square_lattice.csv")))

# Calculate fluctuations using histograms
@inline function hist_fluctuations(values::AbstractVector, nbins::Integer)
    hist = Histogram(values, nbins)
    mean_value = mean(hist)
    return (mean_value, var(hist, mean=mean_value))
end

# @inline make_ticks_log(powers::AbstractVector{<:Real}, base::Integer=10) = (Float64(base) .^ powers, (map(x -> latexstring("$(base)^{$(x)}"), powers)))
@inline make_ticks_log(powers::AbstractVector{<:Real}, base::Integer=10) = (powers, (map(x -> latexstring("$(base)^{$(x)}"), powers)))
@inline make_ticks(powers::AbstractVector{<:Real}, base::Integer=10) = (Float64(base) .^ powers, (map(x -> latexstring("$(base)^{$(x)}"), powers)))

@inline function axis_ticks_int_range(low::Real, high::Real, length::Integer)
    x_final = floor(Int64, high)
    tick_vals = collect(ceil(Int64, low):ceil(Int64, (high - low) / length):floor(Int64, high))
    if tick_vals[end] != x_final
        push!(tick_vals, x_final)
    end
    return (tick_vals, map(tck -> latexstring("$(tck)"), tick_vals))
end

@inline function axis_ticks_range(low::Real, high::Real, length::Integer)
    tick_vals = map(x -> round(x, digits=5), range(low, high, length=length))
    return (tick_vals, map(tck -> latexstring("$(tck)"), tick_vals))
end

@inline axis_ticks(vals) = (vals, map(x -> latexstring("$(x)"), vals))

@inline function axis_ticks_range(low::Real, high::Real, length::Integer)
    tck = range(low, high, length=length)
    return axis_ticks(map(x -> round(x; digits=2), tck))
end

@inline function axis_ticks_range_int(low::Real, high::Real, length::Integer)
    tck = range(low, high, length=length)
    return axis_ticks(map(x -> floor(Int64, x), tck))
end

@inline function get_critical_temperature(df_temperatures, D::Real)
    df_D_row = df_temperatures[only(findall(==(D), df_temperatures.anisotropy_field)), 2:end]
    transition_order = lowercase(string(df_D_row[:transition_order]))
    crit_temp_source = findfirst(!ismissing, df_D_row)
    T_c = df_D_row[crit_temp_source]
    return (T_c, transition_order, crit_temp_source)
end

@inline get_spacings(eigvals_matrix::AbstractMatrix{<:Real}) = diff(eigvals_matrix, dims=2)
@inline function get_normalized_spacings(eigvals_matrix::AbstractMatrix{<:Real})
    eigvals_spacings = get_spacings(eigvals_matrix)
    eigvals_spacings_means = mean(eigvals_spacings, dims=2)
    return eigvals_spacings_means .\ eigvals_spacings
end

const global_prefix = "BlumeCapelSq2D"

# D values considered
const D_vals_2order = [1.0, 1.75, 1.9]
const D_val_tcp = 1.96582
const D_vals_1order = [1.9777, 1.99932488]

const D_vals = [D_vals_2order..., D_val_tcp, D_vals_1order...]

# Plot output root directory
const output_root = plotsdir("blume_capel_paper")
