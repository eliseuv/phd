# Dr Watson helper
using DrWatson
@quickactivate "phd"

# External libraries
using Logging, Statistics, PyCall, CSV, DataFrames, LinearAlgebra, LaTeXStrings

# My libs
include("../../src/Thesis.jl")
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

# @inline function axis_ticks_range(low::Real, high::Real, length::Integer)
#     tick_vals = map(x -> round(x, digits=5), range(low, high, length=length))
#     return (tick_vals, map(tck -> latexstring("$(tck)"), tick_vals))
# end

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

@inline function get_critical_temperature_info(df_temperatures, D::Real)
    df_D_row = df_temperatures[only(findall(==(D), df_temperatures.anisotropy_field)), 2:end]
    transition_order = lowercase(string(df_D_row[:transition_order]))
    transition_order_str = replace(transition_order,
        "first" => "1st order",
        "second" => "2nd order",
        "tcp" => "TCP")
    crit_temp_source = findfirst(!ismissing, df_D_row)
    crit_temp_source_str = replace(string(crit_temp_source), "_" => " ")
    T_c = df_D_row[crit_temp_source]
    return (T_c, transition_order_str, crit_temp_source_str)
end

@inline get_spacings(eigvals_matrix::AbstractMatrix{<:Real}) = diff(eigvals_matrix, dims=2)
@inline function get_normalized_spacings(eigvals_matrix::AbstractMatrix{<:Real})
    eigvals_spacings = get_spacings(eigvals_matrix)
    eigvals_spacings_means = mean(eigvals_spacings, dims=2)
    return eigvals_spacings_means .\ eigvals_spacings
end

function discrete_first_derivative(x::AbstractVector{T}, y::AbstractVector{T}) where {T<:Real}
    @assert length(x) == length(y)
    N = length(x)
    x_out = x[2:N-1]
    y_out = Vector{T}(undef, N - 2)
    for k ∈ 2:N-1
        y_out[k-1] = (y[k+1] - y[k-1]) / (x[k+1] - x[k-1])
    end
    return (x_out, y_out)
end

function discrete_second_derivative(x::AbstractVector{T}, y::AbstractVector{T}) where {T<:Real}
    @assert length(x) == length(y)
    N = length(x)
    x_out = x[2:N-1]
    y_out = Vector{T}(undef, N - 2)
    for k ∈ 2:N-1
        y_out[k-1] = 2 * ((x[k] - x[k-1]) * (y[k+1] - y[k]) - (x[k+1] - x[k]) * (y[k] - y[k-1])) / ((x[k+1] - x[k-1]) * (x[k+1] - x[k]) * (x[k] - x[k-1]))
    end
    return (x_out, y_out)
end

# Temperatures considered
const T_idxs = [1, 3, 5, 6, 8, 10, 11, 19, 21]
