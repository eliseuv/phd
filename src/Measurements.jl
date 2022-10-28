module Measurements

export
    magnet_ts!, magnet_ts_matrix!,
    magnet_ts_avg!, magnet_ts_sq_avg!,
    fit_dynamic_exponent!

using Statistics, DataFrames, GLM

using ..CellularAutomata
using ..FiniteStates
using ..SpinModels

"""
###########################################
    Magnetization Time Series Measurements
###########################################
"""

"""
    magnet_ts!(ising::AbstractIsingModel, β::Real, n_steps::Integer)

Generate total magnetization time series matrix for a given Ising spin model `ising` at temperature `β` for `n_steps` steps.
"""
@inline magnet_ts!(ising::AbstractIsingModel, β::Real, n_steps::Integer) = metropolis_measure!(SpinModels.magnet_total, ising, β, n_steps)

"""
    magnet_ts!(blumecapel::AbstractBlumeCapelModel, β::Real, n_steps::Integer)

Generate total magnetization time series matrix for a given Blume-Capel spin model `blumecapel` at temperature `β` for `n_steps` steps.
"""
@inline magnet_ts!(blumecapel::AbstractBlumeCapelModel, β::Real, n_steps::Integer) = heatbath_measure!(SpinModels.magnet_total, blumecapel, β, n_steps)

"""
    magnet_ts!(ca::AbstractCellularAutomaton, n_steps::Integer)

Generate total magnetization time series matrix for a given cellular automaton `ca` of length `n_steps`.
"""
@inline magnet_ts!(ca::AbstractCellularAutomaton, n_steps::Integer) = advance_measure!(CellularAutomata.magnet_total, ca, n_steps)

"""
#####################################
    Magnetization Time Series Matrix
#####################################
"""

"""
    magnet_ts_matrix!(ca::AbstractCellularAutomaton{<:AbstractFiniteState{T}}, σ₀::T, n_steps::Integer, n_samples::Integer) where {T}

Generate a total magnetization time series matrix with `n_steps` rows and `n_samples` columns for a cellular automaton `ca` beggining with all sites at initial state `σ₀`.
"""
magnet_ts_matrix!(ca::AbstractCellularAutomaton{<:AbstractFiniteState{T}}, σ₀::T, n_steps::Integer, n_samples::Integer) where {T} =
    hcat(map(1:n_samples) do _
        set_state!(ca.state, σ₀)
        return magnet_ts!(ca, n_steps)
    end...)

"""
    magnet_ts_matrix!(ca::AbstractCellularAutomaton, ::Val{:rand}, n_steps::Integer, n_samples::Integer)

Generate a total magnetization time series matrix with `n_steps` rows and `n_samples` columns for a cellular automaton `ca` beggining with all sites at in random states.
"""
magnet_ts_matrix!(ca::AbstractCellularAutomaton, ::Val{:rand}, n_steps::Integer, n_samples::Integer) =
    hcat(map(1:n_samples) do _
        randomize_state!(ca.state)
        return magnet_ts!(ca, n_steps)
    end...)

"""
#########################################
    Magnetization Time Series Statistics
#########################################
"""

@inline magnet_ts_avg!(ca::AbstractCellularAutomaton{<:AbstractFiniteState{T}}, σ₀::T, n_steps::Integer, n_samples::Integer) where {T} = vec(mean(magnet_ts_matrix!(ca, σ₀, n_steps, n_samples), dims=2))

"""
    magnet_ts_var!(ca::AbstractCellularAutomaton, n_steps::Integer, n_samples::Integer)

TBW
"""
@inline magnet_ts_sq_avg!(ca::AbstractCellularAutomaton, n_steps::Integer, n_samples::Integer) = vec(mean(magnet_ts_matrix!(ca, Val(:rand), n_steps, n_samples) .^ 2, dims=2))


function fit_dynamic_exponent!(ca::AbstractCellularAutomaton, n_steps::Integer, n_samples::Integer)
    # System size
    dim = FiniteStates.dim(CellularAutomata.state(ca))
    N = length(CellularAutomata.state(ca))
    # Average magnetization
    M_avg = magnet_ts_avg!(ca, n_steps, n_samples) ./ N
    # Average squared magnetization
    M_sq_avg = magnet_ts_sq_avg!(ca, n_steps, n_samples) ./ (N^2)
    # F₂(t) = <M²>(t) / <M>²(t) ∼ t^{dim/z}
    F₂ = M_sq_avg ./ (M_avg .^ 2)
    # Dataframe for fitting
    df = DataFrame(logTime=log.(collect(1:n_steps)), logF₂=log.(F₂[begin+1:end]))
    # log-log fit
    lr = lm(@formula(logF₂ ~ logTime), df)
    return (dim / coef(lr)[2], r2(lr))
end

end
