module Measurements

export
    magnet_ts!,
    magnet_ts_avg!, magnet_ts_var!

using Statistics

using ..CellularAutomata
using ..FiniteStates
using ..SpinModels
using ..TimeSeries

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

@inline magnet_ts_matrix!(ca::AbstractCellularAutomaton{<:AbstractFiniteState{T}}, σ₀::T, n_steps::Integer, n_samples::Integer) where {T} =
    hcat(map(1:n_samples) do _
        set_state!(ca.state, σ₀)
        return magnet_ts!(ca, n_steps)
    end...)

@inline magnet_ts_matrix!(ca::AbstractCellularAutomaton{<:AbstractFiniteState{T}}, ::Val{:rand}, n_steps::Integer, n_samples::Integer) where {T} =
    hcat(map(1:n_samples) do _
        randomize_state!(ca.state)
        return magnet_ts!(ca, n_steps)
    end...)


# Magnetization average time series
@inline magnet_ts_avg!(ca::BrassCellularAutomaton, n_steps::Integer, n_samples::Integer) = vec(mean(magnet_ts_matrix!(ca, BrassState.TH1, n_steps, n_samples), dims=2))

# Magnetization variance time series
@inline magnet_ts_var!(ca::BrassCellularAutomaton, n_steps::Integer, n_samples::Integer) = vec(var(magnet_ts_matrix!(ca, Val(:rand), n_steps, n_samples), dims=2))

end
