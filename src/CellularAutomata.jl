@doc raw"""
    Cellular Automata


"""
module CellularAutomata

export CellularAutomaton

using EnumX, Random

include("Lattices.jl")

"""
    CellularAutomaton{T<:AbstractArray}

Supertype for all cellular automata
"""
abstract type AbstractCellularAutomaton{T,N} <: AbstractArray{T,N} end

@inline state(ca::AbstractCellularAutomaton) = ca.state

function step!(ca::AbstractCellularAutomaton{T}, state::T, state′::T) where {T<:AbstractArray}
    @inbounds Threads.@threads for i in eachindex(state(ca))
        state′[i] = new_site_state(ca, state, i)
    end
end

function advance!(ca::AbstractCellularAutomaton, n_steps::Integer=1)
    @assert n_steps > 0 "Number of steps must be positive."
    # Auxiliar state
    state′ = similar(ca.state)
    # Time steps iteration
    @inbounds for _ in 1:floor(Int, n_steps / 2)
        step!(ca, ca.state, state′)
        step!(ca, state′, ca.state)
    end
    if isodd(n_steps)
        step!(ca, ca.state, state′)
        ca.state = state′
    end
end

@enumx BrassState::Int8 begin
    TH0 = 0
    TH1 = +1
    TH2 = -1
end

struct BrassCA{N} <: AbstractCellularAutomaton{BrassState.T,N}

    "State of Brass CA"
    state::AbstractCAState{BrassState.T,N}

    "Parameters of the model"
    p::Real
    r::Real

end



end
