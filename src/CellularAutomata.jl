@doc raw"""


"""
module CellularAutomata

export CellularAutomaton

using Random

include("Metaprogramming.jl")
include("Geometry.jl")

using .Metaprogramming

"""
    CellularAutomaton

Supertype for all cellular automata
"""
abstract type CellularAutomaton end

function advance!(ca::CellularAutomaton, n_steps::Integer = 1)
    @assert n_steps > 0 "Number of steps must be positive."
    # Auxiliar state
    σ′ = similar(ca.σ)
    # Time steps iteration
    @inbounds for _ in 1:floor(Int, n_steps / 2)
        step!(ca, ca.σ, σ′)
        step!(ca, σ′, ca.σ)
    end
    if isodd(n_steps)
        step!(ca, ca.σ, σ′)
        ca.σ = σ′
    end
end

function advance_parallel!(ca::CellularAutomaton, n_steps::Integer = 1)
    @assert n_steps > 0 "Number of steps must be positive."
    # Auxiliar state
    σ′ = similar(ca.σ)
    # Time steps iteration
    @inbounds for _ in 1:floor(Int, n_steps / 2)
        step_parallel!(ca, ca.σ, σ′)
        step_parallel!(ca, σ′, ca.σ)
    end
    if isodd(n_steps)
        step_parallel!(ca, ca.σ, σ′)
        ca.σ = σ′
    end
end

end
