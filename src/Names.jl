module Names

export name

using ..CellularAutomata
using ..FiniteStates
using ..SpinModels

"""
    Names for finite states
"""
@inline name(::MeanFieldFiniteState) = "MeanField"
@inline name(::SquareLatticeFiniteState{T,N}) where {T,N} = "SquareLattice" * string(N) * "D"
@inline name(::SimpleGraphFiniteState) = "SimpleGraph"

"""
    Names for spin models
"""
@inline name(ising::IsingModel) = "Ising_" * name(ising.state)

"""
    Names for cellular automata
"""
@inline name(ca::BrassCellularAutomaton) = "BrassCA_" * name(ca.state)

end
