using Printf, BenchmarkTools, Graphs

BenchmarkTools.DEFAULT_PARAMETERS.overhead = BenchmarkTools.estimate_overhead()
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 10

include("./IsingModel.jl")
using .IsingModel

S = ntuple(_ -> 16, Val(3))
N = prod(S)

"""
Testing Ising square lattice
"""

println("\nIsing Square Lattice")
ising_lattice = IsingSquareLattice(S, Val(-1))

@printf "M = %d\n" magnet_total(ising_lattice)
@btime magnet_total(ising_lattice)

@printf "m = %f\n" magnet(ising_lattice)
@btime magnet(ising_lattice)

@printf "H = %f\n" energy(ising_lattice)
@btime energy(ising_lattice)

@show nearest_neighbors(ising_lattice, 1)
@btime nearest_neighbors(ising_lattice, 1)

@show energy_local(ising_lattice, 1)
@btime energy_local(ising_lattice, 1)

"""
Testing Ising complete graph
"""

println("\nIsing Complete Graph")
ising_k_graph = IsingCompleteGraph(Val(N), Val(-1))

@printf "M = %d\n" magnet_total(ising_k_graph)
@btime magnet_total(ising_k_graph)

@printf "m = %f\n" magnet(ising_k_graph)
@btime magnet(ising_k_graph)

@printf "H = %f\n" energy(ising_k_graph)
@btime energy(ising_k_graph)

@show nearest_neighbors(ising_k_graph, 1)
@btime nearest_neighbors(ising_k_graph, 1)

@show energy_local(ising_k_graph, 1)
@btime energy_local(ising_k_graph, 1)

"""
Testing Ising arbitrary graph
"""

println("\nIsing Arbitrary Graph")
ising_graph = IsingGraph(erdos_renyi(N, 3 * N), Val(-1))

@printf "M = %d\n" magnet_total(ising_graph)
@btime magnet_total(ising_graph)

@printf "m = %f\n" magnet(ising_graph)
@btime magnet(ising_graph)

@printf "H = %f\n" energy(ising_graph)
@btime energy(ising_graph)

@show nearest_neighbors(ising_graph, 1)
@btime nearest_neighbors(ising_graph, 1)

@show energy_local(ising_graph, 1)
@btime energy_local(ising_graph, 1)
