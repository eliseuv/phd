using BenchmarkTools

include("../src/Geometry.jl")
using .Geometry

# System
const dim = 3
const L = 256
@show dim L

lattice = rand(Int8[-1, +1], ntuple(_ -> L, Val(dim)))

idx_first = CartesianIndex(ntuple(_ -> 1, Val(dim)))
idx_last = CartesianIndex(ntuple(_ -> L, Val(dim)))

BenchmarkTools.DEFAULT_PARAMETERS.overhead = BenchmarkTools.estimate_overhead()
BenchmarkTools.DEFAULT_PARAMETERS.samples = 20000
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 10

# Generated function
@info "NN generated function - First index"
@btime square_lattice_nearest_neighbors($lattice, $idx_first)
@info "NN generated function - Last index"
@btime square_lattice_nearest_neighbors($lattice, $idx_last)

# NTuple function
@info "NN NTuple function - First index"
@btime square_lattice_nearest_neighbors_($lattice, $idx_first)
@info "NN NTuple function - Last index"
@btime square_lattice_nearest_neighbors_($lattice, $idx_last)

# Generated function
@info "Generated function NN sum - First index"
@btime square_lattice_nearest_neighbors_sum($lattice, $idx_first)
@info "Generated function NN sum - Last index"
@btime square_lattice_nearest_neighbors_sum($lattice, $idx_last)

# Generated function
@info "Sum on NN generated function - First index"
@btime sum(lattice[nn] for nn ∈ square_lattice_nearest_neighbors($lattice, $idx_first))
@info "Sum on NN fenerated function - Last index"
@btime sum(lattice[nn] for nn ∈ square_lattice_nearest_neighbors($lattice, $idx_last))

# NTuple function
@info "Sum of NN NTuple function - First index"
@btime sum(lattice[nn] for nn ∈ square_lattice_nearest_neighbors_($lattice, $idx_first))
@info "Sum of NN NTuple function - Last index"
@btime sum(lattice[nn] for nn ∈ square_lattice_nearest_neighbors_($lattice, $idx_last))
