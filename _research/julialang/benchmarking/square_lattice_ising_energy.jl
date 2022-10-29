
using BenchmarkTools

BenchmarkTools.DEFAULT_PARAMETERS.overhead = BenchmarkTools.estimate_overhead()
BenchmarkTools.DEFAULT_PARAMETERS.samples = 100000
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 10

# System
const dim = 3
const L = 256

state = rand(Int8[-1, +1], ntuple(_ -> L, Val(dim)))

function energy1(state::Array{T,N}) where {T,N}
    # Interaction energy
    H::Int64 = 0
    # Loop on dimensions
    @inbounds for d ∈ 1:N
        # Bulk
        front_bulk = selectdim(state, d, 1:(size(state, d)-1))
        back_bulk = selectdim(state, d, 2:size(state, d))
        H -= sum(front_bulk .* back_bulk)
        # Periodic boundaries
        first_slice = selectdim(state, d, 1)
        last_slice = selectdim(state, d, size(state, d))
        H -= sum(last_slice .* first_slice)
    end
    return H
end

function energy2(state::Array{T,N}) where {T,N}
    # Interaction energy
    H::Int64 = 0
    # Loop on dimensions
    @inbounds for d ∈ 1:N
        for k ∈ 1:size(state, d)
            H -= sum(selectdim(state, d, k) .* selectdim(state, d, mod1(k + 1, size(state, d))))
        end
    end
    return H
end

@show energy1(state)
@btime energy1($state)
@show energy2(state)
@btime energy2($state)
