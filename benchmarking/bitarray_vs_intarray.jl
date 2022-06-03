using BenchmarkTools

BenchmarkTools.DEFAULT_PARAMETERS.overhead = BenchmarkTools.estimate_overhead()
BenchmarkTools.DEFAULT_PARAMETERS.samples = 100000
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 10

# System parameters
const dim = 1
const L = 256
const dimensions = ntuple(_ -> L, Val(dim))

@enum SpinState::Int8 begin
    up = +1
    down = -1
end

# Systems
arr_enum = rand(instances(SpinState), dimensions)
arr_bit = BitArray(σ == up for σ ∈ arr_enum)
arr_int8 = Integer.(arr_enum)
arr_int64 = Array{Int64}(arr_int8)

# Spin flip

idx = CartesianIndex(rand(1:L, dim)...)

function flip!(arr::Array{SpinState,N}, idx::CartesianIndex{N}) where {N}
    arr[idx] = SpinState(-Integer(arr[idx]))
end

function flip!(arr::BitArray{N}, idx::CartesianIndex{N}) where {N}
    arr[idx] = !arr[idx]
end

function flip!(arr::Array{T,N}, idx::CartesianIndex{N}) where {T<:Integer,N}
    arr[idx] = -arr[idx]
end

println("\nSpin flip\n")

println("Flip array of enum")
@btime flip!($arr_enum, $idx)

println("Flip array of bits")
@btime flip!($arr_bit, $idx)

println("Flip array of Int8")
@btime flip!($arr_int8, $idx)

println("Flip array of Int64")
@btime flip!($arr_int64, $idx)

# Energy

Base.promote_rule(T::Type, ::Type{SpinState}) = T
Base.convert(T::Type, σ::SpinState) = T(Integer(σ))

for op in (:*, :/, :+, :-)
    @eval begin
        @inline Base.$op(x::Number, σ::SpinState) = $op(promote(x, σ)...)
        @inline Base.$op(σ::SpinState, y::Number) = $op(promote(σ, y)...)
    end
end

@inline Base.:*(σ₁::SpinState, σ₂::SpinState) = Integer(σ₁) * Integer(σ₂)

function energy(arr::Array{SpinState,N}) where {N}
    # Interaction energy
    H = zero(Int64)
    # Loop on dimensions
    @inbounds for d ∈ 1:N
        # Bulk
        front_bulk = selectdim(arr, d, 1:(size(arr, d)-1))
        back_bulk = selectdim(arr, d, 2:size(arr, d))
        H -= sum(front_bulk .* back_bulk)
        # Periodic boundaries
        last_slice = selectdim(arr, d, size(arr, d))
        first_slice = selectdim(arr, d, 1)
        H -= sum(last_slice .* first_slice)
    end
    return H
end

spin(σ::Bool) = σ ? up : down

function energy(arr::BitArray{N}) where {N}
    # Interaction energy
    H::Int64 = 0
    # Loop on dimensions
    @inbounds for d ∈ 1:N
        # Bulk
        front_bulk = selectdim(arr, d, 1:(size(arr, d)-1))
        back_bulk = selectdim(arr, d, 2:size(arr, d))
        H -= sum(Integer ∘ spin ∘ !, front_bulk .⊻ back_bulk)
        # Periodic boundaries
        last_slice = selectdim(arr, d, size(arr, d))
        first_slice = selectdim(arr, d, 1)
        H -= sum(Integer ∘ spin ∘ !, last_slice .⊻ first_slice)
    end
    return H
end

function energy(arr::Array{<:Integer,N}) where {N}
    # Interaction energy
    H::Int64 = 0
    # Loop on dimensions
    @inbounds for d ∈ 1:N
        # Bulk
        front_bulk = selectdim(arr, d, 1:(size(arr, d)-1))
        back_bulk = selectdim(arr, d, 2:size(arr, d))
        H -= sum(front_bulk .* back_bulk)
        # Periodic boundaries
        last_slice = selectdim(arr, d, size(arr, d))
        first_slice = selectdim(arr, d, 1)
        H -= sum(last_slice .* first_slice)
    end
    return H
end

println("\nEnergy calculation\n")

@show energy(arr_enum)
@btime energy($arr_enum)

@show energy(arr_bit)
@btime energy($arr_bit)

@show energy(arr_int8)
@btime energy($arr_int8)

@show energy(arr_int64)
@btime energy($arr_int64)
