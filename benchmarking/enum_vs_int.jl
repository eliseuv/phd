using BenchmarkTools

BenchmarkTools.DEFAULT_PARAMETERS.overhead = BenchmarkTools.estimate_overhead()
BenchmarkTools.DEFAULT_PARAMETERS.samples = 100000
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 10

SpinType = Int8

@enum SpinState::SpinType begin
    up = +1
    down = -1
end

const dim = 3
const L = 256

arr_enum = rand(instances(SpinState), ntuple(_ -> L, Val(dim)))
arr_int = Integer.(arr_enum)

idx = CartesianIndex(rand(1:L, dim)...)

function flip!(arr::Array{T,N}, idx::CartesianIndex{N}) where {T<:Integer,N}
    arr[idx] = -arr[idx]
end

function flip!(arr::Array{SpinState,N}, idx::CartesianIndex{N}) where {N}
    arr[idx] = SpinState(-Integer(arr[idx]))
end

Base.sum(arr::Array{SpinState}) = sum(Integer.(arr))

mysum(arr::Array{SpinState}) = sum(Integer, arr)

@show flip!(arr_int, idx)
@btime flip!($arr_int, $idx)

@show flip!(arr_enum, idx)
@btime flip!($arr_enum, $idx)

@show sum(arr_int)
@btime sum($arr_int)

@show sum(arr_enum)
@btime sum($arr_enum)

@show mysum(arr_enum)
@btime mysum($arr_enum)
