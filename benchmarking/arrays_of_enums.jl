using BenchmarkTools

BenchmarkTools.DEFAULT_PARAMETERS.overhead = BenchmarkTools.estimate_overhead()
BenchmarkTools.DEFAULT_PARAMETERS.samples = 100000
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 10

# Enums
@enum SpinState::Int8 begin
    up = +1
    down = -1
end

@enum BrassState::Int8 begin
    TH0 = 0
    TH1 = +1
    TH2 = -1
end

const dim = 3
const L = 128

# println("\nArray creation")
# println("\nBase type")
# @btime rand($Integer.(instances(SpinState)), $ntuple(_ -> L, Val(dim)))
# println("\nEnum type")
# @btime rand($instances(SpinState), $ntuple(_ -> L, Val(dim)))

base_array = rand(Integer.(instances(SpinState)), ntuple(_ -> L, Val(dim)))
enum_array = rand(instances(SpinState), ntuple(_ -> L, Val(dim)))

idx = 1

function flip!(state::Array{T}, i::Integer) where {T<:Integer}
    state[i] = -state[i]
end

function flip!(state::Array{SpinState}, i::Integer)
    state[i] = SpinState(-Integer(state[i]))
end
function flip2!(state::Array{SpinState}, i::Integer)
    if state[i] == up
        state[i] = down
    else
        state[i] = up
    end
end

println("\nArray access")
println("\nBase type")
@btime flip!($base_array, $idx)
println("\nEnum type")
@btime flip!($enum_array, $idx)
@btime flip2!($enum_array, $idx)
