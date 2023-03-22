@doc """
    Clone of fortran code
"""
# External libraries
using Logging, Profile

# System parameters
const dim = 2
const L = parse(Int64, ARGS[1])
const D = parse(Float64, ARGS[2])

# Simulation parameters
const T = parse(Float64, ARGS[3])
const n_steps = parse(Int64, ARGS[4])

@show dim L D T n_steps

const beta = 1.0 / T

function nn_array(L::Int64)
    nn = Matrix{Int64}(undef, L, 2)
    for i ∈ 1:L
        nn[i, 1] = i - 1
        nn[i, 2] = i + 1
    end
    nn[1, 1] = L
    nn[L, 2] = 1
    return nn
end

@inline magnet(state::Matrix{Int64}) = sum(state) / length(state)

@inline function mcmove!(state::Matrix{Int64}, D::Float64, beta::Float64, nn::Matrix{Int64}, seed::Int64)
    for _ ∈ 1:length(state)
        # Select random site
        (ix, iy) = rand(1:L, 2)
        # Get neighbor spin values
        sx1 = state[nn[ix, 1], iy]
        sx2 = state[nn[ix, 2], iy]
        sy1 = state[ix, nn[iy, 1]]
        sy2 = state[ix, nn[iy, 2]]
        # sx1 = state[mod1(ix - 1, Lx), iy]
        # sx2 = state[mod1(ix + 1, Lx), iy]
        # sy1 = state[ix, mod1(iy - 1, Ly)]
        # sy2 = state[ix, mod1(iy + 1, Ly)]
        # Nearest neighbors sum
        nn_sum = Float64(sx1 + sx2 + sy1 + sy2)

        # Energy values
        E_down = nn_sum + D
        #E_zero = 0.0
        #E_up = -nn_sum + D

        # Heatbath weights
        W_down = exp(-beta * E_down)
        #W_zero = exp(-beta * 0.0) = 1.0
        W_up = exp(-beta * E_down)
        W_total = W_down + 1.0 + W_up

        # Probabilities
        P_down = W_down / W_total
        P_zero = 1.0 / W_total

        # Heatbath prescription
        # rnd = rand()
        rnd = ran2!(seed)
        state[ix, iy] = if rnd < P_down
            -1
        elseif rnd < P_down + P_zero
            0
        else
            +1
        end
    end
end

function heatbath_measure_magnet!(m::Vector{Float64}, state::Matrix{Int64}, D::Float64, beta::Float64, nn::Matrix{Int64}, seed::Int64)
    n_steps = length(m) - 1
    m[1] = magnet(state)
    for k ∈ 2:n_steps+1
        mcmove!(state, D, beta, nn, seed)
        m[k] = magnet(state)
    end
end

@inline function ran2!(idum::Int64)
    # Constant parameters
    im1 = 2147483563
    im2 = 2147483399
    am = 1.0 / im1
    imm1 = im1 - 1
    ia1 = 40014
    ia2 = 40692
    iq1 = 53668
    iq2 = 52774
    ir1 = 12211
    ir2 = 3791
    ntab = 32
    ndiv = 1 + imm1 / ntab
    eps = 1.2e-7
    rnmx = 1.0 - eps

    iv = Vector{Int64}(undef, ntab)

    if idum < 0
        idum = max(-idum, 1)
        idum2 = idum
        for j ∈ range(ntab + 8, stop=1, step=-1)
            k = idum / iq1
            idum = ia1 * (idum - k * iq1) - k * ir1
            if idum < 0
                idum = idum + im1
            end
            if j <= ntab
                iv[j] = idum
            end
        end
        iy = iv[1]
    end
    k = idum / iq1
    idum = ia1 * (idum - k * iq1) - k * ir1
    if idum < 0
        idum = idum + im1
    end
    k = idum2 / iq2
    idum2 = ia2 * (idum2 - k * iq2) - k * ir2
    if idum2 < 0
        idum2 = idum2 + im2
    end
    j = 1 + iy / ndiv
    iy = iv[j] - idum2
    iv[j] = idum
    if iy < 1
        iy = iy + imm1
    end
    return min(am * iy, rnmx)
end

iseed = 1

# Initialize state
state = rand(-1:1, L, L)

# Create nearest neighbors array
nn = nn_array(L)

# Allocate magnetization vector
m = Vector{Float64}(undef, n_steps + 1)

heatbath_measure_magnet!(Vector{Float64}(undef, 1), state, D, beta, nn, iseed)

@time heatbath_measure_magnet!(m, state, D, beta, nn, iseed)
