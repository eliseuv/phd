using Random, CSV, Logging

include("magnet_ts.jl")
include("../../grid_iter.jl")

# Calculate goodness of fit for given (p,r)
const n_samples = 128
const n_steps = 1024
const init_time = 30
get_goodness!(ca::BrassCA, p::Float64, r::Float64) = goodness_of_fit(magnet_ts_sample!(ca, σ₀, p, r, n_steps, n_samples), init_time)

# Properties of the system
const dim = 2
const L = parse(Int, ARGS[1])
const σ₀ = Int8(+1)
ca = BrassCASquareLattice(Val(dim), L, σ₀)

# Iteration of grid
const it = parse(Int, ARGS[2])

# Output data file
const filename = @sprintf "data/map/BrassCA%dD_L=%d_nsamples=%d_it=%d_pr_map.csv" dim L n_samples it
@show filename

for (p, r) in map(x -> Float64.(x), square_gird_iter(Val(dim), it))
    r2 = get_goodness!(ca, p, r)
    CSV.write(filename, (p = [p], r = [r], r2 = [r2]), append = true)
end
