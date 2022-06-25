using Printf, Polynomials, CSV

include("magnet_ts.jl")

# Properties of the system
const dim = 2
const L = 256
const σ₀ = Int8(+1)
ca = BrassCASquareLattice(Val(dim), L, σ₀)

const n_samples = 128
const n_steps = 1024

# Fit r(p) curve
const p_fit = collect(1:8) * 0.1
const r_fit = [0.06, 0.125335, 0.194421, 0.27, 0.352734, 0.442811, 0.547001, 0.664514]
poly = Polynomials.fit(p_fit, r_fit, 2)
const Δ = 0.05

# Loop on probabilities
const p = parse(Float64, ARGS[1])
for r in range(poly(p) - Δ / 2, poly(p) + Δ / 2, 101)
    @show p r
    df = magnet_ts_sample!(ca, σ₀, p, r, n_steps, n_samples)
    # Save results
    filename = @sprintf "data/time_series/BrassCA%dD_L=%d_nsamples=%d_p=%f_r=%f.csv" dim L n_samples p r
    CSV.write(filename, df)
end
