@doc raw"""
    Generate a magnetization time series
"""

# Dr Watson helper
using DrWatson
@quickactivate "phd"

# External libraries
using Logging, StatsBase, Distributions, Distances

# Custom modules
include("../../src/Thesis.jl")
using .Thesis.TimeSeries

# Parameters
# Time series matrix
const n_series = 128
const t_max = 256
# Target correlation distribution
const ρ_dist = Uniform(-1.0, 1.0)

# Generate time series matrix
M_ts = rand(Normal(), t_max, n_series)

# Cost calculation
@inline function cost(M_ts, ρ_dist::UnivariateDistribution)
    hist = normalize(fit(Histogram, cross_correlation_values(M_ts), nbins=128), mode=:pdf)
    x_target = map(x -> pdf(ρ_dist, x), hist.edges[begin][begin:end-1])
    x_emp = hist.weights
    chisq_dist(x_emp, x_target)
end

@inline function cost_to_uniform(M_ts)
    n_bins = 128
    hist = normalize(fit(Histogram, cross_correlation_values(M_ts), nbins=n_bins), mode=:pdf)
    x_target = fill(0.5, n_bins)
    x_emp = hist.weights
    chisq_dist(x_emp, x_target)
end

@inline function perturbate!(M_ts, k, σ)

end

# Metropolis sampling applied
function metropolis!(M_ts, β, n_iter)
    n_series = size(M_ts, 2)
    M_ts′ = deepcopy(M_ts)
    # Vector to store cost at each iteration
    costs = Vector{Float64}(undef, n_iter)
    # Calculate inital cost
    costs[1] = cost_to_uniform(M_ts)
    for it ∈ 2:n_iter+1
        for k ∈ rand(1:n_series, n_series)
            perturbate!(M_ts′, k, 0.1)
            cost = cost_to_uniform(M_ts′)
            Δcost = cost - costs[it-1]
            # Metropolis prescription
            if Δcost <= 0 || exp(-β * Δcost) > rand()
                # Accept

            else

            end
        end
    end
end
