using Statistics, DataFrames, GLM

include("../Brass.jl")
using .Brass

# Magnetization time series for a single CA
function magnet_ts!(ca::BrassCA, σ₀::Int8, p::Float64, r::Float64, n_steps::Int64)::Vector{Float64}
    set_state!(ca, σ₀)
    advance_and_measure!(magnet, ca, p, r, n_steps)
end

# Calculate magnetization average time series over multiple samples
function magnet_ts_sample!(ca::BrassCA, σ₀::Int8, p::Float64, r::Float64, n_steps::Int64, n_samples::Int64)
    magnet_samples = hcat([magnet_ts!(ca, σ₀, p, r, n_steps) for _ in 1:n_samples]...)
    magnet_mean = vec(mean(magnet_samples, dims = 2))
    magnet_var = vec(varm(magnet_samples, magnet_mean, dims = 2))
    DataFrame(Time = 0:n_steps,
        Mean = magnet_mean,
        Variance = magnet_var)
end

# Calculate exponent and goodness of fit
function critical_exponent(df::DataFrame, t₀::Int)::NTuple{2,Float64}
    # Ignore negative values
    df[!, :Mean] = ifelse.(df[!, :Mean] .< 0, missing, df[!, :Mean])
    # LogLog
    df[!, :logTime] = log.(df[!, :Time])
    df[!, :logMean] = log.(df[!, :Mean])
    # Linear regression
    lr = lm(@formula(logMean ~ logTime), last(df, nrow(df) - t₀))
    (coef(lr)[2], r2(lr))
end

# Calculate goodness of fit
function goodness_of_fit(df::DataFrame, t₀::Int)::Float64
    # Ignore negative values
    df[!, :Mean] = ifelse.(df[!, :Mean] .< 0, missing, df[!, :Mean])
    # LogLog
    df[!, :logTime] = log.(df[!, :Time])
    df[!, :logMean] = log.(df[!, :Mean])
    # Linear regression
    lr = lm(@formula(logMean ~ logTime), last(df, nrow(df) - t₀))
    r2(lr)
end
