"""
    Mangetization time series
"""
module BrassCAMagnetTS

export magnet_ts!, magnet_ts_matrix!, magnet_ts_avg!, magnet_ts_avg_parallel_loop!

using Statistics, DataFrames, DataFramesMeta, GLM

include("BrassCellularAutomaton.jl")
using .BrassCellularAutomaton

"""
    magnet_ts!(ca::BrassCA, p::Float64, r::Float64, n_steps::Int64; σ₀::BrassState = TH1)

Calculate a magnetization time series for a Brass CA `ca` that starts with all its sites with the same value.

For each time step the sites of the CA are updated in parallel.

# Arguments:
- `ca::BrassCA`: Brass CA
- `p::Float64` and `r::Float64`: Probabilities of the model
- `n_steps::Int64`: Number of time steps

# Keywords:
- `σ₀::Int8 = +1`: Initial state for all sites

# Returns:
- Vector contaning the magnetization at each time step
"""
function magnet_ts!(ca::BrassCA, p::Float64, r::Float64, n_steps::Int64; σ₀::BrassState = TH1)
    set_state!(ca, σ₀)
    advance_parallel_and_measure!(magnet, ca, p, r, n_steps)
end

"""
    magnet_ts_matrix!(ca::BrassCA, p::Float64, r::Float64, n_steps::Int64, n_samples::Int64; σ₀::BrassState = TH1)

Matrix (`n_steps+1` rows and `n_samples` columns) whose columns are different runs of the magnetization time series for a given Brass CA `ca`.

For each time step the sites of the CA are updated in parallel.

# Arguments:
- `ca::BrassCA`: Brass CA
- `p::Float64` and `r::Float64`: Probabilities of the model
- `n_steps::Int64`: Number of time steps (number of rows)
- `n_samples::Int64`: Number of samples (number of columns)

# Keywords:
- `σ₀::Int8 = +1`: Initial state for all sites

# Returns:
- Matrix contaning multiple samples of magnetization time series as columns
"""
magnet_ts_matrix!(ca::BrassCA, p::Float64, r::Float64, n_steps::Int64, n_samples::Int64; σ₀::BrassState = TH1) = hcat(ntuple(_ -> magnet_ts!(ca, p, r, n_steps; σ₀ = σ₀), n_samples)...)

"""
    magnet_ts_avg!(ca::BrassCA, p::Float64, r::Float64, n_steps::Int64, n_samples::Int64; σ₀::BrassState = TH1)

Calculate the average magnetization time series for a Brass CA that starts with all its sites with the same value over many samples.

For each time step the sites of the CA are updated in parallel.

# Arguments:
- `ca::BrassCA`: Brass CA
- `p::Float64` and `r::Float64`: Probabilities of the model
- `n_steps::Int64`: Number of time steps
- `n_samples::Int64`: Number of system samples

# Keywords:
- `σ₀::Int8 = +1`: Initial state for all sites

# Returns:
- Data frame contaning the mean and variance of magnetization at each time step
"""
function magnet_ts_avg!(ca::BrassCA, p::Float64, r::Float64, n_steps::Int64, n_samples::Int64; σ₀::BrassState = TH1)
    magnet_samples = magnet_ts_matrix!(ca, p, r, n_steps, n_samples; σ₀ = σ₀)
    magnet_mean = vec(mean(magnet_samples, dims = 2))
    magnet_var = vec(var(magnet_samples, dims = 2, mean = magnet_mean))
    return DataFrame(Time = 0:n_steps,
        Mean = magnet_mean,
        Variance = magnet_var)
end

"""
    magnet_ts_avg_parallel_loop!(ca::BrassCA, p::Float64, r::Float64, n_steps::Int64, n_samples::Int64; σ₀::BrassState = TH1)

Calculate the average magnetization time series for a Brass CA that starts with all its sites with the same value over many samples.

Each system is evolved in parallel.

# Arguments:
- `ca::BrassCA`: Brass CA
- `p::Float64` and `r::Float64`: Probabilities of the model
- `n_steps::Int64`: Number of time steps
- `n_samples::Int64`: Number of system samples

# Keywords:
- `σ₀::Int8 = +1`: Initial state for all sites

# Returns:
- Data frame contaning the mean and variance of magnetization at each time step
"""
function magnet_ts_avg_parallel_loop!(ca::BrassCA, p::Float64, r::Float64, n_steps::Int64, n_samples::Int64; σ₀::BrassState = TH1)
    magnet_samples = Matrix{Float64}(undef, n_steps + 1, n_samples)
    cas = [deepcopy(ca) for _ in 1:Threads.nthreads()]
    @inbounds Threads.@threads for j in 1:n_samples
        tid = Threads.threadid()
        set_state!(cas[tid], σ₀)
        magnet_samples[:, j] = advance_and_measure!(magnet, cas[tid], p, r, n_steps)
    end
    magnet_mean = vec(mean(magnet_samples, dims = 2))
    magnet_var = vec(var(magnet_samples, dims = 2, mean = magnet_mean))
    return DataFrame(Time = 0:n_steps,
        Mean = magnet_mean,
        Variance = magnet_var)
end

"""
Calculate the average F₂ time series for a Brass CA that starts with all its sites with the same value over many samples.

# Arguments:
- `ca`: Brass CA
- `p` and `r`: Probabilities of the model
- `n_steps`: Number of time steps
- `n_samples`: Number of system samples
- `σ₀ = +1`: Initial state for all sites

# Returns:
- Data frame contaning the value of F₂ at each time step
"""
function magnet_F2_ts_sample!(ca::BrassCA, p::Float64, r::Float64, n_steps::Int64, n_samples::Int64; σ₀::BrassState = TH1)
    magnet_samples = hcat([magnet_ts!(ca, p, r, n_steps, σ₀ = σ₀) for _ in 1:n_samples]...)
    display(magnet_samples)
    magnet_mean = vec(mean(magnet_samples, dims = 2))
    display(magnet_mean)
    magnet_mean_of_squares = vec(mean(magnet_samples .^ 2, dims = 2))
    display(magnet_mean_of_squares)
    DataFrame(Time = 0:n_steps,
        F2 = (magnet_mean_of_squares ./ magnet_mean))
end

# Perform power law fit on a given time series. Returns the pair (exponent, r2_goodness_of_fit).
function time_series_plaw_fit(df_in::DataFrame, t_label::Symbol, y_label::Symbol, t_init::Int, t_final::Int = maximum(df_in[!, t_label]))::NTuple{2,Float64}
    # Filter range
    df = @subset(df_in, t_label .>= t_init, t_label .<= t_final)
    # Ignore negative values
    df[!, y_label] = ifelse.(df[!, y_label] .< 0, missing, df[!, y_label])
    # Do not perform fit if no data
    if all(ismissing, df[!, y_label])
        println("Error: No data.")
        return 0
    end
    # LogLog
    df[!, :log_t] = log.(df[!, t_label])
    df[!, :log_y] = log.(df[!, y_label])
    # Linear regression
    lr = lm(@formula(log_y ~ log_t), df)
    (coef(lr)[2], r2(lr))
end

# Calculate power law fit on magnet time series
magnet_ts_plaw_fit(df_in::DataFrame, t_init::Int, t_final::Int = maximum(df_in[!, :Time])) = time_series_plaw_fit(df_in, :Time, :Mean, t_init, t_final)

# Calculate exponent and goodness of fit
function magnet_dynamic_critical_exponent(df_in::DataFrame, t_init::Int, t_final::Int = nrow(df_in[!, :Time]); dim::Int = 2)::NTuple{2,Float64}
    (x, G) = time_series_plaw_fit(df_in, :Time, :F2, t_init, t_final)
    z = dim / x
    (z, G)
end

function critical_exponent_F2(df_in::DataFrame, t_init::Int, t_final::Int = nrow(df_in); dim::Int = 2)::NTuple{2,Float64}
    # Filter range
    df = @subset(df_in, :Time .>= t_init, :Time .<= t_final)
    # Discard negative values
    df[!, :Mean] = ifelse.(df[!, :Mean] .< 0, missing, df[!, :Mean])
    # Do not perform fit if no data
    if all(ismissing, df[!, :Mean])
        println("Whole column Mean missing")
        return 0
    end
    # <M>²
    square_of_mean = df[!, :Mean] .^ 2
    # Var(M) = <M²> - <M>² ⟹ <M²> = Var(M) + <M>²
    mean_of_squares = df[!, :Variance] .+ square_of_mean
    # F₂(t) = <M²>(t) / <M>²(t) ∼ t^{dim/z}
    df[!, :F2] = mean_of_squares ./ square_of_mean
    # LogLog
    df[!, :logTime] = log.(df[!, :Time])
    df[!, :logF2] = log.(df[!, :F2])
    # Linear regression
    lr = lm(@formula(logF2 ~ logTime), df)
    z = dim / coef(lr)[2]
    (z, r2(lr))
end

# Calculate exponent and goodness of fit
function critical_exponent(df_in::DataFrame, t_init::Int, t_final::Int = nrow(df_in); dim::Int = 2)::NTuple{2,Float64}
    # Filter range
    df = @subset(df_in, :Time .>= t_init, :Time .<= t_final)
    # Discard negative values
    df[!, :F2] = ifelse.(df[!, :F2] .< 0, missing, df[!, :F2])
    # Do not perform fit if no data
    if all(ismissing, df[!, :F2])
        println("Whole column F₂ missing")
        return 0
    end
    # LogLog
    df[!, :logTime] = log.(df[!, :Time])
    df[!, :logF2] = log.(df[!, :F2])
    # Linear regression
    lr = lm(@formula(logF2 ~ logTime), df)
    z = dim / coef(lr)[2]
    (z, r2(lr))
end

end
