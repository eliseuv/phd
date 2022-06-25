@doc raw"""
    Calculate the eigenvalues for the normalized correlation matrices of the magnetization time series matrices for Brass cellular automaton
"""

using DrWatson

@quickactivate "phd"

using Logging, JLD2, Statistics, DataFrames, GLM, UnicodePlots

include("../src/DataIO.jl")
include("../src/Matrices.jl")
using .DataIO
using .Matrices

# Path for datafiles
data_dirpath = datadir("brass_ca_ts_matrix", "TH1_start")

# Desired parameters
const params_req = Dict(
    "prefix" => "BrassCA2DMagnetTSMatrix"
)

for data_filename in readdir(data_dirpath)

    @info "Data file:" data_filename
    filename_params = parse_filename(data_filename)
    # script_show(filename_params)

    # Ignore unrelated data files
    if !check_params(parse_filename(data_filename), params_req)
        @info "Skipping unrelated file..."
        continue
    end

    # Load data
    data_filepath = joinpath(data_dirpath, data_filename)
    @info "Loading file..."
    data = load(data_filepath)

    # Skip files already processed
    # if haskey(data, "eigvals")
    #     @info "Skipping file:" data_filename
    #     continue
    # end

    # Fetch parameters
    params = data["Params"]
    print_dict(params)
    r = params["r"]
    p = params["p"]
    n_steps = params["n_steps"]

    # Fetch magnet time series matrix samples
    M_ts_samples = data["M_ts_samples"]
    # display(heatmap(M_ts_samples[begin],
    #     title = "Time series matrix (r = $r)",
    #     xlabel = "i", ylabel = "t", zlabel = "mᵢ(t)",
    #     width = 125))
    # println()

    # Calculate average time series
    println("Calculating average time series...")
    M_ts_mean = mean(hcat(M_ts_samples...), dims = 2)
    M_ts_var = varm(hcat(M_ts_samples...), M_ts_mean, dims = 2)
    # Plot demo
    display(lineplot(0:n_steps, vec(M_ts_mean),
        title = "Mean time series (p = $p, r = $r)",
        xlabel = "t", ylabel = "⟨m⟩",
        width = 125))
    println()
    # display(lineplot(0:(length(M_ts_mean)-1), vec(M_ts_var),
    #     title = "Variance time series (p = $p, r = $r)",
    #     xlabel = "t", ylabel = "⟨m²⟩ - ⟨m⟩²",
    #     width = 125))
    # println()

    # Ignore negative values
    M_ts_mean = ifelse.(M_ts_mean .< 0, missing, M_ts_mean)
    # Store in dataframe
    df = DataFrame(Time = 0:(length(M_ts_mean)-1),
        Mean = vec(M_ts_mean),
        Var = vec(M_ts_var))
    # LogLog scale
    df[!, :logTime] = log10.(df[!, :Time])
    df[!, :logMean] = log10.(df[!, :Mean])
    # Linear regression
    t₀ = 10
    lr = lm(@formula(logMean ~ logTime), last(df, nrow(df) - t₀))
    # Get plaw exponent and goodnes of fit
    α = coef(lr)[2]
    goodness = r2(lr)

    # Store additional data
    # data["M_ts_norm_samples"] = M_ts_norm_samples
    # data["G_samples"] = G_samples

    # script_show(data)
    # println()

    # save(data_filepath, data)

end
