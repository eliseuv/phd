using DrWatson

@quickactivate "phd"

using Logging, JLD2, LinearAlgebra, UnicodePlots

include("../src/DataIO.jl")
include("../src/Matrices.jl")
using .DataIO
using .Matrices

data_dirpath = datadir("ada-lovelace", "brass_ca_ts_matrix_eigvals")

# Select parameters
const p = 0.3
const n_runs = 1000

for data_filename in readdir(data_dirpath)

    filename_params = parse_filename(data_filename)
    # script_show(filename_params)

    # Ignore unrelated data files
    if filename_params["prefix"] != "BrassCA2DMagnetTSMatrix" ||
       !haskey(filename_params, "p") || filename_params["p"] != p ||
       !haskey(filename_params, "n_runs") || filename_params["n_runs"] != n_runs
        continue
    end

    # Load data
    data_filepath = joinpath(data_dirpath, data_filename)
    @info "Loading data:" data_filepath
    data = load(data_filepath)

    # Fetch parameters
    params = data["Params"]
    print_dict(params)
    r = params["r"]

    # # Fetch magnet time series matrix samples
    M_ts_samples = data["M_ts_samples"]
    display(heatmap(M_ts_samples[begin],
        title = "Time series matrix (r = $r)",
        xlabel = "i", ylabel = "t", zlabel = "mᵢ(t)",
        width = 125))
    println()

    # # Normalize time series matrix
    M_ts_norm_samples = map(normalize_ts_matrix, M_ts_samples)
    display(heatmap(M_ts_norm_samples[begin],
        title = "Normalized time series matrix (r = $r)",
        xlabel = "i", ylabel = "t", zlabel = "̄mᵢ(t)",
        width = 125))
    println()

    # # Calculate correlation matrix
    G_samples = map(cross_correlation_matrix, M_ts_norm_samples)
    display(heatmap(G_samples[begin],
        title = "Cross correlation matrix (r = $r)",
        xlabel = "i", ylabel = "j", zlabel = "Gᵢⱼ",
        width = 125))
    println()

    # # Calculate eigenvalues
    λs = sort(vcat(map(eigvals, G_samples)...))

    # # Plot eigenvalues histogram
    display(histogram(λs, nbins = 64, xscale = log10,
        title = "Eigenvalues of cross correlation matrix (r = $r)",
        ylabel = "λ", xlabel = "ρ(λ)",
        width = 125))
    println()

    # # Store additional data
    # data["M_ts_norm_samples"] = M_ts_norm_samples
    # data["G_samples"] = G_samples
    data["eigvals"] = λs

    script_show(data)
    println()

    save(data_filepath, data)

end
