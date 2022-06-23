@doc raw"""
    Calculate the eigenvalues for the normalized correlation matrices of the magnetization time series matrices for Brass cellular automaton
"""

using DrWatson

@quickactivate "phd"

using Logging, JLD2, LinearAlgebra, UnicodePlots

include("../src/DataIO.jl")
include("../src/Matrices.jl")
using .DataIO
using .Matrices

# Path for datafiles
data_dirpath = datadir("ada-lovelace", "brass_ca_ts_matrix_eigvals")

# Desired parameters
const params_req = Dict(
    "prefix" => "BrassCA2DMagnetTSMatrix"
)

for data_filename in readdir(data_dirpath)

    filename_params = parse_filename(data_filename)
    # script_show(filename_params)

    # Ignore unrelated data files
    if !check_params(parse_filename(data_filename), params_req)
        continue
    end

    # Load data
    data_filepath = joinpath(data_dirpath, data_filename)
    @info "Loading data:" data_filepath
    data = load(data_filepath)

    # Skip files already processed
    if haskey(data, "eigvals")
        @info "Skipping file:" data_filename
        continue
    end

    # Fetch parameters
    params = data["Params"]
    print_dict(params)
    r = params["r"]
    p = params["p"]

    # Fetch magnet time series matrix samples
    M_ts_samples = data["M_ts_samples"]
    # display(heatmap(M_ts_samples[begin],
    #     title = "Time series matrix (r = $r)",
    #     xlabel = "i", ylabel = "t", zlabel = "mᵢ(t)",
    #     width = 125))
    # println()

    # Normalize time series matrix
    println("Normalizing time series matrix...")
    M_ts_norm_samples = map(normalize_ts_matrix, M_ts_samples)
    # display(heatmap(M_ts_norm_samples[begin],
    #     title = "Normalized time series matrix (r = $r)",
    #     xlabel = "i", ylabel = "t", zlabel = "̄mᵢ(t)",
    #     width = 125))
    # println()

    # Calculate correlation matrix
    println("Calculating correlation matrix...")
    G_samples = map(cross_correlation_matrix, M_ts_norm_samples)
    # display(heatmap(G_samples[begin],
    #     title = "Cross correlation matrix (r = $r)",
    #     xlabel = "i", ylabel = "j", zlabel = "Gᵢⱼ",
    #     width = 125))
    # println()

    # Calculate eigenvalues
    println("Calculating eigenvectors of correlation matrix...")
    λs = map(eigvals, G_samples)

    # Plot eigenvalues histogram
    # display(histogram(vcat(λs...), nbins = 64, xscale = log10,
    #     title = "Eigenvalues of cross correlation matrix (r = $r)",
    #     ylabel = "λ", xlabel = "ρ(λ)",
    #     width = 125))
    # println()

    # Store additional data
    # data["M_ts_norm_samples"] = M_ts_norm_samples
    # data["G_samples"] = G_samples
    data["eigvals"] = λs

    script_show(data)
    println()

    save(data_filepath, data)

end
