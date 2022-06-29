@doc raw"""
    Calculate the eigenvalues for the normalized correlation matrices of the magnetization time series matrices for Brass cellular automaton
"""

using DrWatson

@quickactivate "phd"

using Logging, JLD2, LinearAlgebra, UnicodePlots

include("../../../../src/DataIO.jl")
include("../../../../src/Matrices.jl")
using .DataIO
using .Matrices

# Path for datafiles
data_dirpath = datadir("sims", "brass_ca", "magnet_ts", "mult_mat", "rand_start")

# Desired parameters
const params_req = Dict(
    "prefix" => "BrassCA2DMagnetTSMatrix",
    "L" => 256,
    "p" => 0.3,
    "n_runs" => 1000,
    "n_samples" => 100,
    "n_steps" => 300
)

for data_filename in readdir(data_dirpath)

    @info data_filename
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
    if haskey(data, "eigvals")
        @info "Skipping already processed file..."
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
    #     title = "Time series matrix (p = $p, r = $r)",
    #     xlabel = "i", ylabel = "t", zlabel = "mᵢ(t)",
    #     width = 125))
    # println()

    # Normalize time series matrix
    println("Normalizing time series matrix...")
    M_ts_norm_samples = map(normalize_ts_matrix, M_ts_samples)
    # display(heatmap(M_ts_norm_samples[begin],
    #     title = "Normalized time series matrix (p = $p, r = $r)",
    #     xlabel = "i", ylabel = "t", zlabel = "̄mᵢ(t)",
    #     width = 125))
    # println()

    # Calculate correlation matrix
    println("Calculating correlation matrix...")
    G_samples = map(cross_correlation_matrix, M_ts_norm_samples)
    # display(heatmap(G_samples[begin],
    #     title = "Cross correlation matrix (p = $p, r = $r)",
    #     xlabel = "i", ylabel = "j", zlabel = "Gᵢⱼ",
    #     width = 125))
    # println()

    # Calculate eigenvalues
    println("Calculating eigenvectors of correlation matrix...")
    λs = map(eigvals, G_samples)

    # Plot eigenvalues histogram
    display(histogram(vcat(λs...), nbins=64, xscale=log10,
        title="Eigenvalues of cross correlation matrix (p = $p, r = $r)",
        ylabel="λ", xlabel="ρ(λ)",
        width=125))
    println()

    # Store additional data
    # data["M_ts_norm_samples"] = M_ts_norm_samples
    # data["G_samples"] = G_samples
    data["eigvals"] = λs

    script_show(data)
    println()

    save(data_filepath, data)

end
