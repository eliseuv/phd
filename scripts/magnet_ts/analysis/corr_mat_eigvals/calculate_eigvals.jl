@doc raw"""
    Calculate the eigenvalues for the normalized correlation matrices of the magnetization time series matrices.
"""

using DrWatson

@quickactivate "phd"

using Logging, JLD2, LinearAlgebra, UnicodePlots

include("../../../../src/Thesis.jl")
using .Thesis.DataIO
using .Thesis.TimeSeries

# Path for datafiles
data_dirpath = datadir("sims", "brass_ca", "magnet_ts", "mult_mat", "rand_start")

# Desired parameters
const prefix_req = "BrassCA2DMagnetTSMatrix"
const params_req = Dict(
    # "dim" => 2,
    "L" => 100,
    "p" => 0.3,
    "n_runs" => 1000,
    "n_samples" => 100,
    "n_steps" => 300
)

for data_filename in readdir(data_dirpath)

    @info data_filename
    (filename_prefix, filename_params) = parse_filename(data_filename)
    # script_show(filename_params)

    # Ignore unrelated data files
    if filename_prefix != prefix_req || !check_params(filename_params, params_req)
        @info "Skipping unrelated file..."
        continue
    end

    # Load data
    data_filepath = joinpath(data_dirpath, data_filename)
    @info "Loading file..."
    data = load(data_filepath)

    # Skip files already processed
    # if haskey(data, "eigvals")
    #     @info "Skipping already processed file..."
    #     continue
    # end

    # Fetch parameters
    params = data["Params"]
    @show params
    dim = 2
    L = params["L"]
    r = params["r"]

    # Fetch magnet time series matrix samples
    M_ts_samples = data["M_ts_samples"]
    M_ts_samples = M_ts_samples ./ (L^dim)
    # display(heatmap(M_ts_samples[begin],
    #     title="Time series matrix (β = $β)",
    #     xlabel="i", ylabel="t", zlabel="mᵢ(t)",
    #     width=110))
    # println()

    # Normalize time series matrix
    println("Normalizing time series matrix...")
    M_ts_norm_samples = map(normalize_ts_matrix, M_ts_samples)
    # display(heatmap(M_ts_norm_samples[begin],
    #     title="Normalized time series matrix (β = $β)",
    #     xlabel="i", ylabel="t", zlabel="̄mᵢ(t)",
    #     width=110))
    # println()

    # Calculate correlation matrix
    println("Calculating correlation matrix...")
    G_samples = map(cross_correlation_matrix, M_ts_norm_samples)
    # display(heatmap(G_samples[begin],
    #     title="Cross correlation matrix (β = $β)",
    #     xlabel="i", ylabel="j", zlabel="Gᵢⱼ",
    #     width=110))
    # println()

    # Calculate eigenvalues
    println("Calculating eigenvectors of correlation matrix...")
    λs = map(eigvals, G_samples)

    # Plot eigenvalues histogram
    display(histogram(vcat(λs...), nbins=64, xscale=log10,
        title="Eigenvalues of cross correlation matrix (r = $r)",
        ylabel="λ", xlabel="ρ(λ)",
        width=110))
    println()

    # Store additional data
    # data["M_ts_norm_samples"] = M_ts_norm_samples
    # data["G_samples"] = G_samples
    data["eigvals"] = λs

    script_show(data)
    println()

    save(data_filepath, data)

end
