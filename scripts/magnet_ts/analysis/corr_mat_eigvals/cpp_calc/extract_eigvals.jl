@doc raw"""
    Plot the histogram of the eigenvalues for the normalized correlation matrices of the magnetization time series matrices.
"""

using DrWatson

@quickactivate "phd"

using Logging, JLD2, StatsBase, DelimitedFiles

include("../../../../../src/DataIO.jl")
using .DataIO

# Path for datafiles
data_dirpath = datadir("sims", "ising", "magnet_ts", "mult_mat", "rand_start")

# Selected parameters
prefix = "IsingMagnetTSMatrix"
const params_req = Dict{String,Any}(
    "dim" => 2,
    "L" => 100,
    "n_runs" => 1000,
    "n_samples" => 100,
    "n_steps" => 300
)

# Loop on datafiles
for data_filename in readdir(data_dirpath)

    @info data_filename
    filename_params = parse_filename(data_filename)
    @info filename_params

    # Ignore unrelated data files
    if !check_params(filename_params, "prefix" => prefix, params_req)
        @info "Skipping unrelated file..."
        continue
    end

    # Load data
    data_filepath = joinpath(data_dirpath, data_filename)
    @info "Loading data file..."
    data = load(data_filepath)

    # Skip files without eigenvalues calculated
    if !haskey(data, "eigvals")
        @info "Skipping file without eigenvalues..." keys(data)
        continue
    end

    # Fetch parameters
    params = data["Params"]
    print_dict(params)
    β = params["beta"]

    # Fetch eigenvalues
    λs = sort(reduce(vcat, data["eigvals"]))
    # script_show(λs)
    # println()

    # Output file
    output_params = deepcopy(params_req)
    output_params["beta"] = β
    output_filepath = joinpath(data_dirpath, "csv_eigvals", filename(prefix * "Eigvals", output_params, ext=".csv"))
    mkpath(dirname(output_filepath))
    @info "Saving output file:" output_filepath
    writedlm(output_filepath, λs, ',')

end
