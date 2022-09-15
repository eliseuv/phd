@doc raw"""

"""

using DrWatson

@quickactivate "phd"

using Logging, SHA, DataFrames, CSV

include("../../../../src/Thesis.jl")
using .Thesis.DataIO
using .Thesis.FiniteStates
using .Thesis.CellularAutomata
using .Thesis.Names
using .Thesis.Measurements

# System parameters
const params_combi::Dict{String} = Dict(
    "dim" => 2,
    "L" => parse(Int64, ARGS[1]),
    "p" => parse(Float64, ARGS[2]),
    "r" => collect(0.0:0.01:1.0),
    "n_steps" => 300,
    "n_samples" => 128
)

# Output dir
output_data_path = datadir("sims", "brass_ca", "magnet_ts", "time_series")
mkpath(output_data_path)
# Ouput filename
output_params = deepcopy(params_combi)
delete!(output_params, "r")
data_filename = filename("BrassCASquareLattice", output_params, ext="csv")
output_file_path = joinpath(output_data_path, data_filename)
@info "Writing data file" output_file_path
df = DataFrame(p=Float64[], r=Float64[], z=Float64[], r2=Float64[])
CSV.write(output_file_path, df)

# Serialize parameters
const params_list = dict_list(params_combi)
@info "Running $(length(params_list)) simulations"

# Loop on simulation parameters
for params in params_list
    @info params

    # Create system
    ca = BrassCellularAutomaton(SquareLatticeFiniteState(Val(params["dim"]), params["L"], BrassState.TH0), params["p"], params["r"])

    # Calculate dynamical exponent
    (z, r2) = fit_dynamic_exponent!(ca, params["n_steps"], params["n_samples"])

    # Save data
    CSV.write(output_file_path, DataFrame(p=params["p"], r=params["r"], z=z, r2=r2), append=true)
end
