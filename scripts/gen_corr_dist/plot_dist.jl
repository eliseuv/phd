# Dr Watson helper
using DrWatson
@quickactivate "phd"

# External libraries
using Logging, FileIO, Gadfly, Cairo

# Custom modules
include("../../src/Thesis.jl")
using .Thesis.DataIO
using .Thesis.TimeSeries

const params_req = Dict("dist" => "nrmsd", "gamma" => 1, "sigma" => 1)

# Get datafiles
datafiles =
    find_datafiles(
        datadir(),
        "GenUniformCorrDistSAFinalState",
        params_req,
        ext="jld2")

@inline params_str(params::Dict{String,T}) where {T} =
    join([string(name) * " = " * string(value) for (name, value) ∈ params], ", ")

@show datafiles

for datafile in datafiles
    run = datafile.params["run"]
    @show run
    M_ts = load(datafile.path, "M_ts")

    corr_vals = cross_correlation_values(M_ts)
    script_show(corr_vals)

    normalize_ts_matrix!(M_ts)

    G = cross_correlation_matrix(M_ts)
    script_show(G)

    corr_vals = cross_correlation_values_norm(M_ts)
    script_show(corr_vals)

    output_plotname = plotsdir(DataIO.filename("GenUniformCorrDistSAFinalState", params_req, "run" => run, ext="png"))
    @show output_plotname

    plt = plot(x=corr_vals,
        Geom.histogram,
        Guide.title("Generate Uniform Correlation Distribution using Simulated Annealing Final Distribution"),
        Guide.xlabel("ρ"))
    draw(PNG(plotsdir(output_plotname), 30cm, 18cm), plt)

end
