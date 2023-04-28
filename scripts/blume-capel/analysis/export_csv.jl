# Dr Watson helper
using DrWatson
@quickactivate "phd"

# External libraries
using Logging, Pickle, DelimitedFiles, DataFrames, CSV

# My libs
include("../../../src/Thesis.jl")
using .Thesis.Metaprogramming
using .Thesis.DataIO

const prefix = "BlumeCapelSq2DEigvals"
const input_dir = datadir("blume_capel_pickles", "eigvals")
const output_dir = datadir("csv_exports")
mkpath(output_dir)

@show input_dir output_dir

# Load temperatures table
@info "Loading temperatures table..."
df_temperatures = DataFrame(CSV.File(projectdir("tables", "butera_and_pernici_2018", "blume-capel_square_lattice.csv")))

const datafiles = find_datafiles(input_dir, prefix, "D" => 0; ext="pickle")
for datafile in datafiles
    @show datafile.params

    D = datafile.params["D"]
    T = datafile.params["T"]

    # Fetch critical temperature info
    df_D_row = df_temperatures[only(findall(==(D), df_temperatures.anisotropy_field)), 2:end]
    crit_temp_source = findfirst(!ismissing, df_D_row)
    T_c = df_D_row[crit_temp_source]
    tau = round(T / T_c; digits=3)

    data = sort(vcat(Pickle.load(datafile.path)...))
    #data = Pickle.load(datafile.path)

    output_filename = filename(prefix, @varsdict(D, tau); ext="csv")
    @show output_filename
    output_path = joinpath(output_dir, output_filename)

    open(output_path, "w") do io
        writedlm(io, data, ',')
    end


end
