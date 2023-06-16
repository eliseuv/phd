# Dr Watson helper
using DrWatson
@quickactivate "phd"

# External libraries
using Logging, Statistics, PyCall, DataFrames, CSV

# My libs
include("../../../../src/Thesis.jl")
using .Thesis.Metaprogramming
using .Thesis.DataIO
using .Thesis.Stats

py"""
import pickle

def load_pickle(fpath):
    with open(fpath, "rb") as f:
        data = pickle.load(f)
    return data
"""

@inline load_pickle = py"load_pickle"

@inline function get_critical_temperature(df_temperatures, D::Real)
    df_D_row = df_temperatures[only(findall(==(D), df_temperatures.anisotropy_field)), 2:end]
    transition_order = lowercase(string(df_D_row[:transition_order]))
    crit_temp_source = findfirst(!ismissing, df_D_row)
    T_c = df_D_row[crit_temp_source]
    return (T_c, transition_order, crit_temp_source)
end

@inline get_spacings(eigvals_matrix::AbstractMatrix{<:Real}) = diff(eigvals_matrix, dims=2)
@inline function get_normalized_spacings(eigvals_matrix::AbstractMatrix{<:Real})
    eigvals_spacings = get_spacings(eigvals_matrix)
    eigvals_spacings_means = mean(eigvals_spacings, dims=2)
    return eigvals_spacings_means .\ eigvals_spacings
end

# Where to take the data from
data_dirpath = datadir("blume_capel_pickles", "eigvals")
@show data_dirpath

const global_prefix = "BlumeCapelSq2D"
const output_prefix = global_prefix * "EigvalsSpacingsHist"
const output_root = "./output/"

# Parse datafiles into dicts
@info "Parsing datafiles..."
eigvals_datafiles = Dict()
correlations_datafiles = Dict()
for (root, _, filenames) in walkdir(data_dirpath)
    for filename in filenames
        path = joinpath(root, filename)
        datafile = DataIO.DataFile(path)
        D = Float64(datafile.params["D"])
        T = Float64(datafile.params["T"])
        if datafile.prefix == global_prefix * "Eigvals"
            if haskey(eigvals_datafiles, D)
                eigvals_datafiles[D][T] = datafile
            else
                eigvals_datafiles[D] = Dict(T => datafile)
            end
            # elseif datafile.prefix == global_prefix * "Correlations"
            #     if haskey(correlations_datafiles, D)
            #         correlations_datafiles[D][T] = datafile
            #     else
            #         correlations_datafiles[D] = Dict(T => datafile)
            #     end
        end
    end
end

D_values = sort(collect(keys(eigvals_datafiles)))

# Load temperatures table
@info "Loading temperatures table..."
df_temperatures = DataFrame(CSV.File(projectdir("tables", "butera_and_pernici_2018", "blume-capel_square_lattice.csv")))

# Loop on anisotropy values
for (D, D_dict) ∈ sort(collect(eigvals_datafiles), by=x -> x[1])

    # if D ∉ [D_vals_second..., D_vals_first..., D_vals_tcp[2]]
    #     continue
    # end

    # Fetch critical temperature info
    T_c, transition_order, crit_temp_source = get_critical_temperature(df_temperatures, D)
    transition_order_str = replace(transition_order, "first" => "First order", "second" => "Second order", "tcp" => "TCP")
    crit_temp_source_str = replace(string(crit_temp_source), "_" => " ")
    T_vec = sort(collect(keys(D_dict)))
    tau_vec = map(T_vec ./ T_c) do x
        round(x; digits=3)
    end

    println("D = $D ($(transition_order_str))")

    # Create dir
    output_dir_D = joinpath(output_root, "D=$D($(transition_order))")
    mkpath(output_dir_D)

    for (T, tau) in zip(T_vec, tau_vec)
        datafile = D_dict[T]
        eigvals_matrix = load_pickle(datafile.path)
        @show tau T

        eigvals_normalized_spacings = vec(get_normalized_spacings(eigvals_matrix))

        hist = Histogram(eigvals_normalized_spacings, 100)
        x, y = hist_coords(hist)
        df = DataFrame(bin=x, counts=y)

        script_show(df)

        output_path = joinpath(output_dir_D, filename(output_prefix, @varsdict(D, tau); ext="csv"))
        @show output_path

        CSV.write(output_path, df)

    end

end
