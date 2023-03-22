using DrWatson
@quickactivate "phd"

using CSV, DataFrames

# Load critical temperatures dataframes
df_temperatures = DataFrame(CSV.File(projectdir("tables", "butera_and_pernici_2018", "blume-capel_square_lattice.csv")))

const τ_vals = [0.5, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 2.0]

for (D, L) ∈ Iterators.product([0, 0.5, 1], 100)

    # Find critical temperature in table
    df_crit_row = df_temperatures[only(findall(==(D), df_temperatures.anisotropy_field)), 2:end]
    crit_temp_source = findfirst(!ismissing, df_crit_row)
    T_c = df_crit_row[crit_temp_source]

    for T ∈ τ_vals .* T_c

        println("$L $D $T")

    end

end
