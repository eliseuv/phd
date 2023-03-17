using DrWatson
@quickactivate "phd"

using CSV, DataFrames

# Load critical temperatures dataframes
df_temperatures = DataFrame(CSV.File(projectdir("tables", "butera_and_pernici_2018", "blume-capel_square_lattice.csv")))

const τ_vals = 2.0 .^ (-0.2:0.05:0.2)

for (D, L) ∈ Iterators.product([0, 0.5, 1], 100)

    # Find critical temperature in table
    df_crit_row = df_temperatures[only(findall(==(D), df_temperatures.anisotropy_field)), 2:end]
    crit_temp_source = findfirst(!ismissing, df_crit_row)
    T_c = df_crit_row[crit_temp_source]

    for T ∈ τ_vals .* T_c

        beta = 1.0 / T

        println("$L $D $beta")

    end

end
