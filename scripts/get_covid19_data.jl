using DrWatson
@quickactivate "phd"

using DataFrames, Dates

include("../src/Covid19Data.jl")
using .Covid19Data

df = fetch_covid19_time_series()
gdf = groupby(df, [:Region, :Category])

ts = covid19_time_series(gdf, "Brazil", confirmed)
