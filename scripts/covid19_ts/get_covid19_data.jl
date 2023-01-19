using DrWatson
@quickactivate "phd"

using DataFrames, Dates

include("../../src/Thesis.jl")
using .Thesis.Covid19Data

# Fetch data
fetch_jhu_c19_time_series()

# Load local files
df_raw = load_jhu_c19_time_series()

# Get time series for a specific country
df_brazil = get_c19_region_time_series(df_raw, "Brazil")
