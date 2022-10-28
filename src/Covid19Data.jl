module Covid19Data

export
    Covid19TimeSeriesCategory, confirmed, recovered, deaths,
    fetch_jhu_c19_time_series,
    load_jhu_c19_time_series,
    get_c19_region_time_series

using DrWatson
@quickactivate "phd"

using Logging, Dates, CSV, DataFrames

# Information on the remote repository
const JHU_C19_DATA_REPO = "CSSEGISandData/COVID-19"
const JHU_C19_TIME_SERIES_PATH = "csse_covid_19_data/csse_covid_19_time_series"

# Categories of time series available
@enum Covid19TimeSeriesCategory begin
    confirmed
    recovered
    deaths
end

# Filenames in the remote repo for each time series
@inline jhu_c19_time_series_remote_filenames(cat::Covid19TimeSeriesCategory) =
    "time_series_covid19_" * string(cat) * "_global.csv"

# URL for raw time series data
@inline jhu_c19_time_series_url(cat::Covid19TimeSeriesCategory) =
    joinpath("https://raw.githubusercontent.com/", JHU_C19_DATA_REPO, "master", JHU_C19_TIME_SERIES_PATH, jhu_c19_time_series_remote_filenames(cat))

# Canonical filepath for raw time series data
@inline jhu_c19_raw_time_series_path(cat::Covid19TimeSeriesCategory) =
    datadir("exp_raw", "jhu_covid19", "JHU_Covid19_raw_" * string(cat) * ".csv")

"""
    fetch_jhu_c19_raw_time_series([cat::Covid19TimeSeriesCategory])

Fetch a specific JHU Covid-19 raw data from the remote repository.

If no category is specified, fetch them all.
"""
function fetch_jhu_c19_time_series(cat::Covid19TimeSeriesCategory)

    # Local raw data file
    filepath = jhu_c19_raw_time_series_path(cat)
    @debug filepath

    # URL with raw data
    url = jhu_c19_time_series_url(cat)
    @debug url

    # Check if local file already exists
    if !isfile(filepath)
        # Download new file
        download(url, filepath)
        @debug "File $filepath created."
    else
        # Download temporary file
        tmp_file = download(url)
        @debug "Downloading to temporary file $tmp_file"
        # Check if data has changed
        if success(`cmp --quiet $tmp_file $filepath`)
            @debug "Data in $filepath is already up to date."
            rm(tmp_file)
        else
            mv(tmp_file, filepath, force=true)
            @debug "File $filepath updated."
        end
    end
end

@inline fetch_jhu_c19_time_series() = foreach(fetch_jhu_c19_time_series, instances(Covid19TimeSeriesCategory))

# Rename columns to a better standard
@inline jhu_c19_time_series_rename_cols!(df_raw::DataFrame) =
    rename!(df_raw,
        # Shorten some cols names
        "Country/Region" => :Region,
        "Province/State" => :Province,
        # Rename dates cols to ISO standard
        map(names(df_raw)[5:end]) do from_name
            from_name => Date(replace(from_name, r"/(\d{2})$" => s"/20\1"), "mm/dd/yyyy") |> Symbol
        end...)

# Join all time series data into a single dataframe
@inline jhu_c19_join_time_series(df_dict::Dict{Covid19TimeSeriesCategory,DataFrame}) =
    foldl(df_dict, init=DataFrame()) do df_join, (cat, df_cat)
        select!(df_cat, Not([:Province, :Lat, :Long]))
        df_cat = combine(groupby(df_cat, :Region), names(df_cat, Not(:Region)) .=> sum, renamecols=false)
        insertcols!(df_cat, 2, :Category => cat)
        return vcat(df_join, df_cat)
    end

"""
    load_jhu_c19_time_series([cat::Covid19TimeSeriesCategory])

Load a specific JHU Covid-19 raw data from local file.

If no category is specified, load them all.
"""
function load_jhu_c19_time_series(cat::Covid19TimeSeriesCategory)

    # Local data file path
    filepath = jhu_c19_raw_time_series_path(cat)

    # Check if file exists
    if isfile(filepath)
        df = CSV.File(filepath) |> DataFrame
        jhu_c19_time_series_rename_cols!(df)
        @debug "File $filepath loaded."
        return df
    else
        @error "File $filepath does not exist."
        return nothing
    end
end

function load_jhu_c19_time_series()
    df_dict = Dict{Covid19TimeSeriesCategory,DataFrame}()
    for cat in instances(Covid19TimeSeriesCategory)
        @debug "Loading $cat time series"
        df = load_jhu_c19_time_series(cat)
        if !isnothing(df)
            df_dict[cat] = df
        end
    end
    @debug "Joining data frames"
    return jhu_c19_join_time_series(df_dict)
end


"""
    get_c19_region_time_series(df_raw::DataFrame, region::String)

Get JHU Covid-19 time series data for a given `region`.
"""
@inline get_c19_region_time_series(df_raw::DataFrame, region::String) =
    transform(
        permutedims(
            transform(
                df_raw[df_raw.Region.==region, Not(:Region)],
                :Category => ByRow(string), renamecols=false),
            :Category, :Date),
        :Date => ByRow(Date), renamecols=false)

end
