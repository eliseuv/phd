module Covid19Data

export Covid19TimeSeriesCategory, confirmed, recovered, deaths,
    fetch_covid19_time_series,
    load_covid19_time_series,
    join_covid19_time_series,
    covid19_time_series

using Logging, Dates, CSV, DataFrames, DataFramesMeta

# Information on the remote repository
const COVID19_DATA_REPO = "CSSEGISandData/COVID-19"
const COVID19_TIME_SERIES_PATH = "csse_covid_19_data/csse_covid_19_time_series"

# Categories of time series available
@enum Covid19TimeSeriesCategory begin
    confirmed
    recovered
    deaths
end

# Filenames in the remote repo for each time series
covid19_time_series_remote_filenames(cat::Covid19TimeSeriesCategory) = "time_series_covid19_" * string(cat) * "_global.csv"

# URL for raw time series data
covid19_time_series_url(cat::Covid19TimeSeriesCategory) = "https://raw.githubusercontent.com/" * COVID19_DATA_REPO * "/master/" * COVID19_TIME_SERIES_PATH * "/" * covid19_time_series_remote_filenames(cat)

# Canonical filename for time series data
covid19_time_series_path(cat::Covid19TimeSeriesCategory) = joinpath("..", "data", "JHUCovid19Data_" * string(cat) * ".csv")

function covid19_time_series_rename_cols(df::DataFrame)
    # Rename some long column names to avoid using `/`
    rename!(df, "Country/Region" => :Region, "Province/State" => :Province)
    # Rename date columns to make it easier to parse later
    rename!(df, [names(df)[begin:4]...,
        string.(Date.(replace.(names(df)[5:end], r"/(\d{2})$" => s"/20\1"), "mm/dd/yyyy"))...])
end

# Get Covid 19 time series data for a given category
function fetch_covid19_time_series(cat::Covid19TimeSeriesCategory)

    filepath = covid19_time_series_path(cat)
    @info filepath

    # URL with raw data
    url = covid19_time_series_url(cat)
    @info url

    # Check if local file already exists
    if !isfile(filepath)
        # Download new file
        download(url, filepath)
        @info "File $filepath created."
    else
        # Download temporary file
        tmp_file = download(url)
        @info "Downloading to temporary file $tmp_file"
        # Check if data has changed
        if success(`cmp --quiet $tmp_file $filepath`)
            @info "Data in $filepath is already up to date."
            rm(tmp_file)
        else
            mv(tmp_file, filepath, force = true)
            @info "File $filepath updated."
        end
    end

    # Read data to data frame
    return load_covid19_time_series(cat)
end

# Load Covid 19 time series data for a given category
function load_covid19_time_series(cat::Covid19TimeSeriesCategory)

    filepath = covid19_time_series_path(cat)

    # Check if file exists
    if isfile(filepath)
        df = CSV.File(filepath) |> DataFrame
        covid19_time_series_rename_cols(df)
        @info "File $filepath loaded."
        return df
    else
        @error "File $filepath does not exist."
        return nothing
    end
end

# Join all time series data into a single dataframe
function join_covid19_time_series(df_dict::Dict{Covid19TimeSeriesCategory,DataFrame})
    df_final = DataFrame()
    for (cat, df) in df_dict
        # Remove Province/State and Lat, Long information
        df_cat = select(df, Not([:Province, :Lat, :Long]))
        df_cat = combine(groupby(df_cat, :Region), names(df_cat, Not(:Region)) .=> sum, renamecols = false)
        # Insert category column
        insertcols!(df_cat, 2, :Category => cat)
        df_final = vcat(df_final, df_cat)
    end
    sort!(df_final, :Region)
    return df_final
end

# Get Covid 19 time series data for all categories
function fetch_covid19_time_series()
    df_dict = Dict{Covid19TimeSeriesCategory,DataFrame}()
    for cat in instances(Covid19TimeSeriesCategory)
        df_dict[cat] = fetch_covid19_time_series(cat)
    end
    return join_covid19_time_series(df_dict)
end

# Load Covid 19 time series data for all categories
function load_covid19_time_series()
    df_dict = Dict{Covid19TimeSeriesCategory,DataFrame}()
    for cat in instances(Covid19TimeSeriesCategory)
        df = load_covid19_time_series(cat)
        if !isnothing(df)
            df_dict[cat] = df
        end
    end
    return join_covid19_time_series(df_dict)
end

function covid19_time_series(gdf::GroupedDataFrame{DataFrame}, region::String, cat::Covid19TimeSeriesCategory)
    # Select data
    df = gdf[(Region = region, Category = cat)]
    # Rotate dataframe
    df = permutedims(select(transform(df, :Category => ByRow(x -> string(x)), renamecols = false), Not(:Region)), :Category, :Date)
    # Convert data
    df = transform!(df, :Date => ByRow(x -> Date(x)), renamecols = false)
end

end
