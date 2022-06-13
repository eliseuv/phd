module Covid19Data

export Covid19TimeSeriesCategory, confirmed, recovered, deaths,
    get_covid19_time_series,
    load_covid19_time_series

using Logging, Dates, CSV, DataFrames, DataFramesMeta

# Information on the remote repository
const COVID19_DATA_REPO = "CSSEGISandData/COVID-19"
const COVID19_TIME_SERIES_PATH = "csse_covid_19_data/csse_covid_19_time_series"

# Categories of time series available
@enum Covid19TimeSeriesCategory::Int8 begin
    confirmed
    recovered
    deaths
end

# Filenames for each time series
covid19_time_series_remote_filenames(cat::Covid19TimeSeriesCategory) = "time_series_covid19_" * string(cat) * "_global.csv"

# URL for raw time series data
covid19_time_series_url(cat::Covid19TimeSeriesCategory) = "https://raw.githubusercontent.com/" * COVID19_DATA_REPO * "/master/" * COVID19_TIME_SERIES_PATH * "/" * covid19_time_series_remote_filenames(cat)

# Canonical filename for time series data
covid19_time_series_path(cat::Covid19TimeSeriesCategory) = "data/JHUCovid19Data_" * string(cat) * ".csv"

# Get Covid 19 time series data for a given category
function get_covid19_time_series(cat::Covid19TimeSeriesCategory; output = covid19_time_series_path(cat))

    @info output

    # URL with raw data
    url = covid19_time_series_url(cat)
    @info url

    # Check if local file already exists
    if !isfile(output)
        # Download new file
        download(url, output)
        @info "File $output created."
    else
        # Download temporary file
        tmp_file = "/tmp/JHUCovid19Data.csv.temp"
        download(url, tmp_file)
        # Check if data has changed
        if success(`cmp --quiet $tmp_file $output`)
            @info "Data in $output is already up to date."
        else
            cp(tmp_file, output)
            @info "File $output updated."
        end
    end

    # Read data to data frame
    df = CSV.File(output) |> DataFrame
    return df
end

# Get Covid 19 time series data for all categories
function get_covid19_time_series()
    df_dict = Dict{Covid19TimeSeriesCategory,DataFrame}()
    for cat in instances(Covid19TimeSeriesCategory)
        df_dict[cat] = get_covid19_time_series(cat)
    end
    return df_dict
end

# Load Covid 19 time series data for a given category
function load_covid19_time_series(cat::Covid19TimeSeriesCategory)

    filepath = "data/JHUCovid19Data_" * string(cat) * ".csv"

    # Check if file exists
    if isfile(filepath)
        df = CSV.File(filepath) |> DataFrame
        @info "File $filepath loaded."
        return df
    else
        @error "File $filepath does not exist."
        return nothing
    end
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
    return df_dict
end

function parse_covid19_time_series!(df::DataFrame)
    # Remove Province/State and Lat, Long information
    select!(df, Not(["Province/State", "Lat", "Long"]))
    df = combine(groupby(df, "Country/Region"), names(df, Not("Country/Region")) .=> sum, renamecols = false)
    # Transpose dataframe
    permutedims!(df, "Country/Region", "Date")
    # Convert dates to Date type
    df[!, :Date] = Date.(replace.(df[!, :Date], r"/(\d\d)$" => s"/20\1"), "mm/dd/yyyy")
end

end
