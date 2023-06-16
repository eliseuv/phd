@doc raw"""
    DataIO

Utilities for reading and writing to data files.
"""
module DataIO

export
    # Print in script the same way as in the REPL
    script_show,
    # File extensions
    keep_extension,
    # Filenames
    filename,
    parse_filename,
    # DataFile type
    DataFile,
    # Check parameters
    check_params,
    find_datafiles,
    # Python pickle
    load_pickle

using Logging, PyCall

# Load Python pickle module
@pyimport pickle

using ..Metaprogramming

##################
# Script preview #
##################

@doc raw"""
    script_show(x...)

Print the entities `x...` to standard output while in a script in the same way it prints in a REPL session.
"""
function script_show(x...)
    show(IOContext(stdout, :limit => true), "text/plain", x...)
    println()
end

#######################
# Filename generation #
#######################

@doc raw"""
    params_str(params::Union{Dict{String},Pair{String}}...; sep::AbstractString="_")

Generate a string containing the 'name=value' of the parameters specified by the arguments `params`
(name-value pairs or dictionaries) sorted alphabetically and separated by the string `sep`.

# Example:
    ```julia
    julia> params_str(Dict("foo" => 0.5), "bar" => "baz")
    "bar=baz_foo=0.5"
    julia> params_str("foo" => 0.5, "bar" => "baz", sep=";")
    "bar=baz;foo=0.5"
    ```

"""
@inline params_str(params::Union{Dict{String},Pair{String}}...; sep::AbstractString="_") =
    params_str(foldl(merge, [map(Dict, params)...]; init=Dict{String,Any}()), sep=sep)

@doc raw"""
    params_str(params::Dict{String}; sep::AbstractString="_")

Generate a string containing the 'name=value' of the parameters specified by the dictionary `params`
sorted alphabetically and separated by the string `sep`.

# Example:
    ```julia
    julia> params = Dict("foo" => 0.5, "bar" => "baz")
    julia> params_str(params)
    "bar=baz_foo=0.5"
    julia> params_str(params, sep=";")
    "bar=baz;foo=0.5"
    ```
"""
@inline params_str(params::Dict{String}; sep::AbstractString="_") =
    join([name * "=" * string(value) for (name, value) ∈ sort(collect(params), by=x -> x[1])], sep)

@doc raw"""
    filename(prefix::AbstractString, params...; sep::AbstractString="_", ext::AbstractString="jld2")

Generate a filename give an `prefix`, dictionaries of parameters, or pairs `params...` and a file extension `ext`.

Each parameter is written as `param_name=param_value` and separated by a `sep` string.

The dot `.` in the extension can be omitted: `ext=".csv"` and `ext="csv"` are equivalent.

The default file extension is `.jld2`.
To create a file without extension, use either `ext=nothing` or `ext=""`.
"""
function filename(prefix::AbstractString, params...; sep::AbstractString="_", ext::AbstractString="jld2")
    # Prefix and parameters
    filename = prefix
    parameters = params_str(params..., sep=sep)
    if !isempty(parameters)
        filename *= sep * parameters
    end
    # Extension
    if !isnothing(ext) && ext != ""
        if ext[begin] == '.'
            filename *= ext
        else
            filename *= '.' * ext
        end
    end
    return filename
end

####################
# Filename parsing #
####################

@doc raw"""
    parse_filename(path::AbstractString; sep::AbstractString = "_")

Attempts to parse parameters in name of file given by `path` using `sep` as parameter separator.

It assumes the following pattern for the filename (using the default separator `"_"`):
    `SomePrefix_first_param=foo_second_param=42_third_param=3.14.ext`

Returns a tuple `(String, Dict{String,Any}, String)` containing:
    [1] Filename prefix
    [2] Dictionary with keys being the names of the parameters as symbols and the values the parsed parameter values
    [3] File extension

# Example:
    ```julia
    julia> parse_filename("/path/to/SomePrefix_first_param=foo_second_param=42_third_param=3.14.ext")
    ("SomePrefix", Dict("first_param" => "foo", "second_param" => 42, "third_param" => 3.14), ".ext")
    ```
"""
function parse_filename(path::AbstractString; sep::AbstractString="_")
    # Get filename and extension
    (filename, ext) = path |> basename |> splitext
    # Split name into chunks
    namechunks = split(filename, sep)
    # The first chunk is always the prefix
    prefix = popfirst!(namechunks)
    # Dictionary to store parsed parameter values
    params_dict = Dict{String,Any}()
    while length(namechunks) != 0
        param = popfirst!(namechunks)
        while !occursin("=", param) && length(namechunks) != 0
            param = param * sep * popfirst!(namechunks)
        end
        if occursin("=", param)
            (param_name, param_value) = split(param, "=")
        else
            break
        end
        # Try to infer type
        ParamType = infer_type_sized(param_value)
        if ParamType != Any
            # Type could be inferred, parse it
            params_dict[string(param_name)] = parse(ParamType, param_value)
        else
            # Type could not be inferred, keep it as String
            params_dict[string(param_name)] = param_value
        end
    end
    return (prefix, params_dict, ext)
end

#################
# DataFile type #
#################

@doc raw"""
    DataFile
"""
struct DataFile

    path::AbstractString
    prefix::AbstractString
    params::Dict{String,T} where {T}
    ext::AbstractString

    DataFile(path::AbstractString; sep::AbstractString="_") = new(path, parse_filename(path, sep=sep)...)
end

"""
    keep_extension(ext::AbstractString, paths::AbstractVector{<:AbstractString})

Keep only the files from `paths` with a given extension `ext`.
"""
@inline function keep_extension(ext::AbstractString, paths::AbstractVector{<:AbstractString})
    if !startswith(ext, '.')
        ext = '.' * ext
    end
    return filter(path -> (splitext(path)[2] == ext), paths)
end

@doc raw"""
    check_params(params::Dict{String}, req::Pair{String,T}) where {T}

Checks if the prameters dictionary `params` has the key-value pair specified by the pair `req`.
"""
@inline check_params(params::Dict{String}, req::Pair{String,T}) where {T} =
    let (key, value) = req
        haskey(params, key) && params[key] == value
    end

@doc raw"""
    check_params(params::Dict{String}, reqs::Dict{String})

Checks if the parameters dictionary `params` satisfies the values defined in the parameters requirements dictionary `reqs`.
"""
@inline check_params(params::Dict{String}, reqs::Dict{String,T}) where {T} =
    all(check_params(params, key => value) for (key, value) ∈ reqs)

@doc raw"""
    check_params(params::Dict{String}, reqs...)

Checks if the parameters dictionary `params` satisfies the values defined in the parameters dictionaries and pairs `reqs...`.
"""
@inline check_params(params::Dict{String}, reqs...) = all(check_params(params, req) for req ∈ reqs)

@doc raw"""
    check_params(datafile::DataFile, reqs...)
"""
@inline check_params(datafile::DataFile, reqs...) = check_params(datafile.params, reqs...)

"""
    find_datafiles(path::AbstractString, prefix::AbstractString, reqs...; ext::AbstractString=".jld2", sep::AbstractString="_")

Find data files in the directory `datadir` that have the satisfies the required parameters `reqs...`.
"""
@inline find_datafiles(path::AbstractString, prefix::AbstractString, reqs...; ext::AbstractString=".jld2", sep::AbstractString="_") =
    readdir(path, join=true) |>
    fs -> keep_extension(ext, fs) |>
          fs -> map(f -> DataFile(f, sep=sep), fs) |>
                dfs -> filter(df -> df.prefix == prefix && check_params(df.params, reqs...), dfs)

"""
    load_pickle(path::AbstractString)

Load Python `pickle` data file.
"""
@inline function load_pickle(path::AbstractString)
    # Check if file exists
    if !isfile(path)
        @error "File $(path) not found!"
    end
    # Load data
    f = open(path, "r")
    data = pickle.load(f)
    close(f)
    return data
end

end
