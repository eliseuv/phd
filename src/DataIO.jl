@doc raw"""
    DataIO

Utilities for reading and writing to data files.
"""
module DataIO

export
    # Print in script the same way as in the REPL
    script_show,
    # File extensions
    get_extension,
    keep_extension,
    # Filenames
    filename,
    parse_filename,
    # Check parameters
    check_params,
    find_datafiles

using Logging

using ..Metaprogramming

# Script preview

"""
    script_show(x...)

Print the entities `x...` to standard output while in a script in the same way it prints in a REPL session.
"""
function script_show(x...)
    show(IOContext(stdout, :limit => true), "text/plain", x...)
    println()
end

# Filenames

@inline params_str(params::Union{Dict{String},Pair{String}}...; sep::AbstractString="_") =
    params_str(merge(map(Dict, params)...), sep=sep)

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
    filename = prefix * sep * params_str(params..., sep=sep)
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

@doc raw"""
    parse_filename(path::AbstractString; sep::AbstractString = "_")

Attempts to parse parameters in name of file given by `path` using `sep` as parameter separator.

It assumes the following pattern for the filename (using the default separator `"_"`):
    `SomePrefix_first_param=foo_second_param=42_third_param=3.14.ext`

Retuns a `Dict{Symbol,Any}` with keys being the names of the parameters as symbols and the values the parsed parameter values.

# Example:
    ```julia
    julia> test_path = "/path/to/SomePrefix_first_param=foo_second_param=42_third_param=3.14.ext"
    julia>  for (key, value) in parse_filename(test_filename)
                println("$key => $value ($(typeof(value)))")
            end
    prefix => SomePrefix (SubString{String})
    third_param => 3.14 (Float64)
    second_param => 42 (Int64)
    first_param => foo (SubString{String})
    ```
"""
function parse_filename(path::AbstractString; sep::AbstractString="_")
    # Discard directory path and extension
    filename = splitext(basename(path))[begin]
    # Split name into chunks
    namechunks = split(filename, sep)
    # The first chunk is always the prefix
    prefix = popfirst!(namechunks)
    # Dictionary to store parsed parameter values
    param_dict = Dict{String,Any}()
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
            param_dict[string(param_name)] = parse(ParamType, param_value)
        else
            # Type could not be inferred, keep it as String
            param_dict[string(param_name)] = param_value
        end
    end
    return (prefix, param_dict)
end

"""
    get_extension(path::AbstractString)

Get the file extension from a given `filename`.
"""
function get_extension(path::AbstractString)
    ext = splitext(path)[end]
    if ext != ""
        return ext
    else
        @debug "File " * "\"$path\"" * " without extension"
        return nothing
    end
end

@doc raw"""
    DataFile
"""
struct DataFile

    path::AbstractString
    ext::AbstractString
    prefix::AbstractString
    params::Dict{String,T} where {T}

    DataFile(path::AbstractString; sep::AbstractString="_") = new(path, get_extension(path), parse_filename(path, sep=sep)...)
end

"""
    keep_extension(ext::AbstractString, paths::AbstractVector{<:AbstractString})

Keep only the files from `paths` with a given extension `ext`.
"""
@inline function keep_extension(ext::AbstractString, paths::AbstractVector{<:AbstractString})
    if !startswith(ext, '.')
        ext = '.' * ext
    end
    return filter(path -> (get_extension(path) == ext), paths)
end

@doc raw"""
    check_params(params::Dict{String}, req::Pair{String,T}) where {T}

Checks if the parameters dictionary `params` has the key-value pair specified by the pair `req`.
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

end
