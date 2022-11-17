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
    # Filanames
    filename,
    parse_filename,
    # Check parameters
    check_params,
    find_datafiles_with_params,
    # File locks
    get_lock, remove_lock

using Logging, SHA, JLD2

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

"""
    get_extension(path::AbstractString)

Get the file extension from a given `filename`.
"""
function get_extension(path::AbstractString)
    ext = splitext(path)[end]
    if ext != ""
        return ext
    else
        @info "File " * "\"$path\"" * " without extension"
        return nothing
    end
end

"""
    keep_file_extension(ext::AbstractString, paths::AbstractVector{AbstractString})

Keep only the files from `paths` with a given extension `ext`.
"""
@inline keep_extension(ext::AbstractString, paths::AbstractVector{AbstractString}) = filter(path -> (get_extension(path) == ext), paths)

@doc raw"""
    filename(prefix::AbstractString, params::Dict{String,Any}; sep::AbstractString = "_", ext::Union{AbstractString,Nothing} = "jld2")

Generate a filname give an `prefix` a dictionary of parameters `params` and a file extension `ext`.

Each parameter is written as `param_name=param_value` and separated by a `sep` string.

The dot `.` in the extension can be ommited: `ext=".csv"` and `ext="csv"` are equivalent.

The default file extension is `.jld2`.
To create a file without extension, use either `ext=nothing` or `ext=""`.
"""
function filename(prefix::AbstractString, params::Dict{String}; sep::AbstractString="_", ext::Union{AbstractString,Nothing}="jld2")
    # Prefix
    filename = prefix
    # Parameters in alphabetical order
    for (param_name, param_value) in sort(collect(params), by=x -> x.first)
        filename = filename * sep * string(param_name) * '=' * string(param_value)
    end
    # Extension
    if !isnothing(ext) && ext != ""
        if ext[begin] == '.'
            filename = filename * ext
        else
            filename = filename * '.' * ext
        end
    end
    return filename
end

@doc raw"""
    parse_filename(path::AbstractString; sep::AbstractString = "_")

Attempts to parse paramenters in name of file given by `path` using `sep` as parameter separator.

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

@doc raw"""
    check_params(params::Dict{String}, req::Pair{String})

Checks if the parameters dictionary `params` has the key-value pair specified by the pair `req`.
"""
@inline check_params(params::Dict{String}, req::Pair{String}) =
    let (key, value) = req
        haskey(params, key) && params[key] == value
    end

@doc raw"""
    check_params(params::Dict{String}, reqs::Dict{String})

Checks if the parameters dictionary `params` satisfies the values defined in the parameters requirements dictionary `reqs`.
"""
@inline check_params(params::Dict{String}, reqs::Dict{String}) =
    all(check_params(params, key => value) for (key, value) ∈ reqs)

@doc raw"""
    check_params(params::Dict{String}, reqs...)

Checks if the parameters dictionary `params` satisfies the values defined in the parameters dictionaries and pairs `reqs...`.
"""
@inline check_params(params::Dict{String}, reqs...) = all(check_params(params, req) for req ∈ reqs)

"""
    find_datafiles_with_params(datadir::String, reqs...)

Find data files in the directory `datadir` that have the satisfies the required parameters `reqs...`.
"""
@inline find_datafiles_with_params(path::AbstractString, prefix_req::String, reqs...) =
    filter(readdir(path)) do datafile_name
        (filename_prefix, filename_params) = parse_filename(datafile_name)
        return filename_prefix == prefix_req && check_params(filename_params, reqs...)
    end

@inline filepath_lock(path::AbstractString) =
    let path_hash = path |> sha256 |> bytes2hex
        "/tmp/$(path_hash).lock"
    end

@inline function get_lock(path::AbstractString)
    lock_file = filepath_lock(path)
    # Wait if file is already locked
    while isfile(lock_file)
        sleep(0.001)
    end
    # Once file unlock create lock
    touch(lock_file)
end

@inline remove_lock(path::AbstractString) =
    let lock_file = filepath_lock(path)
        rm(lock_file, force=true)
    end

end
