@doc raw"""
    DataIO

Utilities for reading and writing to data files.
"""
module DataIO

export number_regex, int_regex, float_regex,
    get_extension, keep_extension,
    parse_filename

using Logging, JLD2

include("Metaprogramming.jl")
using .Metaprogramming

# File system

"""
    get_extension(path::AbstractString)

Get the file extension from a given `filename`.
"""
function get_extension(path::AbstractString)
    ext = splitext(path)[2]
    if ext != ""
        return ext
    else
        @warn "File " * "\"$path\"" * " without extension"
        return nothing
    end
end

"""
    keep_file_extension(ext::AbstractString, paths::AbstractVector{AbstractString})

Keep only the files from `paths` with a given extension `ext`.
"""
keep_extension(ext::AbstractString, paths::AbstractVector{AbstractString}) = filter(path -> (get_extension(path) == ext), paths)

"""
    parse_filename(path::AbstractString; sep::AbstractString="_")

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
function parse_filename(path::AbstractString; sep::AbstractString = "_")
    filename = splitext(basename(path))[1]
    namechunks = split(filename, sep)
    param_dict = Dict{Symbol,Any}()
    param_dict[:prefix] = popfirst!(namechunks)
    while length(namechunks) != 0
        param = popfirst!(namechunks)
        while !occursin("=", param) && length(namechunks) != 0
            param = param * sep * popfirst!(namechunks)
        end
        (param_name, param_value) = split(param, "=")
        # Try to infer type
        ParamType = infer_type(param_value)
        if ParamType != Any
            # Type could be inferred, parse it
            param_dict[Symbol(param_name)] = parse(ParamType, param_value)
        else
            # Type could not be inferred, keep it as String
            param_dict[Symbol(param_name)] = param_value
        end
    end
    return param_dict
end

end
