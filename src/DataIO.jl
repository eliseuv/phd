@doc raw"""
    DataIO

Utilities for reading and writing to data files.
"""
module DataIO

export file_extension,
    keep_filenames_with_extension,
    parse_vars_in_filename

using Logging, JLD2

include("Metaprogramming.jl")
using .Metaprogramming

# Reading data

function file_extension(filepath::String)
    filename = basename(filepath)
    re_ext = r"\.(?<ext>.+?)$"
    m = match(re_ext, filename)
    if isnothing(m)
        @warn "File " * dq"$filename" * " without extension"
        return nothing
    else
        return m[:ext]
    end
end

"""
Keep only the filenames with a given extension

# Arguments:
    - `ext`: Desired extension
    - `filenames`: Vector of filenames

# Returns:
    - Vectors of filenames with desired extension
"""
function keep_filenames_with_extension(ext::String, filenames::Vector{String})::Vector{String}
    resulting_filenames = Vector{String}()
    for filename in filenames
        file_extension_m = match(r"\.(\D+)$", basename(filename))
        if isnothing(file_extension_m)
            println("Ignoring file without extension: $(filename)")
            continue
        end
        file_extension = file_extension_m.captures[1]
        if file_extension == ext
            push!(resulting_filenames, filename)
        end
    end
    resulting_filenames
end

"""
Parse for variables in filenames

The expected format of the filenames is "GeneralDescription:var1=value1:var2=value2:...:varN=valueN.ext".

# Arguments:
    - filename

# Returns:
    - Dictionary whose keys are variables names and values are parsed variable values.
"""
function parse_vars_in_filename(filename::String)::Dict{String,Any}
    vars_dict = Dict()
    # Parse floats
    re = Regex(p":(\w+?)=(" * float_regex * p")")
    if !isnothing(match(re, filename))
        vars_list = [m.captures for m in eachmatch(re, filename)]
        for var in vars_list
            var_name = var[1]
            var_value = var[2]
            if !haskey(vars_dict, var_name)
                vars_dict[var_name] = parse(Float64, var_value)
            end
        end
    end
    # Parse integers
    re = Regex(p":(\w+?)=(" * int_regex * p")")
    if !isnothing(match(re, filename))
        vars_list = [m.captures for m in eachmatch(re, filename)]
        for var in vars_list
            var_name = var[1]
            var_value = var[2]
            if !haskey(vars_dict, var_name)
                vars_dict[var_name] = parse(Int, var_value)
            end
        end
    end
    # Parse everything else
    re = r":(\w+?)=(\w+?)"
    if !isnothing(match(re, filename))
        vars_list = [m.captures for m in eachmatch(re, filename)]
        for var in vars_list
            var_name = var[1]
            var_value = var[2]
            if !haskey(vars_dict, var_name)
                vars_dict[var_name] = var_value
            end
        end
    end
    return vars_dict
end

end
