module DataIO

export @defvar,
    @strvar, @strvars,
    @p_str,
    number_regex, int_regex, float_regex,
    keep_filenames_with_extension,
    parse_vars_in_filename

@doc raw"""
    @defvar(name_str, value)

Define a variable with given name and variable

# Example:
    ```julia
    var_name = "new_var"
    var_value = 5
    @defvar(var_name, var_value)
    # The variable `new_var` is decalerd with value `5`
    println(new_var)
    # Prints `5`
    ```

Since macros are expanded at parse time, in order to declare variable in a loop:

    ```julia
    for (name,value) in zip(names,values)
        @eval(@defvar($name, $value))
    end
    ```

# Arguments:
    - `name_str::String`: String with variable name
    - `value: Value to be assigned to variable
"""
macro defvar(name_str, value)
    name_symb = Symbol(eval(name_str))
    return :($(esc(name_symb)) = $value)
end

@doc raw"""
    @strvar(var)

String representation of variable and its value

# Example:
    ```julia
    my_name = "new_var"
    var_value = 5
    @defvar(var_name, var_value)
    # The variable `new_var` is decalerd with value `5`
    println(new_var)
    # Prints `5`
    ```

# Arguments:
    - `var`: Variable (name `variable_name` and value `variable_value`)

# Returns:
    - String: `"variable_name=variable_value"`
"""
macro strvar(var)
    quote
        $(string(var)) * "=" * string($(esc(var)))
    end
end

@doc raw"""
    @strvars(vars...)

String representation of multiple variables

# Example:
    ```julia

    ```

# Arguments:
    - `vars...`: Variables (names `variable1`, `variable2` etc. and values `value1`, `value2` etc.)

# Retuns:
    - String: `"variable1=value1:variable2=value2:...:variableN=valueN"`
"""
macro strvars(vars...)
    result = :($(string(vars[1])) * "=" * string($(esc(vars[1]))))
    for k in 2:length(vars)
        result = :($result * ":" * $(string(vars[k])) * "=" * string($(esc(vars[k]))))
    end
    return result
end

"""
Quote macros
(Suggested by [[https://stackoverflow.com/a/20483464]])
"""

"""
p"quote" macro

Since regular strings do not allow for escape chars and the r"quote" macro
compiles the regular expression when defined, this intermediate quote allows
us to store parts of regular expressions that can later be combined and compiled.

# Usage:

    pq1 = p"\\D+"
    pq2 = p"\\d+"
    re = Regex(pq1 * p"=" * pq2)

The regex `re` is equivalent to r"\\D+=\\d+"

"""
macro p_str(s)
    s
end

"""
Useful p"quotes" for building regular expressions
"""
# Match any number (integer or floating point)
const number_regex = p"[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?"
# Match integers
const int_regex = p"[-+]?[0-9]*([eE][+]?[0-9]+)?"
# Match floating point numbers
const float_regex = p"[-+]?[0-9]*\.[0-9]+([eE][-+]?[0-9]+)?"

"""
Double quotes quote string

# Usage:
    dq"with double quotes" -> "with double quotes"
"""
macro dq_str(s)
    "\"" * s * "\""
end

"""
Simple quotes quote string

# Usage:
    sq"with simple quotes" -> 'with simple quotes'
"""
macro sq_str(s)
    "'" * s * "'"
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
