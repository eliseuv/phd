@doc raw"""
    Metaprogramming

Macros and quote macros for manipulating variables and regular expressions.
"""
module Metaprogramming

export @defvar,
    @strvar, @strvars,
    @p_str, @dq_str, @sq_str,
    NUMBER_REGEXES,
    infer_type

@doc raw"""
    @defvar(name, value)

Define a variable with given name (stored `name`) and value (stored in `value`).

# Example:
    ```julia
    julia> name_sym = :myvar1
    julia> value1 = "foo"
    # Declare variable `myvar1` with value `"foo"`
    julia> @defvar name_sym value1
    julia> myvar1
    "foo"

    julia> name_str = "myvar2"
    julia> value2 = 5
    # Declare variable `myvar2` with value `5`
    julia> @defvar name_str value2
    julia> myvar2
    5

    # The value can also be given to the macro directly:
    julia> @defvar name_sym "foo"
    julia> @defvar name_str 5
    ```

Since macros are expanded at parse time, in order to declare variable in a loop:

    ```julia
    for (name,value) in zip(names,values)
        @eval(@defvar($name, $value))
    end
    ```

# Arguments:
    - `name`: Variable name
    - `value`: Value to be assigned to variable
"""
macro defvar(name, value)
    name = esc(Symbol(eval(name)))
    value = esc(eval(value))
    return :($name = $value)
end

@doc raw"""
    @strvar(var::Symbol)

String representation of variable and its value.

# Example:
    ```julia
    julia> myvar = 3
    julia> @strvar myvar
    "myvar=3"
    ```

# Arguments:
    - `var::Symbol`: Variable with name `name` and value value`

# Returns:
    - String: `"name=value"`

See also: [`@strvars`](@ref).
"""
macro strvar(var::Symbol)
    name_str = string(var)
    value_str = string(eval(var))
    return :($name_str * "=" * $value_str)
end

@doc raw"""
    @strvars([sep::String="_"], vars::Symbol...)

String representation of multiple variables.

# Example:
    ```julia
    julia> myvar1 = "foo"
    julia> myvar2 = 5
    julia> @strvars "," myvar1 myvar2
    "myvar1=foo,myvar2=5"
    julia> @strvars myvar1 myvar2
    "myvar1=foo_myvar2=5"
    ```

# Arguments:
    - `sep::String`: Variable separator (default = "_")
    - `vars::Symbol...`: Variables (names `variable1`, `variable2`, ..., `variableN` and values `value1`, `value2`, ..., `valueN`)

# Retuns:
    - String: `"variable1=value1'sep'variable2=value2'sep'...'sep'variableN=valueN"` for a given `sep='sep'`

See also: [`@strvar`](@ref).
"""
macro strvars(sep::String, vars::Symbol...)
    result = :(@strvar($(vars[1])))
    for k in 2:length(vars)
        result = :($result * $sep * @strvar($(vars[k])))
    end
    return result
end
macro strvars(vars::Symbol...)
    return :(@strvars("_", $(vars...)))
end

# Quote macros (Suggested by https://stackoverflow.com/a/20483464)

@doc raw"""
    p"quote"

p"quote" macro

Since regular strings do not allow for escape chars and the r"quote" macro
compiles the regular expression when defined, this intermediate quote allows
us to store parts of regular expressions that can later be combined and compiled.

# Example:
    ```julia
    julia> pq1 = p"\\D+"
    julia> pq2 = p"\\d+"
    julia> re = Regex(pq1 * p"=" * pq2)
    julia> re
    r"\\D+=\\d+"
    ```
"""
macro p_str(s)
    s
end

@doc raw"""
Double quotes quote string

# Usage:
    dq"with double quotes" -> "with double quotes"
"""
macro dq_str(s)
    "\"" * s * "\""
end

@doc raw"""
Simple quotes quote string

# Usage:
    sq"with simple quotes" -> 'with simple quotes'
"""
macro sq_str(s)
    "'" * s * "'"
end

# Useful p"quotes" for building regular expressions

"Regex to match any number (integer or floating point)"
const number_regex = p"[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?"
"Regex to match integers"
const int_regex = p"[-+]?[0-9]*([eE][+]?[0-9]+)?"
"Regex to match floating point numbers"
const float_regex = p"[-+]?[0-9]*\.[0-9]+([eE][-+]?[0-9]+)?"

# Regular expressions that match certain types
@doc raw"""
    NUMBER_REGEXES
"""
const NUMBER_REGEXES = Dict{Type,String}(
    Int64 => p"[-+]?[0-9]*([eE][+]?[0-9]+)?",
    Float64 => p"[-+]?[0-9]*\.[0-9]+([eE][-+]?[0-9]+)?",
    Complex{Int64} => p"[-+]?[0-9]*([eE][+]?[0-9]+)?([ ]*)?[-+]([ ]*)?[0-9]*([eE][+]?[0-9]+)?([ ]*)[ij]",
    Complex{Float64} => p"[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?([ ]*)?[-+]([ ]*)?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?([ ]*)[ij]",
    Real => p"[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?",
)

@doc raw"""
    infer_type(value::AbstractString)::Type

Try to infer the type of the value in the string `value`.

If no type could be inferred, returns `Any`.
"""
function infer_type(value::AbstractString)::Type
    # Test if a number
    for Type in (Int64, Float64, Complex{Int64}, Complex{Float64})
        re = Regex(p"^" * NUMBER_REGEXES[Type] * p"$")
        if occursin(re, strip(value))
            return Type
        end
    end
    # If not a number, return Any
    return Any
end

end
