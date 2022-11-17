@doc raw"""
    Metaprogramming

Macros and quote macros for manipulating variables and regular expressions.
"""
module Metaprogramming

export
    # Extract from ::Val
    extract_val,
    # Define variable
    @defvar,
    # Print variables
    @strvar, @strvars,
    # Special strings
    @dq_str, @sq_str, @p_str,
    # Regexes
    NUMBER_REGEXES,
    # Infer variable type from string
    infer_type, infer_type_sized

@doc raw"""
    extract_val(::Val{X}) where {X}

Extracts the static value of a value type.
"""
@inline @generated extract_val(::Val{X}) where {X} = X

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

# Regular expressions that match certain types
@doc raw"""
    NUMBER_REGEXES

Dictionary that associates each numerical type with the corresponding regular expression used to infer it.
"""
const NUMBER_REGEXES = Dict{Type,String}(
    # Real numbers
    Integer => p"[-+]?[0-9]*([eE][+]?[0-9]+)?",
    AbstractFloat => p"[-+]?[0-9]*\.[0-9]+([eE][-+]?[0-9]+)?",
    # Complex numbers
    Complex{Integer} => p"[-+]?[0-9]*([eE][+]?[0-9]+)?([ ]*)?[-+]([ ]*)?[0-9]*([eE][+]?[0-9]+)?([ ]*)[ij]",
    Complex{AbstractFloat} => p"[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?([ ]*)?[-+]([ ]*)?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?([ ]*)[ij]",
)

@doc raw"""
    infer_type(value::AbstractString)::Type

Try to infer the type of the numerical value in the string `value`.

If no type could be inferred, returns `Any`.
"""
function infer_type(value::AbstractString)::Type
    # Test numerical types in a given order
    for Type in (Integer, AbstractFloat, Complex{Integer}, Complex{AbstractFloat})
        re = Regex(p"^" * NUMBER_REGEXES[Type] * p"$")
        if occursin(re, strip(value))
            return Type
        end
    end
    # If no numerical type could be inferred, return `Any`
    return Any
end

function infer_type_sized(value::AbstractString)::Type
    InferredType = infer_type(value)
    if InferredType == Integer
        return Int64
    elseif InferredType == AbstractFloat
        return Float64
    elseif InferredType == Complex{Integer}
        return Complex{Int64}
    elseif InferredType == Complex{AbstractFloat}
        return Complex{Float64}
    else
        return InferredType
    end
end

end
