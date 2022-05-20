@doc raw"""
    Metaprogramming
"""
module Metaprogramming

export @defvar, @defvarstr,
    @strvar, @strvars,
    @p_str,
    number_regex, int_regex, float_regex

using Logging, JLD2

macro sym2str(sym::Symbol)
    return :(String(sym))
end

@doc raw"""
    @defvar(name::Symbol, value)

Define a variable with given symbol and value.

# Example:
    ```julia
    var_sym = :new_var
    var_value = 5
    @defvar(var_sym, var_value)
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
    - `name::Symbol`: Symbol with variable name
    - `value`: Value to be assigned to variable
"""
macro defvar(name::Symbol, value)
    return :($(esc(eval(name))) = $(esc(eval(value))))
end

@doc raw"""
    @defvarstr(name::String, value)

Define a variable with given name and value.

# Example:
    ```julia
    var_name = "new_var"
    var_value = 5
    @defvarstr(var_name, var_value)
    # The variable `new_var` is decalerd with value `5`
    println(new_var)
    # Prints `5`
    ```

Since macros are expanded at parse time, in order to declare variable in a loop:

    ```julia
    for (name,value) in zip(names,values)
        @eval(@defvarstr($name, $value))
    end
    ```

# Arguments:
    - `name::String`: String with variable name
    - `value`: Value to be assigned to variable
"""
macro defvarstr(name::String, value)
    name_sym = Symbol(eval(name))
    return :($(esc(eval(name_sym))) = $(esc(eval(value))))
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
