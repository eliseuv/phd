# Abstract super type
abstract type Super end

# Method definition for Super type
f(::Super) = "Method for Super."

# Structs derived from Super type
struct A <: Super end
struct B <: Super end

# Method defines specifically for struct B
f(::B) = "Method for B."

# Test for A
a = A()
@show f(a)

# Test for B
b = B()
@show f(b)
