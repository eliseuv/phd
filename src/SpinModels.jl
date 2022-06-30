module SpinModels

export SpinState, SpinHalfState, SpinOneState

using EnumX, Random, Distributions, Graphs

include("Metaprogramming.jl")
include("Geometry.jl")

using .Metaprogramming

"""
    SpinHalfState::Int8 <: SpinState

Enumeration of possible spin `1/2` values.
"""
@enumx SpinHalfState::Int8 begin
    down = -1
    up = +1
end

"""
    SpinOneState::Int8 <: SpinState

Enumeration of possible spin `1` values.
"""
@enumx SpinOneState::Int8 begin
    down = -1
    zero = 0
    up = +1
end

"""
    SpinState

Supertype for all spin states.
"""
SpinState = Union{SpinOneState.T,SpinHalfState.T}

"""
    convert(::Type{T}, σ::SpinState) where {T<:Number}

Use the integer representation of `σ::SpinState` in order to convert it to a numerical type `T<:Number`.
"""
@inline Base.convert(::Type{T}, σ::SpinState) where {T<:Number} = T(Integer(σ))

"""
    promote_rule(T::Type, ::Type{SpinState})

Always try to promote the `SpinState` to whatever the other type is.
"""
@inline Base.promote_rule(T::Type, ::Type{SpinState}) = T

# Arithmetic with numbers and spin states
for op in (:*, :/, :+, :-)
    @eval begin
        @inline Base.$op(x::Number, σ::SpinState) = $op(promote(x, σ)...)
        @inline Base.$op(σ::SpinState, y::Number) = $op(promote(σ, y)...)
    end
end

"""
    *(σ₁::SpinState, σ₂::SpinState)

Multiplication of spin states.
"""
@inline Base.:*(σ₁::SpinState, σ₂::SpinState) = Integer(σ₁) * Integer(σ₂)


"""
    show(io::IO, ::MIME"text/plain", σ::SpinHalfState)

Text representation of `SpinHalfState`.
"""
function Base.show(io::IO, ::MIME"text/plain", σ::SpinHalfState.T)
    spin_char = σ == up ? '↑' : '↓'
    print(io, spin_char)
end

"""
    show(io::IO, ::MIME"text/plain", σ::SpinOneState)

Text representation of `SpinOneState`.
"""
function Base.show(io::IO, ::MIME"text/plain", σ::SpinOneState.T)
    spin_char = σ == up ? '↑' : σ == down ? '↓' : '-'
    print(io, spin_char)
end

"""
    SpinModel <: AbstractArray{SpinState}

Supertype for all spin models.

Since all site are indexable, they all inherit from `AbstractArray{SpinState}`.
"""
abstract type SpinModel{T} where {T<:SpinState} end

@doc raw"""
    MeanFieldSpinModel <: SpinModel

Spin system with mean field interaction:
Every spin interacts equally with every other spin.

Since in the mean field model there is no concept of space and locality,
we represent the state of the system simply by total number of spins in each state.

The `state` member is therefore of type `NamedTuple`.

An `AbstractVector{SpinState}` interface for the `MeanFieldSpinModel` type can be implemented
if we assume that the all spin states are stored in a sorted vector.

Assume the possible spin values are `s₁`, `s₂`, etc.

    σ = (s₁, s₁, …, s₁, s₂, s₂, …, s₂, …)
        |----- N₁ ----||----- N₂ ----|
        |-------------- N ---------------|

Therefore, for an `spinmodel::MeanFieldSpinModel` we can access the `i`-th spin `σᵢ = spinmodel[i]`:
    σᵢ = sₖ if Nₖ₋₁ < i ≤ Nₖ
"""
abstract type MeanFieldSpinModel{T} <: SpinModel{T} where {T<:SpinState} end

@doc raw"""
    length(spinmodel::MeanFieldSpinModel)

Total number of spins (`N`) in an spin system with mean field interaction `spinmodel`
is simply the sum of the number of spins in each state.
"""
Base.length(spinmodel::MeanFieldSpinModel) = sum(spinmodel.state)

@doc raw"""
    IndexStyle(::MeanFieldSpinModel)

Use only linear indices for the `AbstractVector{SpinState}` interface for the `MeanFieldSpinModel` type.
"""
@inline Base.IndexStyle(::Type{<:MeanFieldSpinModel}) = IndexLinear()

@doc raw"""
    firstindex(spinmodel::MeanFieldSpinModel)

Index of the first spin site in the `AbstractVector{SpinState}` interface of `MeanFieldSpinModel` is `1`.
"""
@inline Base.firstindex(spinmodel::MeanFieldSpinModel) = 1

@doc raw"""
    lastindex(spinmodel::MeanFieldSpinModel)

Index of the last spin site in the `AbstractVector{SpinState}` interface of `MeanFieldSpinModel` is equal the total number of sites `N`.
"""
@inline Base.lastindex(spinmodel::MeanFieldSpinModel) = length(spinmodel)

@doc raw"""
    ConcreteSpinModel <: SpinModel

Supertype for all spin models that have a concrete representation of its state in memory
in the form of a concrete array member `state::Array{SpinState}`.

The whole indexing interface of the `state::Array{SpinState}` can be passed to the `::ConcreteSpinModel` object itself.
"""
abstract type ConcreteSpinModel <: SpinModel end

"""
    length(spinmodel::ConcreteSpinModel)

Total number of sites of an spin system `spinmodel`.
"""
@inline Base.length(spinmodel::ConcreteSpinModel) = length(spinmodel.state)

"""
    size(spinmodel::ConcreteSpinModel)

Size of the state of an spin system `spinmodel`.
"""
@inline Base.size(spinmodel::ConcreteSpinModel) = size(spinmodel.state)

"""
    size(spinmodel::ConcreteSpinModel, dim)

Size of the state of an spin system `spinmodel` along a given dimension `dim`.
"""
@inline Base.size(spinmodel::ConcreteSpinModel, dim) = size(spinmodel.state, dim)

"""
    IndexStyle(::Type{<:ConcreteSpinModel{N}}) where {N}

Use same indexing style used to index the state array.
"""
@inline Base.IndexStyle(::Type{<:ConcreteSpinModel}) = IndexStyle(Array{SpinState})

"""
    getindex(spinmodel::ConcreteSpinModel, inds...)

Index the spin system itself to access its state.
"""
@inline Base.getindex(spinmodel::ConcreteSpinModel, inds...) = getindex(spinmodel.state, inds...)

"""
    setindex!(spinmodel::ConcreteSpinModel, σ, inds...)

Set the state of a given spin at site `i` to `σ` in the spin system `spinmodel`.
"""
@inline Base.setindex!(spinmodel::ConcreteSpinModel, σ, inds...) = setindex!(spinmodel.state, σ, inds...)

"""
    firstindex(spinmodel::ConcreteSpinModel)

Get the index of the first spin in the system.
"""
@inline Base.firstindex(spinmodel::ConcreteSpinModel) = firstindex(spinmodel.state)

"""
    lastindex(spinmodel::ConcreteSpinModel)

Get the index of the last spin in the system.
"""
@inline Base.lastindex(spinmodel::ConcreteSpinModel) = lastindex(spinmodel.state)

"""
    set_state!(spinmodel::ConcreteSpinModel{T}, σ₀::T) where {T<:SpinState}

Set the state of all sites of an Spinmodel system `spinmodel` to a given site state `σ₀`.
"""
@inline function set_state!(spinmodel::ConcreteSpinModel{T}, σ₀::T) where {T<:SpinState}
    fill!(spinmodel, σ₀)
end

"""
    randomize_state!(spinmodel::ConcreteSpinModel{T}) where {T<:SpinState}

Set the state of each site of an spin system `spinmodel` to a random state `σ₀ ∈ SpinState`.
"""
@inline function randomize_state!(spinmodel::ConcreteSpinModel{T}) where {T<:SpinState}
    rand!(spinmodel, instances(T))
end

end
