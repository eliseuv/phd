@doc raw"""
    Spin Models

"""
module SpinModels

export
    # Single spin states
    SingleSpinState, SpinHalfState, SpinOneState,
    # Properties of single spin states
    rand_new_spin,
    # Measurements in spin states
    magnet_total, magnet,
    # Measurements on spins states that depend on locality
    energy_interaction,
    # Symmetries explored
    flip!,
    # Spin models
    AbstractSpinModel,
    # General properties of spin models
    state, state_type, spin_type, spin_instances, state,
    # Metropolis sampling
    metropolis!,
    metropolis_measure!,
    metropolis_measure_energy!,
    # Heatbath sampling
    heatbath_weights,
    heatbath_measure!,
    # Energy measurements
    energy,
    energy_local, minus_energy_local,
    energy_diff,
    # Ising models
    AbstractIsingModel,
    # Implementations of Ising models
    IsingModel, IsingModelExtField,
    # Blume-Capel models
    AbstractBlumeCapelModel,
    # Implementations of Blume-Capel models
    BlumeCapelModel

using Random, EnumX, Combinatorics, StatsBase, Distributions, Graphs

using ..FiniteStates

"""
    SpinHalfState::Int8 <: SingleSpinState

Enumeration of possible spin `1/2` values.
"""
@enumx SpinHalfState::Int8 begin
    down = -1
    up = +1
end

"""
    other_spin(σ::SpinHalfState.T)

Returns the complementary of the spin-`1/2` state `σ`:
    - up    => down
    - down  => up
"""
@inline other_spin(σ::SpinHalfState.T) = SpinHalfState.T(-Integer(σ))

"""
    SpinOneState::Int8 <: SingleSpinState

Enumeration of possible spin `1` values.
"""
@enumx SpinOneState::Int8 begin
    down = -1
    zero = 0
    up = +1
end

"""
    SingleSpinState

Supertype for all spin states.

They are usually enums, but even if they are not enums,
all single spin states must provide a method `instances(<:SingleSpinState)` that returns a tuple with all possible single spin states.
"""
SingleSpinState = Union{SpinOneState.T,SpinHalfState.T}

"""
    rand_new_spin(σ::T) where {T<:SingleSpinState}

Select a new random single spin state `σ′ ∈ SingleSpinState` different from `σ`.
"""
@inline rand_new_spin(σ::T) where {T<:SingleSpinState} = rand(filter(!=(σ), instances(T)))

"""
    rand_new_spin(σ::SpinHalfState.T)

Returns the complementary of the single spin state `σ`.
"""
@inline rand_new_spin(σ::SpinHalfState.T) = other_spin(σ)

"""
    convert(::Type{T}, σ::SingleSpinState) where {T<:Number}

Use the integer representation of `σ::SingleSpinState` in order to convert it to a numerical type `T<:Number`.
"""
@inline Base.convert(::Type{T}, σ::SingleSpinState) where {T<:Number} = T(Integer(σ))

"""
    promote_rule(T::Type, ::Type{SingleSpinState})

Always try to promote the `SingleSpinState` to whatever the other type is.
"""
@inline Base.promote_rule(T::Type, ::Type{SingleSpinState}) = T

# Arithmetic with numbers and spin states
for op in (:*, :/, :+, :-)
    @eval begin
        @inline Base.$op(x::Number, σ::SingleSpinState) = $op(promote(x, σ)...)
        @inline Base.$op(σ::SingleSpinState, y::Number) = $op(promote(σ, y)...)
    end
end

"""
    *(σ₁::SingleSpinState, σ₂::SingleSpinState)

Multiplication of spin states.
"""
@inline Base.:*(σ₁::SingleSpinState, σ₂::SingleSpinState) = Integer(σ₁) * Integer(σ₂)

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
    magnet_total(fs::AbstractFiniteState)

Total magnetization of a spin state `fs`.
"""
@inline magnet_total(fs::AbstractFiniteState) = sum(fs)

@doc raw"""
    magnet(fs::AbstractFiniteState)

Magnetization of the spin state `fs`.

``m = M / N = (1/N) ∑ᵢ sᵢ``
"""
@inline magnet(fs::AbstractFiniteState) = magnet_total(fs) / length(fs)

@doc raw"""
    energy_interaction(fs::AbstractFiniteState)

Interaction energy for a spin state `fs`.

``H_{int} = - ∑_⟨i,j⟩ sᵢ sⱼ``

where `⟨i,j⟩` represents a pair of nearest neighbors sites.
"""
@inline energy_interaction(fs::AbstractFiniteState) = @inbounds -sum(Integer(fs[i]) * Integer(fs[j]) for (i, j) ∈ nearest_neighbors(fs))

"""
    energy_interaction(fs::MeanFieldFiniteState{T}) where {T<:SingleSpinState}

Get the interaction energy of the mean field spin state `fs`.
"""
function energy_interaction(fs::MeanFieldFiniteState{T}) where {T<:SingleSpinState}
    S_equal = sum(instances(T)) do σₖ
        Nₖ = fs.counts[σₖ]
        return ((Nₖ * (Nₖ - 1)) ÷ 2) * Integer(σₖ)^2
    end
    S_diff = sum(combinations(instances(T), 2)) do (σₖ, σₗ)
        Nₖ = fs.counts[σₖ]
        Nₗ = fs.counts[σₗ]
        return Nₖ * Nₗ * Integer(σₖ) * Integer(σₗ)
    end
    return S_equal + S_diff
end

@doc raw"""
    energy_interaction(fs::MeanFieldFiniteState{SpinHalfState.T})

Interaction energy of the spin-`1/2` spin state with mean field interaction `fs`.

``H_{int} = - ∑_⟨i,j⟩ sᵢsⱼ = (N - M^2) / 2``
"""
@inline energy_interaction(fs::MeanFieldFiniteState{SpinHalfState.T}) = (length(fs) - magnet_total(fs)^2) ÷ 2

@doc raw"""
    energy_interaction(fs::SquareLatticeFiniteState)

Interaction energy for a `N`-dimensional square lattice spin model `fs`.

``H_{int} = - \sum_⟨i,j⟩ σᵢ σⱼ``
"""
function energy_interaction(fs::SquareLatticeFiniteState{T,N}) where {T<:SingleSpinState,N}
    # Varaible to accumulate
    H = zero(Int64)
    # Loop on dimensions
    @inbounds for d ∈ 1:N
        # Bulk
        front_bulk = selectdim(state(fs), d, 1:(size(fs, d)-1))
        back_bulk = selectdim(state(fs), d, 2:size(fs, d))
        H -= sum(Integer, front_bulk .* back_bulk)
        # Periodic boundaries
        last_slice = selectdim(state(fs), d, size(fs, d))
        first_slice = selectdim(state(fs), d, 1)
        H -= sum(Integer, last_slice .* first_slice)
    end
    return H
end

"""
    energy_interaction(fs::SimpleGraphFiniteState)

Get the interaction energy for a spin state on a simple graph `fs`.
"""
@inline energy_interaction(fs::SimpleGraphFiniteState) = @inbounds -sum(edges(fs.graph)) do edge
    fs[src(edge)] * fs[dst(edge)]
end

"""
    flip!(fs::AbstractFiniteState{SpinHalfState.T}, i)

Flips the `i`-th spin in the spin-`1/2` state `fs`.
"""
@inline function flip!(fs::AbstractFiniteState{SpinHalfState.T}, i)
    @inbounds fs[i] = SpinHalfState.T(-Integer(fs[i]))
end

@doc raw"""
    flip!(fs::MeanFieldSpinState{SpinHalfState.T}, i::Integer)

Flip the state of the `i`-th spin in the spin-`1/2` state with mean field interaction `fs`.
"""
@inline function flip!(fs::MeanFieldFiniteState{SpinHalfState.T}, i::Integer)
    sᵢ = Integer(fs[i])
    fs[SpinHalfState.up] -= sᵢ
    fs[SpinHalfState.down] += sᵢ
end

"""
    flip!(fs::MeanFieldFiniteState{SpinHalfState.T}, σ::SpinHalfState.T)

Flips one spin with state `σ` in the spin-`1/2` mean field state `fs`.
"""
@inline function flip!(fs::MeanFieldFiniteState{SpinHalfState.T}, σ::SpinHalfState.T)
    fs[σ] -= 1
    fs[other_spin(σ)] += 1
end

"""
    AbstractSpinModel{T<:AbstractFiniteState}

Supertype for all spin models.
"""
abstract type AbstractSpinModel{T<:AbstractFiniteState} end

"""
    single_spin_type(::AbstractSpinModel)

Get the type of the single spin state of a given spin model.
"""
@inline spin_type(::AbstractSpinModel{<:AbstractFiniteState{T}}) where {T} = T

"""
    single_spin_values(::AbstractSpinModel)

Get a tuple with the possible instances of single spin state.
"""
@inline spin_instances(::AbstractSpinModel{<:AbstractFiniteState{T}}) where {T} = instances(T)

"""
    state_type(::AbstractSpinModel)

Get the type of the spin state of a given spin model.
"""
@inline state_type(::AbstractSpinModel{T}) where {T} = T

"""
    state(spinmodel::AbstractSpinModel)

Get the spins state associated with a given spin model.
"""
@inline state(spinmodel::AbstractSpinModel) = spinmodel.state

"""
    length(spinmodel::AbstractSpinModel)

Total number of sites of an spin system `spinmodel`.
"""
@inline Base.length(spinmodel::AbstractSpinModel) = length(state(spinmodel))

"""
    size(spinmodel::AbstractSpinModel)

Size of the spins of an spin system `spinmodel`.
"""
@inline Base.size(spinmodel::AbstractSpinModel) = size(state(spinmodel))

"""
    IndexStyle(::Type{<:AbstractSpinModel})

Use the same index style from the spin state.
"""
@inline Base.IndexStyle(::Type{<:AbstractSpinModel{T}}) where {T} = IndexStyle(T)

"""
    getindex(spinmodel::AbstractSpinModel, inds...)

Index the spin system itself to access its spins.
"""
@inline Base.getindex(spinmodel::AbstractSpinModel, inds...) = getindex(state(spinmodel), inds...)

"""
    setindex!(spinmodel::AbstractSpinModel, σ, inds...)

Set the spins of a given spin at site `i` to `σ` in the spin system `spinmodel`.
"""
@inline Base.setindex!(spinmodel::AbstractSpinModel, σ, inds...) = setindex!(state(spinmodel), σ, inds...)

"""
    firstindex(spinmodel::AbstractSpinModel)

Get the index of the first spin in the system.
"""
@inline Base.firstindex(spinmodel::AbstractSpinModel) = firstindex(state(spinmodel))

"""
    lastindex(spinmodel::AbstractSpinModel)

Get the index of the last spin in the system.
"""
@inline Base.lastindex(spinmodel::AbstractSpinModel) = lastindex(state(spinmodel))

@inline magnet_total(spinmodel::AbstractSpinModel) = magnet_total(state(spinmodel))

@inline magnet(spinmodel::AbstractSpinModel) = magnet(state(spinmodel))

@inline energy_interaction(spinmodel::AbstractSpinModel) = energy_interaction(state(spinmodel))

# Allow spin state measurements to be done directly on the spin model
# for func in (:magnet_total, :magnet, :energy_interaction)
#     @eval begin
#         @inline $func(spinmodel::AbstractSpinModel) = $func(state(spinmodel))
#     end
# end

@inline function flip!(spinmodel::AbstractSpinModel, i)
    flip!(state(spinmodel), i)
end

"""
    metropolis!(spinmodel::AbstractSpinModel, β::Real, n_steps::Integer)

Sample using the Metropolis algorithm the spin model `spinmodel` at temperature `β` for `n_steps` steps.
"""
function metropolis!(spinmodel::AbstractSpinModel, β::Real, n_steps::Integer)
    # Loop on random sites
    @inbounds for i ∈ rand(eachindex(state(spinmodel)), n_steps)
        # Select random new state
        sᵢ′ = rand_new_spin(spinmodel[i])
        # Get energy difference
        ΔH = energy_diff(spinmodel, i, sᵢ′)
        # Metropolis prescription
        if ΔH < 0 || exp(-β * ΔH) > rand()
            # Change spin
            spinmodel[i] = sᵢ′
        end
    end
end

"""
    metropolis!(spinmodel::AbstractSpinModel{<:AbstractFiniteState{SpinHalfState.T}}, β::Real, n_steps::Integer)

Sample using the Metropolis algorithm the spin-`1/2` model `spinmodel` at temperature `β` for `n_steps` steps.
"""
function metropolis!(spinmodel::AbstractSpinModel{<:AbstractFiniteState{SpinHalfState.T}}, β::Real, n_steps::Integer)
    # Loop on random sites
    @inbounds for i ∈ rand(eachindex(state(spinmodel)), n_steps)
        # Get energy difference
        ΔH = energy_diff(spinmodel, i)
        # Metropolis prescription
        if ΔH < 0 || exp(-β * ΔH) > rand()
            # Flip spin
            flip!(spinmodel, i)
        end
    end
end

"""
    metropolis!(spinmodel::AbstractSpinModel{MeanFieldFiniteState{SpinHalfState.T}}, β::Real, n_steps::Integer)

Sample using the Metropolis algorithm the spin-`1/2` mean field model `spinmodel` at temperature `β` for `n_steps` steps.
"""
function metropolis!(spinmodel::AbstractSpinModel{MeanFieldFiniteState{SpinHalfState.T}}, β::Real, n_steps::Integer)
    # Loop on random sites
    @inbounds for _ ∈ 1:n_steps
        σ = if rand() < (spinmodel.state.counts[SpinHalfState.up] / length(spinmodel))
            SpinHalfState.up
        else
            SpinHalfState.down
        end
        # Get energy difference
        ΔH = energy_diff(spinmodel, σ)
        # Metropolis prescription
        if ΔH < 0 || exp(-β * ΔH) > rand()
            # Flip spin
            flip!(spinmodel, σ)
        end
    end
end

"""
    metropolis_measure!(measurement::Function, spinmodel::AbstractSpinModel, β::Real, n_steps::Integer)

Metropolis sample the spin model `spinmodel` at temperature `β` for `n_steps`
and perform the measurement `measurement` on the spin model at the end of each step.

Note that a single sampling step is equivalent to `N` metropolis prescription steps,
where `N` is the total number of sites in the system.
"""
function metropolis_measure!(measurement::Function, spinmodel::T, β::Real, n_steps::Integer) where {T<:AbstractSpinModel}
    # Results vector
    ResultType = Base.return_types(measurement, (T,))[1]
    results = Vector{ResultType}(undef, n_steps + 1)
    # Initial measurement
    results[1] = measurement(spinmodel)
    # Sampling loop
    @inbounds for t ∈ 1:n_steps
        # Metropolis sampling
        metropolis!(spinmodel, β, length(spinmodel))
        # Update results vector
        results[t+1] = measurement(spinmodel)
    end
    # Return measurement results
    return results
end

"""
    metropolis_measure_energy!(spinmodel::AbstractSpinModel, β::Real, n_steps::Integer)

Sample the spin model `spinmodel` using the Metropolis algorithm at temperature `β` for `n_steps` steps
returning a vector containing the value of the system energy at the end of each step.

Note that a single sampling step is equivalent to `N` metropolis prescription steps,
where `N` is the total number of sites in the system.
"""
function metropolis_measure_energy!(spinmodel::T, β::Real, n_steps::Integer) where {T<:AbstractSpinModel}
    # Energy measurements vector
    EnergyType = Base.return_types(energy, (T,))[1]
    results = Vector{EnergyType}(undef, n_steps + 1)
    # Initial energy measurement
    results[1] = energy(spinmodel)
    # Sampling loop
    @inbounds for t ∈ 1:n_steps
        ΔH_total = zero(EnergyType)
        # Site loop
        @inbounds for i ∈ rand(eachindex(state(spinmodel)), length(spinmodel))
            # Select random new state
            sᵢ′ = rand_new_spin(spinmodel[i])
            # Get energy difference
            ΔH = energy_diff(spinmodel, i, sᵢ′)
            # Metropolis prescription
            if ΔH < 0 || exp(-β * ΔH) > rand()
                # Change spin
                spinmodel[i] = sᵢ′
                # Add energy difference
                ΔH_total += ΔH
            end
        end
        # Update energy measurements vector
        results[t+1] = results[t] + ΔH_total
    end
    return results
end

"""
    metropolis_measure_energy!(spinmodel::AbstractSpinModel{<:AbstractFiniteState{SpinHalfState.T}}, β::Real, n_steps::Integer)

Sample the spin-`1/2` model `spinmodel` using the Metropolis algorithm at temperature `β` for `n_steps` steps
returning a vector containing the value of the system energy at the end of each step.

Note that a single sampling step is equivalent to `N` metropolis prescription steps,
where `N` is the total number of sites in the system.

This implementation takes advantage of symmetries in spin-`1/2` systems to simplify the algorithm.
"""
function metropolis_measure_energy!(spinmodel::T, β::Real, n_steps::Integer) where {T<:AbstractSpinModel{<:AbstractFiniteState{SpinHalfState.T}}}
    # Energy measurements vector
    EnergyType = Base.return_types(energy, (T,))[1]
    results = Vector{EnergyType}(undef, n_steps + 1)
    # Initial energy measurement
    results[1] = energy(spinmodel)
    # Sampling loop
    @inbounds for t ∈ 1:n_steps
        ΔH_total = zero(EnergyType)
        # Site loop
        @inbounds for i ∈ rand(eachindex(state(spinmodel)), n_steps)
            # Get energy difference
            ΔH = energy_diff(spinmodel, i)
            # Metropolis prescription
            if ΔH < 0 || exp(-β * ΔH) > rand()
                # Change spin
                flip!(spinmodel, i)
                # Add energy difference
                ΔH_total += ΔH
            end
        end
        # Update energy measurements vector
        results[t+1] = results[t] + ΔH_total
    end
    return results
end

@doc raw"""
    heatbath_weights(spinmodel::AbstractSpinModel, i, β::Real)

Calculate the weights required by the heatbath sampling algorithm for the spin model `spinmodel` for the `i`-th site and at temperature `β`.

The weight associated with the single spin state `σ` at the `i`-th site at temperature `β` is:

``w(σ, i, β) = exp(-β hᵢ(σ))``

where `hᵢ(σ)` is the local energy associated with `i`-th site assuming that its state is `σ`.
"""
@inline heatbath_weights(spinmodel::AbstractSpinModel{<:AbstractFiniteState{T}}, i, β::Real) where {T} =
    map(σ -> exp(β * minus_energy_local(spinmodel, i, σ)), [instances(T)...]) |> ProbabilityWeights

"""
    heatbath!(spinmodel::AbstractSpinModel, β::Real, n_steps::Integer)

Sample the spin model `spinmodel` using the heatbath algorithm at temperature `β` for `n_steps` steps.
"""
function heatbath!(spinmodel::AbstractSpinModel{AbstractFiniteState{T}}, β::Real, n_steps::Integer) where {T}
    # Site loop
    @inbounds for i ∈ rand(eachindex(state(spinmodel)), n_steps)
        weights = heatbath_weights(spinmodel, i, β)
        spinmodel[i] = sample([instances(T)...], weights)
    end
end

"""
    heatbath_measure!(measurement::Function, spinmodel::AbstractSpinModel, β::Real, n_steps::Integer)

Heatbath sample the spin model `spinmodel` at temperature `β` for `n_steps`
and perform the measurement `measurement` on the spin model at the end of each step.

Note that a single system sampling step is equivalent to `N` heatbath prescription steps,
where `N` is the total number of sites in the system.
"""
function heatbath_measure!(measurement::Function, spinmodel::T, β::Real, n_steps::Integer) where {S,T<:AbstractSpinModel{<:AbstractFiniteState{S}}}
    # Results vector
    ResultType = Base.return_types(measurement, (T,))[1]
    results = Vector{ResultType}(undef, n_steps + 1)
    # Initial measurement
    results[1] = measurement(spinmodel)
    # Sampling loop
    @inbounds for t ∈ 1:n_steps
        # Heatbath sampling
        heatbath!(spinmodel, β, length(spinmodel))
        # Update measurement vector
        results[t+1] = measurement(spinmodel)
    end
    # Return measurement results
    return results
end

"""
    heatbath_measure_energy!(spinmodel::AbstractSpinModel, β::Real, n_steps::Integer)

Sample the spin model `spinmodel` using the heatbath algorithm at temperature `β` for `n_steps` steps
returning a vector containing the value of the system energy at the end of each step.

Note that a single sampling step is equivalent to `N` heatbath steps,
where `N` is the total number of sites in the system.
"""
function heatbath_measure_energy!(spinmodel::T, β::Real, n_steps::Integer) where {S,T<:AbstractSpinModel{<:AbstractFiniteState{S}}}
    # Energy measurements vector
    EnergyType = Base.return_types(energy, (T,))[1]
    results = Vector{EnergyType}(undef, n_steps + 1)
    # Initial energy measurement
    results[1] = energy(spinmodel)
    # Heatbath weights dictionary
    minus_H_local = Dict{S,Float64}()
    # Sampling loop
    @inbounds for t ∈ 1:n_steps
        ΔH_total = zero(EnergyType)
        # Site loop
        @inbounds for i ∈ rand(eachindex(state(spinmodel)), length(spinmodel))
            # Store current state
            σᵢ = spinmodel[i]
            # Calcualte minus energy local
            minus_H_local = Dict(σ => minus_energy_local(spinmodel, i, σ) for σ ∈ instances(S))
            # Calculate weights
            weights = ProbabilityWeights([exp(β * minus_h) for minus_h ∈ values(minus_H_local)])
            # Get new state
            σᵢ′ = sample([instances(T)...], weights)
            spinmodel[i] = σᵢ′
            # Add energy difference
            ΔH_total += minus_H_local[σᵢ] - minus_H_local[σᵢ′]
        end
        # Update measurement vector
        results[t+1] = results[t] + ΔH_total
    end
    # Return measurement results
    return results
end

"""
    AbstractIsingModel{T} <: AbstractSpinModel{T}

Super type for all Ising models.
"""
abstract type AbstractIsingModel{T} <: AbstractSpinModel{T} end

@doc raw"""
    IsingModel{T} <: AbstractIsingModel{T}

The Ising model without external magnetic field.
"""
struct IsingModel{T} <: AbstractIsingModel{T}

    "State of the spins"
    state::T

    """
        IsingModel(state)

    Construct an Ising system without external magnetic field and with given initial spins state `spins`
    """
    IsingModel(state::T) where {T} = new{T}(state)
end

@doc raw"""
    energy(ising::IsingModel)

Total energy of an Ising system `ising`.

Given by the Hamiltonian:

``H = - ∑_⟨i,j⟩ sᵢsⱼ``

where `⟨i,j⟩` means that `i` and `j` are nearest neighbors.
"""
@inline energy(ising::IsingModel) = energy_interaction(ising)

@doc raw"""
    energy_local(ising::IsingModel, i)

Local energy of the `i`-th site in the Ising system `ising`.

``hᵢ = - sᵢ ∑ⱼ sⱼ``

where the sum is over the nearest neighbors `j` of `i`.
"""
@inline energy_local(ising::IsingModel, i) = -Integer(ising[i]) * nearest_neighbors_sum(ising.state, i)

@doc raw"""
    energy_local(ising::IsingModel, i, sᵢ)

Local energy of the `i`-th site assuming its state is `sᵢ` in the Ising system `ising`.

``hᵢ = - sᵢ ∑ⱼ sⱼ``

where the sum is over the nearest neighbors `j` of `i`.
"""
@inline energy_local(ising::IsingModel, i, sᵢ) = -Integer(sᵢ) * nearest_neighbors_sum(ising.state, i)

@doc raw"""
    energy_diff(ising::IsingModel, i, sᵢ′)

Calculate the energy difference for an Ising system `ising` if the `i`-th spin were to be changed to `sᵢ′`.

``ΔHᵢ = (sᵢ - sᵢ′) ∑ⱼ sⱼ``

where the sum is over the nearest neighbors `j` of `i`.
"""
@inline function energy_diff(ising::IsingModel, i, sᵢ′)
    sᵢ = ising[i]
    if sᵢ′ == sᵢ
        return 0
    else
        return (Integer(sᵢ) - Integer(sᵢ′)) * nearest_neighbors_sum(ising.state, i)
    end
end

@doc raw"""
    energy_diff(ising::IsingModel, i)

Calculate the energy difference for a spin-`1/2` Ising system `ising` if the `i`-th spin were to be flipped.

``ΔHᵢ = 2 sᵢ ∑ⱼ sⱼ``

where the sum is over the nearest neighbors `j` of `i`.
"""
@inline energy_diff(ising::IsingModel{<:AbstractFiniteState{SpinHalfState.T}}, i) = 2 * Integer(ising[i]) * nearest_neighbors_sum(ising.state, i)


@doc raw"""
    energy_diff(ising::IsingModel{MeanFieldFiniteState{SpinHalfState.T}}, σ::SpinHalfState.T)

Calculate the energy difference for a spin-`1/2` mean field Ising system `ising` if a spin with current state `σ` were to be flipped.

``ΔH(sᵢ) = 2 sᵢ ∑ⱼ sⱼ = 2 sᵢ (M - sᵢ)``

where the sum is over the nearest neighbors `j` of `i`.
"""
@inline energy_diff(ising::IsingModel{MeanFieldFiniteState{SpinHalfState.T}}, σ::SpinHalfState.T) = 2 * Integer(σ) * nearest_neighbors_sum(ising.state, σ)

@doc raw"""
    IsingModelExtField{T} <: AbstractIsingModel{T}

The Ising model with external magnetic field.
"""
struct IsingModelExtField{T} <: AbstractIsingModel{T}

    "State of the spins"
    state::T

    "External magnetic field"
    h::Real

    """
        IsingModelExtField(state::T, h::Real) where {T}

    Construct an Ising system with external magnetic field `h` and with given initial spins state `spins`
    """
    IsingModelExtField(state::T, h::Real) where {T} = new{T}(state, h)
end

@doc raw"""
    energy(ising::IsingModelExtField)

Total energy of an Ising system `ising`.

Given by the Hamiltonian:

``H = - ∑_⟨i,j⟩ sᵢsⱼ - h ∑ᵢ sᵢ``

where `⟨i,j⟩` means that `i` and `j` are nearest neighbors.
"""
@inline energy(ising::IsingModelExtField) = energy_interaction(ising) - ising.h * magnet_total(ising)

@doc raw"""
    energy_local(ising::IsingModelExtField, i)

Local energy of the `i`-th site in the Ising system `ising`.

``hᵢ = - sᵢ (∑ⱼ sⱼ + h)``

where the sum is over the nearest neighbors `j` of `i`.
"""
@inline energy_local(ising::IsingModelExtField, i) = -Integer(ising[i]) * (nearest_neighbors_sum(ising.state, i) + ising.h)

@doc raw"""
    energy_local(ising::IsingModelExtField, i, sᵢ)

Local energy of the `i`-th site assuming its state is `sᵢ` in the Ising system `ising`.

``hᵢ = - sᵢ (∑ⱼ sⱼ + h)``

where the sum is over the nearest neighbors `j` of `i`.
"""
@inline energy_local(ising::IsingModelExtField, i, sᵢ) = -Integer(sᵢ) * (nearest_neighbors_sum(ising.state, i) + ising.h)

@doc raw"""
    energy_diff(ising::IsingModelExtField, i, sᵢ′)

Calculate the energy difference for an Ising system `ising` if the `i`-th spin were to be changed to `sᵢ′`.

``ΔHᵢ = (sᵢ - sᵢ′) ( ∑ⱼ sⱼ + h)``

where the sum is over the nearest neighbors `j` of `i`.
"""
@inline function energy_diff(ising::IsingModelExtField, i, sᵢ′)
    sᵢ = ising[i]
    if sᵢ′ == sᵢ
        return 0
    else
        return (Integer(sᵢ) - Integer(sᵢ′)) * (nearest_neighbors_sum(ising, i) + ising.h)
    end
end

@doc raw"""
    energy_diff(ising::IsingModelExtField{<:AbstractFiniteState{SpinHalfState.T}}, i)

Calculate the energy difference for a spin-`1/2` Ising system `ising` if the `i`-th spin were to be flipped.

``ΔHᵢ = 2 sᵢ (∑ⱼ sⱼ + h)``

where the sum is over the nearest neighbors `j` of `i`.
"""
@inline energy_diff(ising::IsingModelExtField{<:AbstractFiniteState{SpinHalfState.T}}, i) = 2 * Integer(ising[i]) * (nearest_neighbors_sum(ising, i) + ising.h)

"""
    AbstractBlumeCapelModel{T} <: AbstractSpinModel{T}

Super type for all Blume-Capel models.
"""
abstract type AbstractBlumeCapelModel{T} <: AbstractSpinModel{T} end

@doc raw"""
    BlumeCapelModel{T} <: AbstractSpinModel{T}

Blume-Capel model without external mangnetic field.

``H = - ∑_⟨i,j⟩ sᵢsⱼ + D ∑ᵢ sᵢ²``
"""
struct BlumeCapelModel{T} <: AbstractBlumeCapelModel{T}

    "State of the spins"
    state::T

    "Parameter"
    D::Real

    """
        BlumeCapelModel(state::T, D::Real) where {T}

    Construct an Blume-Capel system without external magnetic field and with given initial spins state `spins` and parameter `D`.
    """
    BlumeCapelModel(state::T, D::Real) where {T} = new{T}(state, D)

end

@doc raw"""
    energy(blumecapel::BlumeCapelModel)

Total energy of an Blume-Capel system `blumecapel`.

Given by the Hamiltonian:

``H = - ∑_⟨i,j⟩ sᵢsⱼ + D ∑ᵢ sᵢ²``

where `⟨i,j⟩` means that `i` and `j` are nearest neighbors.
"""
@inline energy(blumecapel::BlumeCapelModel) = energy_interaction(blumecapel) + blumecapel.D * sum(sᵢ -> sᵢ^2, blumecapel.state)

@doc raw"""
    energy_local(blumecapel::BlumeCapelModel, i)

Local energy of the `i`-th site in the Blume-Capel system `blumecapel`.

``hᵢ = - sᵢ ∑ⱼ sⱼ + D sᵢ²``

where the sum is over the nearest neighbors `j` of `i`.
"""
@inline energy_local(blumecapel::BlumeCapelModel, i) =
    let sᵢ = Integer(blumecapel[i])
        blumecapel.D * sᵢ^2 - sᵢ * nearest_neighbors_sum(blumecapel.state, i)
    end

@doc raw"""
    energy_local(blumecapel::BlumeCapelModel, i, sᵢ)

Local energy of the `i`-th site assuming its state is `sᵢ` in the Blume-Capel system `blumecapel`.

``hᵢ(sᵢ) = - sᵢ ∑ⱼ sⱼ + D sᵢ²``

where the sum is over the nearest neighbors `j` of `i`.
"""
@inline energy_local(blumecapel::BlumeCapelModel, i, sᵢ) =
    let sᵢ = Integer(sᵢ)
        blumecapel.D * sᵢ^2 - sᵢ * nearest_neighbors_sum(blumecapel.state, i)
    end

@doc raw"""
    minus_energy_local(blumecapel::BlumeCapelModel, i, sᵢ)

*Minus* the value of the local energy of the `i`-th site assuming its state is `sᵢ` in the Blume-Capel system `blumecapel`.

``-hᵢ(sᵢ) = sᵢ ∑ⱼ sⱼ - D sᵢ²``

where the sum is over the nearest neighbors `j` of `i`.
"""
@inline minus_energy_local(blumecapel::BlumeCapelModel, i, sᵢ) =
    let sᵢ = Integer(sᵢ)
        sᵢ * nearest_neighbors_sum(blumecapel.state, i) - blumecapel.D * sᵢ^2
    end

@doc raw"""
    energy_diff(blumecapel::BlumeCapelModel, i, sᵢ′)

Calculate the energy difference for an Blume-Capel system `blumecapel` if the `i`-th spin were to be changed to `sᵢ′`.

``ΔHᵢ = (sᵢ - sᵢ′) ∑ⱼ sⱼ + D (sᵢ′² - sᵢ²)``

where the sum is over the nearest neighbors `j` of `i`.
"""
@inline energy_diff(blumecapel::BlumeCapelModel, i, sᵢ′) =
    let sᵢ = Integer(blumecapel[i]), sᵢ′ = Integer(sᵢ′)
        if sᵢ′ == sᵢ
            0
        else
            (sᵢ - sᵢ′) * nearest_neighbors_sum(blumecapel, i) + blumecapel.D * (sᵢ′^2 - sᵢ^2)
        end
    end

end
