@doc raw"""
    Spin Models

"""
module SpinModels

export
    # Single spin states
    SingleSpinState, SpinHalfState, SpinOneState,
    # Properties of single spin states
    new_rand_spin,
    # Measurements in spin states
    magnet_total, magnet,
    # Measurements on spins states that depend on locality
    energy_interaction,
    # Symmetries explored
    flip!, flip_state_index!,
    # Spin models
    AbstractSpinModel,
    # General properties of spin models
    state, state_type, spin_type, spin_instances, state,
    # Metropolis sampling
    metropolis!,
    metropolis_measure!,
    metropolis_measure_energy!,
    # Heatbath sampling
    heatbath!,
    heatbath_measure!,
    # Energy measurements
    energy,
    minus_energy_local,
    minus_energy_diff,
    # Ising models
    AbstractIsingModel,
    # Implementations of Ising models
    IsingModel, IsingModelExtField,
    # Blume-Capel models
    AbstractBlumeCapelModel,
    # Implementations of Blume-Capel models
    BlumeCapelIsotropicModel, BlumeCapelModel

using Random, EnumX, Combinatorics, StatsBase, Distributions, Graphs

using ..FiniteStates

"""
###################
    Spin-1/2 State
###################
"""

"""
    SpinHalfState::Int8 <: SingleSpinState

Enumeration of possible spin `1/2` values.

Attention: The numerical value associated with down (up) spin value is -1 (+1) and not -1/2 (+1/2).
"""
@enumx SpinHalfState::Int8 begin
    down = -1
    up = +1
end

"""
    other_spin(σ::SpinHalfState.T)

Returns the complementary of the spin-`1/2` state `σ`:
    - up   => down
    - down => up
"""
@inline other_spin(σ::SpinHalfState.T) = SpinHalfState.T(-Integer(σ))

"""
    flip!(fs::AbstractFiniteState{SpinHalfState.T}, i)

Flips the `i`-th spin in the spin-`1/2` state `fs`.
"""
@inline function flip!(fs::AbstractFiniteState{SpinHalfState.T}, i)
    @inbounds fs[i] = other_spin(fs[i])
end

@doc raw"""
    flip_state_index!(fs::MeanFieldSpinState{SpinHalfState.T}, k::Integer)

Flip the some spin of the `k`-th (`k ∈ {1,2}`) state in the spin-`1/2` state with mean field interaction `fs`.
"""
@inline function flip_state_index!(fs::MeanFieldFiniteState{SpinHalfState.T}, k::Integer)
    fs.counts[k] -= 1
    fs.counts[mod1(k + 1, 2)] += 1
end

"""
#################
    Spin-1 State
#################
"""

"""
    SpinOneState::Int8 <: SingleSpinState

Enumeration of possible spin-`1` values.
"""
@enumx SpinOneState::Int8 begin
    down = -1
    zero = 0
    up = +1
end

"""
######################
    Single Spin State
######################
"""

"""
    SingleSpinState

Supertype for all single spin states.

They are usually Enums, but even if they are not, all single spin states must provide a method `instances(<:SingleSpinState)`
that returns a tuple with all possible single spin states.
"""
SingleSpinState = Union{SpinOneState.T,SpinHalfState.T}

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

# Arithmetic of spin states
for op in (:+, :-, :*, :/)
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
    spin_char = σ == SpinOneState.up ? '↑' : σ == SpinOneState.down ? '↓' : '-'
    print(io, spin_char)
end

"""
    new_rand_spin(σ::T) where {T<:SingleSpinState}

Select a new random single spin state `σ′ ∈ SingleSpinState` different from `σ`.
"""
@inline new_rand_spin(σ::T) where {T<:SingleSpinState} = rand(filter(!=(σ), instances(T)))

"""
    new_rand_spin(σ::SpinHalfState.T)

Returns the complementary of the single spin-`1/2` state `σ`.
"""
@inline new_rand_spin(σ::SpinHalfState.T) = other_spin(σ)

"""
    new_rand_spin(σ::SpinOneState.T)

Returns a new random single spin-`1` state different from `σ`.
"""
@inline new_rand_spin(σ::SpinOneState.T)::SpinOneState.T = SpinOneState.T(mod(Int(σ) + rand(1:2), -1:1))

"""
###################################
    Magnetization of Finite States
###################################
"""

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

"""
########################################
    Interaction Energy of Finite States
########################################
"""

@doc raw"""
    energy_interaction(fs::AbstractFiniteState)

Interaction energy for a spin state `fs`.

``H_{int} = - ∑_⟨i,j⟩ sᵢ sⱼ``

where `⟨i,j⟩` represents a pair of nearest neighbors sites.
"""
@inline energy_interaction(fs::AbstractFiniteState) = @inbounds -sum(Integer(fs[i]) * Integer(fs[j]) for (i, j) ∈ nearest_neighbors(fs))

@doc raw"""
    energy_interaction(fs::MeanFieldFiniteState{SpinHalfState.T})

Interaction energy of the spin state with mean field interaction `fs`.

``H_{int} = - ∑_⟨i,j⟩ sᵢsⱼ = - \frac{z M^2}{2N}``
"""
@inline energy_interaction(fs::MeanFieldFiniteState) = -(fs.z * magnet_total(fs)^2) / (2 * length(fs))

@doc raw"""
    energy_interaction(fs::SquareLatticeFiniteState)

Interaction energy for a `N`-dimensional square lattice spin model `fs`.

``H_{int} = - \sum_⟨i,j⟩ σᵢ σⱼ``
"""
function energy_interaction(fs::SquareLatticeFiniteState{T,N}) where {T,N}
    # Variable to accumulate
    H = zero(Int64)
    # Loop on dimensions
    @inbounds for d ∈ 1:N
        # Bulk
        front_bulk = selectdim(container(fs), d, 1:(size(fs, d)-1))
        back_bulk = selectdim(container(fs), d, 2:size(fs, d))
        H -= sum(Integer, front_bulk .* back_bulk)
        # Periodic boundaries
        last_slice = selectdim(container(fs), d, size(fs, d))
        first_slice = selectdim(container(fs), d, 1)
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
########################
    Abstract Spin Model
########################
"""

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
################################
    Measurements on Spin Models
################################
"""

@doc raw"""
    magnet_total(spinmodel::AbstractSpinModel)

Calculate the total magnetization of the spin model `spinmodel`.

``M = ∑ᵢ sᵢ``
"""
@inline magnet_total(spinmodel::AbstractSpinModel) = magnet_total(state(spinmodel))

@doc raw"""
    magnet(spinmodel::AbstractSpinModel)

Calculate the magnetization per site of the spin model `spinmodel`.

``m = (1/N) ∑ᵢ sᵢ``
"""
@inline magnet(spinmodel::AbstractSpinModel) = magnet(state(spinmodel))

@doc raw"""
    energy_interaction(spinmodel::AbstractSpinModel)

Calculate the interaction energy of the spin model `spinmodel`.
"""
@inline energy_interaction(spinmodel::AbstractSpinModel) = energy_interaction(state(spinmodel))

"""
    flip!(spinmodel::AbstractSpinModel{<:AbstractFiniteState{SpinHalfState.T}}, i)

Flip a given spin in the spin-`1/2` spin model `spinmodel`.
"""
@inline function flip!(spinmodel::AbstractSpinModel{<:AbstractFiniteState{SpinHalfState.T}}, i)
    flip!(state(spinmodel), i)
end

@inline """
    flip_state_index!(spinmodel::AbstractSpinModel{<:MeanFieldFiniteState}, k)

Flip a given state with index `k` in the spin model with mean field interaction `spinmodel`.
"""
function flip_state_index!(spinmodel::AbstractSpinModel{<:MeanFieldFiniteState}, k)
    flip_state_index!(state(spinmodel), k)
end

"""
########################
    Metropolis Sampling
########################
"""

"""
    metropolis!(spinmodel::AbstractSpinModel, β::Real, n_steps::Integer)

Sample using the Metropolis algorithm the spin model `spinmodel` at temperature `β` for `n_steps` steps.
"""
function metropolis!(spinmodel::AbstractSpinModel, β::Real, n_steps::Integer)
    # Loop on random sites
    @inbounds for i ∈ rand(eachindex(state(spinmodel)), n_steps)
        # Select random new state
        sᵢ′ = new_rand_spin(spinmodel[i])
        # Get energy difference
        minus_ΔH = minus_energy_diff(spinmodel, i, sᵢ′)
        # Metropolis prescription
        if minus_ΔH > 0 || exp(β * minus_ΔH) > rand()
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
        minus_ΔH = minus_energy_diff(spinmodel, i)
        # Metropolis prescription
        if minus_ΔH >= 0 || exp(β * minus_ΔH) > rand()
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
    # Loop on steps
    @inbounds for i ∈ rand(eachindex(state(spinmodel)), n_steps)
        # Get state index of random selected spin
        k = if i <= state(spinmodel).counts[begin]
            1
        else
            2
        end
        # Get state
        σ = instances(SpinHalfState.T)[k]
        # Get energy difference
        minus_ΔH = minus_energy_diff(spinmodel, σ)
        # Metropolis prescription
        if minus_ΔH >= 0 || exp(β * minus_ΔH) > rand()
            # Flip spin
            flip_state_index!(spinmodel, k)
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
            sᵢ′ = new_rand_spin(spinmodel[i])
            # Get energy difference
            minus_ΔH = minus_energy_diff(spinmodel, i, sᵢ′)
            # Metropolis prescription
            if minus_ΔH >= 0 || exp(β * minus_ΔH) > rand()
                # Change spin
                spinmodel[i] = sᵢ′
                # Add energy difference
                ΔH_total -= minus_ΔH
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
            minus_ΔH = minus_energy_diff(spinmodel, i)
            # Metropolis prescription
            if minus_ΔH >= 0 || exp(β * minus_ΔH) > rand()
                # Change spin
                flip!(spinmodel, i)
                # Add energy difference
                ΔH_total -= minus_ΔH
            end
        end
        # Update energy measurements vector
        results[t+1] = results[t] + ΔH_total
    end
    return results
end

"""
######################
    Heatbath Sampling
######################
"""

"""
    heatbath!(spinmodel::AbstractSpinModel, β::Real, n_steps::Integer)

Sample the spin model `spinmodel` using the heatbath algorithm at temperature `β` for `n_steps` steps.

The weight associated with the candidate single spin state `σ` at the `i`-th site at temperature `β` is:

``w(σ, i, β) = exp(-β hᵢ(σ))``

where `hᵢ(σ)` is the local energy associated with `i`-th site assuming that its state is `σ`.
"""
function heatbath!(spinmodel::AbstractSpinModel{<:AbstractFiniteState{T}}, β::Real, n_steps::Integer) where {T}
    # Allocate arrays
    spin_inst = [instances(T)...]
    weights = Vector{Float64}(undef, length(spin_inst))
    # Site loop
    @inbounds for i ∈ rand(eachindex(state(spinmodel)), n_steps)
        map!(σ -> exp(β * minus_energy_local(spinmodel, i, σ)), weights, spin_inst)
        spinmodel[i] = sample(spin_inst, ProbabilityWeights(weights))
    end
end

function heatbath!(spinmodel::AbstractSpinModel{<:AbstractFiniteState{SpinOneState.T}}, β::Real, n_steps::Integer)
    @inbounds for i ∈ rand(eachindex(state(spinmodel)), n_steps)
        # Calculate weights
        W_down = exp(β * minus_energy_local(spinmodel, i, SpinOneState.down))
        # W_zero = exp(-β * 0.0) =  1.0
        W_up = exp(β * minus_energy_local(spinmodel, i, SpinOneState.up))
        W_total = W_down + 1.0 + W_up
        # Calculate probabilities
        # Threshold(↓) = P(↓) = W(↓) / Wₜ
        thr_down = W_down / W_total
        # Threshold(0) = P(↓) + P(0) = (W(↓) + W(0))/Wₜ = (W(↓) + 1.0)/Wₜ
        thr_zero = (W_down + 1.0) / W_total

        # Heatbath prescription
        # 0               < rand() < P_down                     => (↓)
        # P_down          < rand() < P_down + P_zero            => (0)
        # P_down + P_zero < rand() < P_down + P_zero + P_up = 1 => (↑)
        rnd = rand()
        spinmodel[i] = if rnd < thr_down
            SpinOneState.down
        elseif rnd < thr_zero
            SpinOneState.zero
        else
            SpinOneState.up
        end

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
            # Calculate minus energy local
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
#########################
    Abstract Ising Model
#########################
"""

"""
    AbstractIsingModel{T} <: AbstractSpinModel{T}

Super type for all Ising models.
"""
abstract type AbstractIsingModel{T} <: AbstractSpinModel{T} end

"""
################
    Ising Model
################
"""

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
    IsingModel(state::T) where {T<:AbstractFiniteState} = new{T}(state)
end

@inline name(ising::IsingModel) = "Ising_" * FiniteStates.name(ising.state)

@doc raw"""
    energy(ising::IsingModel)

Total energy of an Ising system `ising`.

Given by the Hamiltonian:

``H = - ∑_⟨i,j⟩ sᵢsⱼ``

where `⟨i,j⟩` means that `i` and `j` are nearest neighbors.
"""
@inline energy(ising::IsingModel) = energy_interaction(ising)

@doc raw"""
    minus_energy_local(ising::IsingModel{<:AbstractFiniteState{T}}, i, sᵢ::T=ising[i]) where {T}

*Minus* the local energy of the `i`-th site assuming its state is `sᵢ` in the Ising system `ising`.

If no state `sᵢ` is provided, it assumes the current state `ising[i]`.

``-hᵢ = sᵢ ∑ⱼ sⱼ``

where the sum is over the nearest neighbors `j` of `i`.
"""
@inline minus_energy_local(ising::IsingModel{<:AbstractFiniteState{T}}, i, sᵢ::T=ising[i]) where {T} = Integer(sᵢ) * nearest_neighbors_sum(ising.state, i)

@doc raw"""
    minus_energy_diff(ising::IsingModel, i, sᵢ′)

Calculate the *minus* energy difference for an Ising system `ising` if the `i`-th spin were to be changed to `sᵢ′`.

``-ΔHᵢ = (sᵢ′ - sᵢ) ∑ⱼ sⱼ``

where the sum is over the nearest neighbors `j` of `i`.
"""
@inline minus_energy_diff(ising::IsingModel{<:AbstractFiniteState{T}}, i, sᵢ′::T) where {T} =
    let sᵢ = ising[i]
        if sᵢ′ == sᵢ
            0
        else
            (Integer(sᵢ′) - Integer(sᵢ)) * nearest_neighbors_sum(ising.state, i)
        end
    end

@doc raw"""
    minus_energy_diff(ising::IsingModel, i)

Calculate *minus* the energy difference for a spin-`1/2` Ising system `ising` if the `i`-th spin were to be flipped.

``-ΔHᵢ = - 2 sᵢ ∑ⱼ sⱼ``

where the sum is over the nearest neighbors `j` of `i`.
"""
@inline minus_energy_diff(ising::IsingModel{<:AbstractFiniteState{SpinHalfState.T}}, i) = -2 * Integer(ising[i]) * nearest_neighbors_sum(ising.state, i)

@doc raw"""
    minus_energy_diff(ising::IsingModel{MeanFieldFiniteState{SpinHalfState.T}}, σ::SpinHalfState.T)

Calculate *minus* the energy difference for a spin-`1/2` mean field Ising system `ising` if a spin with current state `σ` were to be flipped.

``-ΔH(sᵢ) = -2 sᵢ ∑ⱼ sⱼ = -2 sᵢ z (M - sᵢ) / N = 2 z (1 - sᵢ M) / N ``

where the sum is over the nearest neighbors `j` of `i`.
"""
@inline minus_energy_diff(ising::IsingModel{MeanFieldFiniteState{SpinHalfState.T}}, σ::SpinHalfState.T) = 2 * ising.state.z * (1 - Integer(σ) * magnet_total(ising)) / length(ising)

"""
####################################
    Ising Model with External Field
####################################
"""

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
    IsingModelExtField(state::T, h::Real) where {T<:AbstractFiniteState} = new{T}(state, h)
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
    minus_energy_local(ising::IsingModelExtField{<:AbstractFiniteState{T}}, i, sᵢ::T=ising[i]) where {T}

*Minus* the local energy of the `i`-th site assuming its state is `sᵢ` in the Ising system `ising`.

if no state `sᵢ` is provided, it assumes the current site state `ising[i]`.

``-hᵢ = sᵢ (∑ⱼ sⱼ + h)``

where the sum is over the nearest neighbors `j` of `i`.
"""
@inline minus_energy_local(ising::IsingModelExtField{<:AbstractFiniteState{T}}, i, sᵢ::T=ising[i]) where {T} = Integer(sᵢ) * (nearest_neighbors_sum(ising.state, i) + ising.h)

@doc raw"""
    minus_energy_diff(ising::IsingModelExtField, i, sᵢ′)

Calculate *minus* the energy difference for an Ising system `ising` if the `i`-th spin were to be changed to `sᵢ′`.

``-ΔHᵢ = (sᵢ′ - sᵢ) (∑ⱼ sⱼ + h)``

where the sum is over the nearest neighbors `j` of `i`.
"""
@inline minus_energy_diff(ising::IsingModelExtField{<:AbstractFiniteState{T}}, i, sᵢ′::T) where {T} =
    let sᵢ = ising[i]
        (Integer(sᵢ′) - Integer(sᵢ)) * (nearest_neighbors_sum(ising, i) + ising.h)
    end

@doc raw"""
    minus_energy_diff(ising::IsingModelExtField{<:AbstractFiniteState{SpinHalfState.T}}, i)

Calculate the energy difference for a spin-`1/2` Ising system `ising` if the `i`-th spin were to be flipped.

``-ΔHᵢ = -2 sᵢ (∑ⱼ sⱼ + h)``

where the sum is over the nearest neighbors `j` of `i`.
"""
@inline minus_energy_diff(ising::IsingModelExtField{<:AbstractFiniteState{SpinHalfState.T}}, i) = -2 * Integer(ising[i]) * (nearest_neighbors_sum(ising, i) + ising.h)

"""
###############################
    Abstract Blume-Capel Model
###############################
"""

"""
    AbstractBlumeCapelModel{T} <: AbstractSpinModel{T}

Super type for all Blume-Capel models.
"""
abstract type AbstractBlumeCapelModel{T} <: AbstractSpinModel{T} end

"""
################################
    Blume-Capel Isotropic Model
################################
"""

@doc raw"""
    BlumeCapelIsotropicModel{T} <: AbstractBlumeCapelModel{T}

Blume-Capel model without external magnetic field and without the single spin anisotropy parameter.

``H = - ∑_⟨i,j⟩ sᵢsⱼ``

where `⟨i,j⟩` means that `i` and `j` are nearest neighbors.
"""
struct BlumeCapelIsotropicModel{T} <: AbstractBlumeCapelModel{T}

    "State of the spin system"
    state::T

    """
        BlumeCapelIsotropicModel(state::T, D::Real) where {T}

    Construct an Blume-Capel system without external magnetic field and with given initial spins state `spins`.
    """
    BlumeCapelIsotropicModel(state::T) where {T<:AbstractFiniteState{SpinOneState.T}} = new{T}(state)

end

@doc raw"""
    energy(blumecapel::BlumeCapelIsotropicModel)

Total energy of an isotropic Blume-Capel system `blumecapel` which is simply the spin interaction energy.

``H = - ∑_⟨i,j⟩ sᵢsⱼ``

where `⟨i,j⟩` means that `i` and `j` are nearest neighbors.
"""
@inline energy(blumecapel::BlumeCapelIsotropicModel) = energy_interaction(blumecapel.state)

@doc raw"""
    minus_energy_local(blumecapel::BlumeCapelIsotropicModel{<:AbstractFiniteState{T}}, i, sᵢ::T=blumecapel[i]) where {T}

*Minus* the value of the local energy of the `i`-th site assuming its state is `sᵢ` in the isotropic Blume-Capel system `blumecapel`.

If no state `sᵢ` is provided, it assumes the current site state `blumecapel[i]`.

``-hᵢ(sᵢ) = sᵢ ∑ⱼ sⱼ``

where the sum is over the nearest neighbors `j` of `i`.
"""
@inline minus_energy_local(blumecapel::BlumeCapelIsotropicModel{<:AbstractFiniteState{T}}, i, sᵢ::T=blumecapel[i]) where {T} =
    Integer(sᵢ) * nearest_neighbors_sum(blumecapel.state, i)

@doc raw"""
    minus_energy_diff(blumecapel::BlumeCapelIsotropicModel{<:AbstractFiniteState{T}}, i, sᵢ′::T) where {T}

Calculate *minus* the energy difference for an isotropic Blume-Capel system `blumecapel` if the `i`-th spin were to be changed to `sᵢ′`.

``-ΔHᵢ = (sᵢ′ - sᵢ) ∑ⱼ sⱼ``

where the sum is over the nearest neighbors `j` of `i`.
"""
@inline minus_energy_diff(blumecapel::BlumeCapelIsotropicModel{<:AbstractFiniteState{T}}, i, sᵢ′::T) where {T} =
    let sᵢ = Integer(blumecapel[i]), sᵢ′ = Integer(sᵢ′)
        (sᵢ′ - sᵢ) * nearest_neighbors_sum(blumecapel.state, i)
    end

"""
######################
    Blume-Capel Model
######################
"""

@doc raw"""
    BlumeCapelModel{T} <: AbstractBlumeCapelModel{T}

Blume-Capel model without external magnetic field.

``H = - ∑_⟨i,j⟩ sᵢsⱼ + D ∑ᵢ sᵢ²``
"""
struct BlumeCapelModel{T} <: AbstractBlumeCapelModel{T}

    "State of the spins"
    state::T

    "Anisotropy parameter"
    D::Real

    """
        BlumeCapelModel(state::T, D::Real) where {T}

    Construct an Blume-Capel system without external magnetic field and with given initial spins state `spins` and anisotropy parameter `D`.
    """
    BlumeCapelModel(state::T, D::Real) where {T} = new{T}(state, D)

    """
        BlumeCapelModel(state::T, ::Val{D}) where {T,D}

    Construct an Blume-Capel system without external magnetic field and with given initial spins state `spins` and anisotropy parameter `D` known at compile time.
    """
    BlumeCapelModel(state::T, ::Val{D}) where {T,D} = new{T}(state, D)

end

"""
    BlumeCapelModel(state::T, ::Val(0)) where {T}

If the anisotropy parameter is zero, use isotropic model.
"""
@inline BlumeCapelModel(state::T, ::Union{Val{0},Val{0.0}}) where {T} = BlumeCapelIsotropicModel(state)

@doc raw"""
    energy(blumecapel::BlumeCapelModel)

Total energy of an Blume-Capel system `blumecapel`.

Given by the Hamiltonian:

``H = - ∑_⟨i,j⟩ sᵢsⱼ + D ∑ᵢ sᵢ²``

where `⟨i,j⟩` means that `i` and `j` are nearest neighbors.
"""
@inline energy(blumecapel::BlumeCapelModel) = energy_interaction(blumecapel.state) + blumecapel.D * sum(sᵢ -> sᵢ^2, blumecapel.state)

@doc raw"""
    minus_energy_local(blumecapel::BlumeCapelModel{<:AbstractFiniteState{T}}, i, sᵢ::T=blumecapel[i]) where {T}

*Minus* the value of the local energy of the `i`-th site assuming its state is `sᵢ` in the Blume-Capel system `blumecapel`.

If no state `sᵢ` is provided, it assumes the current site state `blumecapel[i]`.

``-hᵢ(sᵢ) = sᵢ ∑ⱼ sⱼ - D sᵢ²``

where the sum is over the nearest neighbors `j` of `i`.
"""
@inline minus_energy_local(blumecapel::BlumeCapelModel{<:AbstractFiniteState{T}}, i, sᵢ::T=blumecapel[i]) where {T} =
    let sᵢ = Integer(sᵢ)
        sᵢ * nearest_neighbors_sum(blumecapel.state, i) - blumecapel.D * sᵢ^2
    end

@doc raw"""
    minus_energy_diff(blumecapel::BlumeCapelModel{<:AbstractFiniteState{T}}, i, sᵢ′::T) where {T}

Calculate *minus* the energy difference for an Blume-Capel system `blumecapel` if the `i`-th spin were to be changed to `sᵢ′`.

``-ΔHᵢ = (sᵢ′ - sᵢ) ∑ⱼ sⱼ + D (sᵢ² - sᵢ′²)``

where the sum is over the nearest neighbors `j` of `i`.
"""
@inline minus_energy_diff(blumecapel::BlumeCapelModel{<:AbstractFiniteState{T}}, i, sᵢ′::T) where {T} =
    let sᵢ = Integer(blumecapel[i]), sᵢ′ = Integer(sᵢ′)
        (sᵢ′ - sᵢ) * nearest_neighbors_sum(blumecapel.state, i) + blumecapel.D * (sᵢ^2 - sᵢ′^2)
    end

end
