# Dr Watson helper
using DrWatson
@quickactivate "phd"

# External libraries
using Logging, LinearAlgebra, StatsBase, CairoMakie, LaTeXStrings

# My libs
include("../../src/Thesis.jl")
using .Thesis.CorrelatedPairs
using .Thesis.TimeSeries
using .Thesis.DataIO

const ρ_vals = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
# const ρ_vals = [0.9]
const n_steps = 300
const n_pairs = 100
const n_samples = 4000

mean_eigvals = Vector{Float64}()
var_eigvals = Vector{Float64}()
for ρ ∈ ρ_vals
    @show ρ

    # @info "Creating samplers..."
    corr_sampler = CorrelatedPairSampler(ρ)
    corr_mat_sampler = CorrelatedTimeSeriesMatrixSampler(corr_sampler, n_steps, n_pairs)

    # @info "Creating matrices..."
    matrices = [rand(corr_mat_sampler) for _ ∈ 1:n_samples]
    # script_show(matrices)

    # @info "Calculating eigenvalues..."
    eigenvalues = vcat(map(M -> eigvals(cross_correlation_matrix(M)), matrices)...)
    # script_show(eigenvalues)

    mean_eigenvalue = mean(eigenvalues)
    push!(mean_eigvals, mean_eigenvalue)
    var_eigenvalue = var(eigenvalues)
    push!(var_eigvals, var_eigenvalue)
    @show mean_eigenvalue var_eigenvalue

    fig = Figure()
    ax = Axis(fig[1, 1],
        title=L"Eigenvalues histogram ($\rho = %$(ρ)$)",
        xlabel=L"$\lambda$", ylabel=L"$\sigma(\lambda)$",
        yscale=Makie.pseudolog10
    )
    hist!(ax, eigenvalues, bins=100)
    save(joinpath(plotsdir("fitas_corr"), filename("FitasCorrHist", "rho" => ρ; ext="png")), fig)

end

fig = Figure()
ax = Axis(fig[1, 1],
    title="Mean eigenvalue",
    xlabel=L"$\rho$", ylabel=L"$\langle \lambda \rangle$",
    limits=((0, nothing), (0, nothing))
)
scatter!(ax, ρ_vals, mean_eigvals)
save(joinpath(plotsdir("fitas_corr"), filename("FitasCorrMean"; ext="png")), fig)

fig = Figure()
ax = Axis(fig[1, 1],
    title="Eigenvalues variance",
    xlabel=L"$\rho$", ylabel=L"$\langle \lambda^2 \rangle - \langle \lambda \rangle^2$"
)
scatter!(ax, ρ_vals, var_eigvals)
save(joinpath(plotsdir("fitas_corr"), filename("FitasCorrVar"; ext="png")), fig)
