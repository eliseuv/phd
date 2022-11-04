### A Pluto.jl notebook ###
# v0.19.14

using Markdown
using InteractiveUtils

# ╔═╡ 60ac48d6-5bd4-11ed-3996-0deba86d5f1f
begin
	import Pkg
	Pkg.activate(Base.current_project())
	using Random, LinearAlgebra, PlutoUI, CairoMakie
end

# ╔═╡ 63ab397d-cff5-42fc-9468-b926b5c9005e
begin
	include("/home/evf/.julia/pluto_notebooks/ingredients.jl")
	Thesis = ingredients("../src/Thesis.jl").Thesis
	import .Thesis.CorrelatedPairs
	import .Thesis.RandomMatrices
end

# ╔═╡ eaca5296-6844-4317-8f90-a78af6c63520
begin
	const ρ = 0.9
	const t_max = 128
	const n_pairs = 32
	const n_samples = 512
end

# ╔═╡ cc888bde-cbc4-4540-9bff-f16efd0f45db
const spl = CorrelatedPairs.CorrelatedTimeSeriesMatrixSampler(ρ, t_max, n_pairs)

# ╔═╡ 2ed14f0b-be40-43f2-8c07-20a33d5d98fc
λs = reduce(vcat,
    map(eigvals
		∘ RandomMatrices.cross_correlation_matrix
		∘ RandomMatrices.normalize_ts_matrix!,
        rand(spl, n_samples))) |> sort!

# ╔═╡ 584546b9-15c0-49f6-91af-5aeb6eb17c79
hist(λs, bins=128, normalization=:pdf;
	 axis = (; title="Correlated time series matrix ρ = $ρ", xscale=Makie.pseudolog10, yscale=Makie.pseudolog10))

# ╔═╡ Cell order:
# ╠═60ac48d6-5bd4-11ed-3996-0deba86d5f1f
# ╠═63ab397d-cff5-42fc-9468-b926b5c9005e
# ╠═eaca5296-6844-4317-8f90-a78af6c63520
# ╠═cc888bde-cbc4-4540-9bff-f16efd0f45db
# ╠═2ed14f0b-be40-43f2-8c07-20a33d5d98fc
# ╠═584546b9-15c0-49f6-91af-5aeb6eb17c79
