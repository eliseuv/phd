### A Pluto.jl notebook ###
# v0.19.14

using Markdown
using InteractiveUtils

# ╔═╡ 910c49c4-25cc-40ac-b62c-46ab7a1aefd0
begin
    # Select project
    import Pkg
    Pkg.activate(Base.current_project())
    # Libraries
    using PlutoUI, Random, Distributions, LinearAlgebra, Gadfly
    # Custom modules
    include("/home/evf/.julia/pluto_notebooks/ingredients.jl")
    Thesis = ingredients("../src/Thesis.jl").Thesis
    import .Thesis.CorrelatedPairs
    import .Thesis.RandomMatrices
end

# ╔═╡ a90b8d5c-4601-4134-9871-5d952aebe5ec
begin
	# Distribution from which the correlation values are drawn
	
	# T << T_c
	const ρ_dist = 2*Bernoulli(0.5) - 1
	
	# T ~ T_c
	const ρ_dist = Uniform(-1, +1)
	
	# T >> T_c
	const ρ_dist = TruncatedNormal(0, 0.2, -1, +1)

end

# ╔═╡ 3702c170-454a-4707-84d7-43d02befeabc
begin
	# Lenght of time series
	const t_max = 300
	# Number of pairs of correlated time series
	const n_pairs = 50
	# Number of matrix samples
	const n_samples = 256
end

# ╔═╡ 2daa63c8-f4ad-4fe4-9ea7-5fc31fd71708
M_samples = map(eachcol(map(ρ -> rand(CorrelatedPairs.CorrelatedTimeSeriesMatrixSampler(ρ, t_max, 1)), rand(ρ_dist, n_pairs, n_samples)))) do col
	reduce(hcat, col)
end

# ╔═╡ 9cc312b0-2534-4b31-a3bd-cb18c4e0c8c8
λs = mapreduce(eigvals ∘ RandomMatrices.cross_correlation_matrix ∘ RandomMatrices.normalize_ts_matrix!,
	vcat,
	M_samples)

# ╔═╡ dcbb193e-875d-4a6a-a627-7a0293975794
plot(x=λs,
	Geom.histogram(bincount=128, density=true),
	Guide.Title("Histogram of eigenvalues"), Guide.xlabel("λ"))

# ╔═╡ Cell order:
# ╟─910c49c4-25cc-40ac-b62c-46ab7a1aefd0
# ╠═a90b8d5c-4601-4134-9871-5d952aebe5ec
# ╠═3702c170-454a-4707-84d7-43d02befeabc
# ╟─2daa63c8-f4ad-4fe4-9ea7-5fc31fd71708
# ╟─9cc312b0-2534-4b31-a3bd-cb18c4e0c8c8
# ╠═dcbb193e-875d-4a6a-a627-7a0293975794
