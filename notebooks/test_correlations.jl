### A Pluto.jl notebook ###
# v0.19.25

using Markdown
using InteractiveUtils

# ╔═╡ 95687a90-f427-11ed-1b0c-9fda4b0aaa6c
begin
	# Select project
	import Pkg
	Pkg.activate(Base.current_project())
	# Libraries
	using PlutoUI, Statistics, StatsBase, CairoMakie
	# Custom modules
	include("/home/evf/.julia/pluto_notebooks/ingredients.jl")
	Thesis = ingredients("../src/Thesis.jl").Thesis
	import .Thesis.DataIO
end

# ╔═╡ d2cc1ca1-61ba-45af-ad2c-7fcd39a4cd48
@inline function corr_product(x::AbstractArray{<:Real}, y::AbstractArray{<:Real})::Real
	#@assert size(x) == size(y)
	N = length(x)
	return sum(x[i]*y[j] for i ∈ 1:N for j ∈ (i+1):N) / (N^2)
end

# ╔═╡ d03520d2-bd42-475c-ac8c-26e9e657f7f2
@inline function self_corr_product(x::AbstractArray{<:Real})::Real
	N = length(x)
	return sum(x[i]*x[j] for i ∈ 1:N for j ∈ (i+1):N) / (N^2)
end

# ╔═╡ 1b83a70c-7dd8-46a1-b752-2d0c019e2539
@inline corr_prod_sample(shape::NTuple{N, T}) where {N, T<:Integer} = corr_product(rand(-1:1, shape), rand(-1:1, shape))

# ╔═╡ 7b9e5aaa-a3ef-48d2-ace1-640ac8215eff
begin
	shape = (128, 128)
	n_samples = 10000
	vals = [corr_prod_sample(shape) for _ ∈ 1:n_samples]
end

# ╔═╡ 672d1256-5b4a-43fb-9d7f-6f5faae948ae
hist(vals, bins=range(extrema(vals)..., length=64);
	axis = (;yscale=Makie.pseudolog10))

# ╔═╡ Cell order:
# ╠═95687a90-f427-11ed-1b0c-9fda4b0aaa6c
# ╠═d2cc1ca1-61ba-45af-ad2c-7fcd39a4cd48
# ╠═d03520d2-bd42-475c-ac8c-26e9e657f7f2
# ╠═1b83a70c-7dd8-46a1-b752-2d0c019e2539
# ╠═7b9e5aaa-a3ef-48d2-ace1-640ac8215eff
# ╠═672d1256-5b4a-43fb-9d7f-6f5faae948ae
