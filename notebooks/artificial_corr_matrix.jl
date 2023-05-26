### A Pluto.jl notebook ###
# v0.19.25

using Markdown
using InteractiveUtils

# ╔═╡ 9ff9dde0-fc16-11ed-31d2-2b5fa0a56e7d
@inline rand_ts_matrix(n_samples::Integer, t_max::Integer) = hcat(map(s -> map(x -> s*x, 0:(t_max-1)), rand([-1, +1], n_samples))...)

# ╔═╡ 9dc1dcd6-1aa7-4bfe-85d8-9936526281ab
rand_ts_matrix(300, 100)

# ╔═╡ Cell order:
# ╠═9ff9dde0-fc16-11ed-31d2-2b5fa0a56e7d
# ╠═9dc1dcd6-1aa7-4bfe-85d8-9936526281ab
