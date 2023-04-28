### A Pluto.jl notebook ###
# v0.19.24

using Markdown
using InteractiveUtils

# ╔═╡ f720108b-f5f5-474c-922a-ce8711b0b7f2
using Statistics, Random, StatsBase, LinearAlgebra, BenchmarkTools

# ╔═╡ 8c12d3d3-3098-4fea-90ac-0e8aadc126a3
md"""
# Original Fortran code
"""

# ╔═╡ c2ee6ed6-fdd9-488a-a8fa-c52d2df35dda
@inline function hist_mean_fortran(vals::AbstractVector{<:Real}, nbins::Integer)
    n = length(vals)
    (low, high) = extrema(vals)
    bin_width = (high - low) / nbins
    freqs = zeros(Float64, nbins)
    for x ∈ vals
        i = min(floor(Int64, (x - low) / bin_width) + 1, nbins)
        freqs[i] += 1.0 / n
    end

    aver = 0.0
    for i ∈ 1:nbins
        aver += ((i - 1) * bin_width + low) * (freqs[i] / bin_width)
    end
    znorm = sum(freqs) / bin_width
    return aver / znorm
end

# ╔═╡ c9af0bd5-efb0-46ff-85a9-d24b41397cd6
md"""
# StatsBase empirical estimations
"""

# ╔═╡ 670edada-3b07-4e6d-a02c-fed745ece9aa
@inline function make_hist_statsbase(values::AbstractVector{<:Real}, nbins::Integer)
	hist = normalize(fit(Histogram, values, range(extrema(values)..., length=nbins+1)), mode=:probability)
	return (hist.edges[begin], hist.weights)
end

# ╔═╡ 26b21c8e-1b52-447b-809e-e9bcf2e4ce12
@inline function hist_mean_statsbase(values::AbstractVector{<:Real}, nbins::Integer)
	edges, weights = make_hist_statsbase(values, nbins)
	return sum(e * w for (e, w) ∈ zip(edges, weights))
end

# ╔═╡ 2c59d5e6-3791-476f-a484-5e8750c97946
md"""
# My implementation
"""

# ╔═╡ f9047abb-583c-4daa-a3f3-dd0a689d88db
begin
	
	struct MyHistogram{T<:Real}
		edges::AbstractRange{T}
		freqs::AbstractVector{UInt64}
	end
	
	@inline function MyHistogram(values::AbstractVector{T}, nbins::Integer) where {T<:Real}
		low, high = extrema(values)
		
		edges = range(low, high, length=nbins + 1)
	
		freqs = zeros(UInt64, nbins)
		bin_width = (high - low) / nbins
		for x ∈ values
		    idx = min(floor(UInt64, (x - low) / bin_width) + 1, nbins)
		    freqs[idx] += 1
		end
		
		
		MyHistogram{T}(edges, freqs)
		
	end
	
end

# ╔═╡ 19c5e9f0-b1c7-42fe-81d1-1b801800ff26
@inline Statistics.mean(hist::MyHistogram) = let n = sum(hist.freqs)
	sum(e * (f / n) for (e, f) ∈ zip(hist.edges, hist.freqs))
end

# ╔═╡ fef69c65-5e0f-49fd-a98a-007591f726fe
md"""
# Comparing values
"""

# ╔═╡ 1f526ac6-356d-4012-b567-cb6e5be33eeb
begin
	values = randn(1000000)
	nbins = 100
end

# ╔═╡ 41c189e8-f201-4cdb-a800-3cb6edc4f7e8
begin
	@show mean(values)
	@show hist_mean_fortran(values, nbins)
	@show mean(MyHistogram(values, nbins))
	@show hist_mean_statsbase(values, nbins)
end

# ╔═╡ 4f18d1e3-3dab-4252-ad6e-4ee2f46cb73e
md"""
# Benchmarking
"""

# ╔═╡ 14216b09-0512-491c-9ed8-f9577679752f
begin
	BenchmarkTools.DEFAULT_PARAMETERS.samples = 100000
	BenchmarkTools.DEFAULT_PARAMETERS.seconds = 10
	BenchmarkTools.DEFAULT_PARAMETERS.overhead = BenchmarkTools.estimate_overhead()
end

# ╔═╡ a4d2b386-e184-11ed-07c7-470f57650082
@inline function make_hist(values::AbstractVector, nbins::Integer)
    (low, high) = extrema(values)
    bin_width = (high - low) / nbins

    edges = [low + (k*bin_width) for k ∈ 0:nbins]

    freqs = zeros(Float64, nbins)
    for x ∈ values
        i = min(floor(Int64, (x - low) / bin_width) + 1, nbins)
        freqs[i] += 1.0
    end

	foreach(freqs) do x
		x / length(values)
	end

	return edges, freqs

end

# ╔═╡ 50d60965-723d-4152-9628-6313e1d9c74a
@btime make_hist($values, $nbins)

# ╔═╡ 5ab86e18-d3ee-4d5e-810a-23e8ef7952ce
@inline function make_hist_candidate(values::AbstractVector, nbins::Integer)
    (low, high) = extrema(values)
    bin_width = (high - low) / nbins

    edges = [low + (k*bin_width) for k ∈ 0:nbins]

    freqs = zeros(Float64, nbins+1)
    for x ∈ values
        i = floor(Int64, (x - low) / bin_width) + 1
        freqs[i] += 1.0
    end

	freqs[end-1] += pop!(freqs)

	foreach(freqs) do x
		x / length(values)
	end

	return edges, freqs

end

# ╔═╡ 3150edf9-1e57-41ff-8625-8b26c1494d2f
@btime make_hist_candidate($values, $nbins)

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"

[compat]
BenchmarkTools = "~1.3.2"
StatsBase = "~0.33.21"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.8.5"
manifest_format = "2.0"
project_hash = "edd9a249fa37a11a3b988cdfb26faad052cd9982"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.BenchmarkTools]]
deps = ["JSON", "Logging", "Printf", "Profile", "Statistics", "UUIDs"]
git-tree-sha1 = "d9a9701b899b30332bbcb3e1679c41cce81fb0e8"
uuid = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
version = "1.3.2"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "c6d890a52d2c4d55d326439580c3b8d0875a77d9"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.15.7"

[[deps.ChangesOfVariables]]
deps = ["ChainRulesCore", "LinearAlgebra", "Test"]
git-tree-sha1 = "485193efd2176b88e6622a39a246f8c5b600e74e"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.6"

[[deps.Compat]]
deps = ["Dates", "LinearAlgebra", "UUIDs"]
git-tree-sha1 = "7a60c856b9fa189eb34f5f8a6f6b5529b7942957"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.6.1"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.0.1+0"

[[deps.DataAPI]]
git-tree-sha1 = "e8119c1a33d267e16108be441a287a6981ba1630"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.14.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "d1fff3a548102f48987a52a2e0d114fa97d730f0"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.13"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "2fb1e02f2b635d0845df5d7c167fec4dd739b00d"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.3"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "49510dfcb407e572524ba94aeae2fced1f3feb0f"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.8"

[[deps.IrrationalConstants]]
git-tree-sha1 = "630b497eafcc20001bba38a4651b327dcfc491d2"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.2"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.LinearAlgebra]]
deps = ["Libdl", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LogExpFunctions]]
deps = ["ChainRulesCore", "ChangesOfVariables", "DocStringExtensions", "InverseFunctions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "0a1b7c2863e44523180fdb3146534e265a91870b"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.23"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "f66bdc5de519e8f8ae43bdc598782d35a25b1272"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.1.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.20+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "d321bf2de576bf25ec4d3e4360faca399afca282"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.6.0"

[[deps.Parsers]]
deps = ["Dates", "SnoopPrecompile"]
git-tree-sha1 = "478ac6c952fddd4399e71d4779797c538d0ff2bf"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.5.8"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "47e5f437cc0e7ef2ce8406ce1e7e24d44915f88d"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.3.0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.Profile]]
deps = ["Printf"]
uuid = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.SnoopPrecompile]]
deps = ["Preferences"]
git-tree-sha1 = "e760a70afdcd461cf01a575947738d359234665c"
uuid = "66db9d55-30c0-4569-8b51-7e840670fc0c"
version = "1.0.3"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "a4ada03f999bd01b3a25dcaa30b2d929fe537e00"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.1.0"

[[deps.SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "45a7769a04a3cf80da1c1c7c60caf932e6f4c9f7"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.6.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "d1bf48bfcc554a3761a133fe3a9bb01488e06916"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.21"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.0"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.1.1+0"
"""

# ╔═╡ Cell order:
# ╠═f720108b-f5f5-474c-922a-ce8711b0b7f2
# ╟─8c12d3d3-3098-4fea-90ac-0e8aadc126a3
# ╟─c2ee6ed6-fdd9-488a-a8fa-c52d2df35dda
# ╟─c9af0bd5-efb0-46ff-85a9-d24b41397cd6
# ╟─670edada-3b07-4e6d-a02c-fed745ece9aa
# ╟─26b21c8e-1b52-447b-809e-e9bcf2e4ce12
# ╟─2c59d5e6-3791-476f-a484-5e8750c97946
# ╠═f9047abb-583c-4daa-a3f3-dd0a689d88db
# ╠═19c5e9f0-b1c7-42fe-81d1-1b801800ff26
# ╟─fef69c65-5e0f-49fd-a98a-007591f726fe
# ╠═1f526ac6-356d-4012-b567-cb6e5be33eeb
# ╠═41c189e8-f201-4cdb-a800-3cb6edc4f7e8
# ╟─4f18d1e3-3dab-4252-ad6e-4ee2f46cb73e
# ╠═14216b09-0512-491c-9ed8-f9577679752f
# ╠═a4d2b386-e184-11ed-07c7-470f57650082
# ╠═50d60965-723d-4152-9628-6313e1d9c74a
# ╠═5ab86e18-d3ee-4d5e-810a-23e8ef7952ce
# ╠═3150edf9-1e57-41ff-8625-8b26c1494d2f
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
