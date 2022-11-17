### A Pluto.jl notebook ###
# v0.19.14

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 60ac48d6-5bd4-11ed-3996-0deba86d5f1f
begin
	# Select project
	import Pkg
	Pkg.activate(Base.current_project())
	# Libraries
	using PlutoUI, JLD2, CairoMakie
	# Custom modules
	include("/home/evf/.julia/pluto_notebooks/ingredients.jl")
	Thesis = ingredients("../src/Thesis.jl").Thesis
	import .Thesis.DataIO
end

# ╔═╡ 706cb414-a41b-4460-b62d-8183a5e472d6
begin
	datadir = "../data/sims/random_matrices/corr_ts"
	filenames = DataIO.find_datafiles_with_params(datadir, "CorrTSEigvals")
	params_sets = Dict{String,Set}()
	for filename ∈ filenames
		for (key, value) ∈ DataIO.parse_filename(filename)
			
		
	end
	datafiles = map(filename -> joinpath(datadir, filename), filenames)
end

# ╔═╡ 4674c299-ef79-4894-80f2-47e6967b9f73
begin
	rho_slider = @bind ρ PlutoUI.Slider(-1:0.1:1, default=0, show_value=true)
	t_max_slider = @bind t_max PlutoUI.Slider(2 .^ (1:9), show_value=true)
	n_pairs_slider = @bind n_pairs PlutoUI.Slider(2 .^ (0:8), show_value=true)
	n_samples_slider = @bind n_samples PlutoUI.Slider(2 .^ (1:10), show_value=true)

	md"""
	``\rho`` $(rho_slider)

	``t_{max}`` $(t_max_slider)

	``n_{pairs}`` $(n_pairs_slider)

	``n_{samples}`` $(n_samples_slider)
	"""
end

# ╔═╡ 2ed14f0b-be40-43f2-8c07-20a33d5d98fc
begin
	const spl = CorrelatedPairs.CorrelatedTimeSeriesMatrixSampler(ρ, t_max, n_pairs)
	const λs = reduce(vcat,
    	map(eigvals
			∘ RandomMatrices.cross_correlation_matrix
			∘ RandomMatrices.normalize_ts_matrix!,
        	rand(spl, n_samples)))
end

# ╔═╡ 584546b9-15c0-49f6-91af-5aeb6eb17c79
hist(λs, bins=64, normalization=:pdf;
	 axis = (; title="Correlated time series matrix ρ = $ρ"))

# ╔═╡ Cell order:
# ╠═60ac48d6-5bd4-11ed-3996-0deba86d5f1f
# ╠═706cb414-a41b-4460-b62d-8183a5e472d6
# ╟─4674c299-ef79-4894-80f2-47e6967b9f73
# ╠═2ed14f0b-be40-43f2-8c07-20a33d5d98fc
# ╠═584546b9-15c0-49f6-91af-5aeb6eb17c79
