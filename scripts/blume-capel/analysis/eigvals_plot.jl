### A Pluto.jl notebook ###
# v0.19.22

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

# ╔═╡ 97e1cec4-b47c-11ed-0f49-59ce159b2c34
begin
    # Select project
    import Pkg
    Pkg.activate(Base.current_project())
	PROJ_ROOT = dirname(Base.current_project())
    # Libraries
    using PlutoUI, DrWatson, CSV, DataFrames, JLD2, StatsBase, LinearAlgebra, LaTeXStrings, CairoMakie
    # Custom modules
    include(joinpath(PROJ_ROOT, "notebooks", "ingredients.jl"))
    Thesis = ingredients(joinpath(PROJ_ROOT, "src", "Thesis.jl")).Thesis
    import .Thesis.DataIO
end


# ╔═╡ 14624983-36e2-4142-ba23-c1013cb1a91e
@quickactivate "phd"

# ╔═╡ 12eefbf8-7d41-4a63-9496-22f45f6b6fd4
make_ticks_log(powers::AbstractVector{<:Real}, base::Integer=10) = (Float64(base) .^ powers, (map(x -> latexstring("$(base)^{$(x)}"), powers)))

# ╔═╡ 7ebbf011-7064-4c75-9b9b-e8a320e08a76
make_ticks_log_(powers::AbstractVector{<:Real}, base::Integer=10) = (powers, (map(x -> latexstring("$(base)^{$(x)}"), powers)))

# ╔═╡ 359e6e36-f476-4a1e-a676-6453ccb9195f
function histogram_mean(vals::AbstractVector{<:Real}, nbins::Integer=128, closed::Symbol=:left)
	hist = fit(Histogram, vals, nbins=nbins, closed=closed)
	x = hist.edges[begin][begin:end-1]
	y = hist.weights
	return (x ⋅ y) / sqrt(y ⋅ y)
end

# ╔═╡ dc10501c-0fe5-43d3-90f1-4173bdd34311
# Load dataframes with the critical temperature data
df_temperatures = DataFrame(CSV.File(joinpath(PROJ_ROOT, "tables", "butera_and_pernici_2018", "blume-capel_square_lattice.csv")))

# ╔═╡ 1713b9bd-8d8c-4b79-8062-3d73d6a4e6d3
begin
	const data_dirpath = datadir("sims", "blume-capel", "square_lattice")
	const prefix = "BlumeCapelSqLatticeCorrMatEigvals"

	const params_req = Dict(
	    "dim" => 2,
	    "L" => 64,
	    "n_runs" => 1024,
	    "n_samples" => 128,
	    "n_steps" => 512
	)
	
	const datafiles = sort(DataIO.find_datafiles(data_dirpath, prefix, params_req), by=x -> (x.params["D"], x.params["beta"]))

	const D_values = unique(map(x -> x.params["D"], datafiles))

end

# ╔═╡ 4089cee9-8cc1-4143-9fbc-92e2261ce189
@bind D_idx PlutoUI.Slider(1:length(D_values))

# ╔═╡ e1b9515e-1e2b-47d2-83fa-7c1bc893e033
begin

	# Get value for anisotropy
	const D = D_values[D_idx]
	
	# Get datafiles associated with chosen anisotropy
	const datafiles_D = filter(x -> x.params["D"] == D, datafiles)

	# Get beta values available
	const beta_values = map(x -> x.params["beta"], datafiles_D)

	# Critical temperature for such systems
	const df_D_row = df_temperatures[only(findall(==(D), df_temperatures.anisotropy_field)), 2:end]
	const trans_order = replace(string(df_D_row[:transition_order]), "Second" => "Second order")
	const crit_temp_source = findfirst(!ismissing, df_D_row)
	const T_c = df_D_row[crit_temp_source]

	const crit_temp_source_str = replace(string(crit_temp_source), "_" => " ")
	
	md"""
	 $D =$ $(D)
	
	 $T_c =$ $(T_c) \[ $(trans_order) ( $(crit_temp_source_str) ) \]
	"""
end

# ╔═╡ 4ae757a6-7b12-41e5-b47d-9d47c813470b
begin
	
	average_eigval(datafile) = mean(map(histogram_mean, load(datafile.path, "eigvals")))

	scatter(1.0 ./ (beta_values .* T_c), map(average_eigval, datafiles_D),
		    axis=(; title=L"Eigenvalues average $D = %$(D)$ [ %$(trans_order) ( %$(crit_temp_source_str) ) ]",
	 	 	        xlabel=L"\tau",
	 		 	    ylabel=L"\langle\lambda\rangle",
	  	     		#limits=((0, nothing), (0, nothing)),
	         		#yticks=make_ticks_log(0:5),
			 		#yscale=Makie.pseudolog10
	 ))
end

# ╔═╡ 460c15c6-17db-4e3a-a8b6-66b5f8c2d1e7


# ╔═╡ 7d9f981e-7130-4db8-8865-dd6388445296
@bind beta_idx PlutoUI.Slider(1:length(beta_values))

# ╔═╡ becf0b50-6428-4b56-b5b4-ab4c4d2e360a
begin

	# Get beta value
	const beta = beta_values[beta_idx]

	# Get datafile
	const datafile = datafiles_D[beta_idx]

	# Calculate tau
	const tau = round(1.0 / (T_c * beta), digits=5)
	
	md"""
	 $\beta =$ $(beta)
	
	 $\tau =$ $(tau)
	"""
end

# ╔═╡ fdad1390-fab2-4103-a213-fd244e47faf7
begin
	# Load data from file
	(corr_vals, eigvals) = load(datafile.path, "corr_vals", "eigvals")

	# Sort eigenvalues
	foreach(sort!, eigvals)

	datafile.params
end

# ╔═╡ 4b7c96b7-f847-4f88-97d4-4e6afbfe3a72
function make_histogram_log(values::AbstractVector{<:Real}, nbins::Integer)
	hist = normalize(fit(Histogram, values; closed=:left, nbins=nbins), mode=:pdf)
	edges = collect(hist.edges[begin])
	x = edges[begin:end-1] .+ (diff(edges) ./ 2)
	y = hist.weights
	coords = filter(x -> x[2] > 0, collect(zip(x, y)))
	return (getindex.(coords, 1), log10.(getindex.(coords, 2)))
end

# ╔═╡ f7760564-f389-452d-8f4b-945b53cdd42d
hist(vcat(eigvals...), bins=100;
     axis=(; title=L"Eigenvalues distribution $D = %$(D)$, $\tau = %$(tau)$",
	 	 	 xlabel=L"\lambda",
	 		 #ylabel=L"\rho(\lambda)",
	  	     limits=((0, nothing), (0, nothing)),
	         yticks=make_ticks_log(0:5),
			 yscale=Makie.pseudolog10
	 ))

# ╔═╡ 6b2f7168-fdbf-4d3e-9fe7-e07637849385
hist(vcat(map(diff, eigvals)...), bins=100;
     axis=(; title=L"Eigenvalues spacing distribution $D = %$(D)$, $\tau = %$(tau)$",
	 	 	 xlabel=L"\Delta\lambda",
	 		 #ylabel=L"\rho(\Delta\lambda)",
	  	     limits=((0, nothing), (0, nothing)),
	         yticks=make_ticks_log(0:5),
			 yscale=Makie.pseudolog10))

# ╔═╡ da380746-1003-4108-8807-030dc463fb10
# ╠═╡ disabled = true
#=╠═╡
hist(map(last, eigvals), bins=32;
     axis=(; title=L"Maximum $D = %$(D)$, $\tau = %$(tau)$",
	 	 	 xlabel=L"\lambda_{max}",
	 		 #ylabel=L"\rho(\Delta\lambda)",
	  	     limits=(nothing, (0, nothing)),
	         yticks=make_ticks_log(0:5),
			 yscale=Makie.pseudolog10))
  ╠═╡ =#

# ╔═╡ 3e82e093-4d9f-440e-b28c-e5e9a990ad71
# ╠═╡ disabled = true
#=╠═╡
hist(map(first, eigvals), bins=32;
     axis=(; title=L"Minimum $D = %$(D)$, $\tau = %$(tau)$",
	 	 	 xlabel=L"\lambda_{min}",
	 		 #ylabel=L"\rho(\Delta\lambda)",
	  	     limits=(nothing, (0, nothing)),
	         yticks=make_ticks_log(0:5),
			 yscale=Makie.pseudolog10))
  ╠═╡ =#

# ╔═╡ 78ec2dea-5564-4099-8e12-431b62358b96
# ╠═╡ disabled = true
#=╠═╡
begin
	(x, y) = make_histogram_log(vcat(map(diff, eigvals)...), 100)
	low = minimum(y)*1.1
	barplot(x, y;
	        fillto=low, gap=0,
			axis=(; title=L"Eigenvalues gaps $D = %$(D)$, $\tau = %$(tau)$",
	 	 			xlabel=L"\Delta\lambda", ylabel=L"\rho(\Delta\lambda)",
			
	        yticks=make_ticks_log_(-5:0),
			limits=(nothing, (low, nothing))
			))
end
  ╠═╡ =#

# ╔═╡ Cell order:
# ╟─97e1cec4-b47c-11ed-0f49-59ce159b2c34
# ╟─14624983-36e2-4142-ba23-c1013cb1a91e
# ╟─12eefbf8-7d41-4a63-9496-22f45f6b6fd4
# ╟─7ebbf011-7064-4c75-9b9b-e8a320e08a76
# ╟─359e6e36-f476-4a1e-a676-6453ccb9195f
# ╟─dc10501c-0fe5-43d3-90f1-4173bdd34311
# ╟─1713b9bd-8d8c-4b79-8062-3d73d6a4e6d3
# ╟─4089cee9-8cc1-4143-9fbc-92e2261ce189
# ╟─e1b9515e-1e2b-47d2-83fa-7c1bc893e033
# ╠═4ae757a6-7b12-41e5-b47d-9d47c813470b
# ╠═460c15c6-17db-4e3a-a8b6-66b5f8c2d1e7
# ╟─7d9f981e-7130-4db8-8865-dd6388445296
# ╟─becf0b50-6428-4b56-b5b4-ab4c4d2e360a
# ╟─fdad1390-fab2-4103-a213-fd244e47faf7
# ╟─4b7c96b7-f847-4f88-97d4-4e6afbfe3a72
# ╠═f7760564-f389-452d-8f4b-945b53cdd42d
# ╟─6b2f7168-fdbf-4d3e-9fe7-e07637849385
# ╟─da380746-1003-4108-8807-030dc463fb10
# ╟─3e82e093-4d9f-440e-b28c-e5e9a990ad71
# ╟─78ec2dea-5564-4099-8e12-431b62358b96
