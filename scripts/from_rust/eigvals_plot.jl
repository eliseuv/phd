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
    using PlutoUI, DrWatson, CSV, DataFrames, JLD2, StatsBase, LinearAlgebra, LaTeXStrings, CairoMakie, PyCall
    # Custom modules
    include(joinpath(PROJ_ROOT, "notebooks", "ingredients.jl"))
    Thesis = ingredients(joinpath(PROJ_ROOT, "src", "Thesis.jl")).Thesis
    import .Thesis.DataIO
end


# ╔═╡ 14624983-36e2-4142-ba23-c1013cb1a91e
@quickactivate "phd"

# ╔═╡ 0a7ae038-43f6-4477-adfc-dc25a9404222
make_ticks_log(powers::AbstractVector{<:Real}, base::Integer=10) = (Float64(base) .^ powers, (map(x -> latexstring("$(base)^{$(x)}"), powers)))

# ╔═╡ 144fd029-364a-4274-bc70-b690e7b9e463
make_ticks_log_(powers::AbstractVector{<:Real}, base::Integer=10) = (powers, (map(x -> latexstring("$(base)^{$(x)}"), powers)))

# ╔═╡ 7362c864-9f4c-46a1-ad41-10a9ca08fbc6
function histogram_mean(vals::AbstractVector{<:Real}, nbins::Integer=128, closed::Symbol=:left)
	hist = fit(Histogram, vals, nbins=nbins, closed=closed)
	x = hist.edges[begin][begin:end-1]
	y = hist.weights
	return (x ⋅ y) / sqrt(y ⋅ y)
end

# ╔═╡ dc10501c-0fe5-43d3-90f1-4173bdd34311
# Load dataframes with the critical temperature data
df_temperatures = DataFrame(CSV.File(joinpath(PROJ_ROOT, "tables", "butera_and_pernici_2018", "blume-capel_square_lattice.csv")))

# ╔═╡ 52683fbe-d51c-4257-9a18-c77e9c16204f
begin
	
	py"""
import pickle
 
def load_pickle(fpath):
    with open(fpath, "rb") as f:
        data = pickle.load(f)
    return data
"""
	# Function to load pi
	load_pickle = py"load_pickle"
end

# ╔═╡ e4e373cb-e6dd-4a81-9baa-a87a0ac7b3ac
begin
	# Parse data
	eigvals_data = Dict()
	correlation_data = Dict()
	const data_dirpath = datadir("blume_capel_pickles")
	for (root, _, filenames) in walkdir(data_dirpath)
		for filename in filenames
			path = joinpath(root, filename)
			datafile = DataIO.DataFile(path)
			D = Float64(datafile.params["D"])
			T = Float64(datafile.params["T"])
			if datafile.prefix == "BlumeCapelSq2DEigvals"
				if haskey(eigvals_data, D)
					eigvals_data[D][T] = datafile
				else
					eigvals_data[D] = Dict(T => datafile)
				end
			elseif datafile.prefix == "BlumeCapelSq2DCorrelations"
				if haskey(correlation_data, D)
					correlation_data[D][T] = datafile
				else
					correlation_data[D] = Dict(T => datafile)
				end
			end
		end
	end
end

# ╔═╡ c7cbe81d-0221-42fc-95e9-24890ec2219b
@bind D PlutoUI.Slider(sort([keys(eigvals_data)...]))

# ╔═╡ d9c5e3ca-e276-4adf-9b18-b1e307a4c23d
begin

    # Get beta values available
    const beta_values = sort([keys(eigvals_data[D])...])

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

# ╔═╡ 175e523c-6656-46da-9076-b9da46ec30e9
md"""
# Eigenvalues fluctuiations
"""

# ╔═╡ cd70d787-984b-41f0-be2a-4d41552a4912
md"""
## Mean eigenvalue
"""

# ╔═╡ d61222d3-1ae7-44cd-86ef-818839291ae8
begin
	eigvals_mean_data = sort(map(Tuple.(collect(pairs(eigvals_data[D])))) do (T, datafile)
		mean = histogram_mean(vec(load_pickle(datafile.path)), 100)
		return (T, mean)
	end, by = x -> first(x))
	(T_eigvals_mean, eigvals_mean) = tuple(map((f) -> map(f, eigvals_mean_data), [first, last])...)
end

# ╔═╡ 4e4806fd-5f03-4924-b291-8588419b7c6f
begin
	scatter(T_eigvals_mean ./ T_c, eigvals_mean,
		    axis=(; title=L"Eigenvalues average $D = %$(D)$ [ %$(trans_order) ( %$(crit_temp_source_str) ) ]",
	 	 	        xlabel=L"\tau",
	 		 	    ylabel=L"\langle\lambda\rangle",
					xticks=0:0.5:6.5,
	  	     		#limits=((0, nothing), (0, nothing)),
	         		yticks=make_ticks_log(0:0.2:1),
			 		yscale=Makie.pseudolog10
	 ))
end

# ╔═╡ 04f40158-fe3b-44f3-82d4-e5b20bd9813e
# ╠═╡ disabled = true
#=╠═╡
begin
	correlations_mean_data = sort(map(Tuple.(collect(pairs(correlation_data[D])))) do (T, datafile)
		#corr_mean = histogram_mean(vec(load_pickle(datafile.path)), 100)
		corr_mean = mean(load_pickle(datafile.path))
		return (T, corr_mean)
	end, by = x -> first(x))
	(T_correlations_mean, correlations_mean) = tuple(map((f) -> map(f, correlations_mean_data), [first, last])...)
end
  ╠═╡ =#

# ╔═╡ 226c1b4b-df87-411a-9e65-64483b22ddd1
#=╠═╡
begin
	scatter(T_correlations_mean ./ T_c, correlations_mean,
		    axis=(; title=L"Correlations average $D = %$(D)$ [ %$(trans_order) ( %$(crit_temp_source_str) ) ]",
	 	 	        xlabel=L"\tau",
	 		 	    ylabel=L"\langle\lambda\rangle",
					xticks=0:0.5:6.5,
	  	     		#limits=((0, nothing), (0, nothing)),
	         		yticks=make_ticks_log(0:0.2:1),
			 		yscale=Makie.pseudolog10
	 ))
end
  ╠═╡ =#

# ╔═╡ 0dd764ff-7536-451e-8bed-cb2cadab5a29
md"""
## Eigenvalues variance
"""

# ╔═╡ e62fe6e9-b67c-41b5-9dd7-2d9e533ef4d9
begin
	eigvals_var_data = sort(map(Tuple.(collect(pairs(eigvals_data[D])))) do (T, datafile)
		variance = var(load_pickle(datafile.path))
		return (T, variance)
	end, by = x -> first(x))
	(T_vals_var, eigvals_var) = tuple(map((f) -> map(f, eigvals_var_data), [first, last])...)
end

# ╔═╡ 0cb12e4d-3f9a-493c-bb61-1b72891efb68
begin
	scatter(T_vals_var ./ T_c, eigvals_var,
		    axis=(; title=L"Eigenvalues variance $D = %$(D)$ [ %$(trans_order) ( %$(crit_temp_source_str) ) ]",
	 	 	        xlabel=L"\tau",
	 		 	    ylabel=L"\langle\lambda\rangle",
					xticks=0:0.5:6.5,
	  	     		#limits=((0, nothing), (0, nothing)),
	         		yticks=make_ticks_log(0:0.5:2),
			 		yscale=Makie.pseudolog10
	 ))
end

# ╔═╡ 91e2f97c-b901-416f-beab-3844f1faa3cd
md"""
# Eigenvalues histograms
"""

# ╔═╡ 214cec1b-823b-404a-b644-ead476400910
@bind T PlutoUI.Slider(sort([keys(eigvals_data[D])...]))

# ╔═╡ fd6c247c-896a-4b25-a570-cf434affb048
begin

	# Get datafile
	const eigvals_matrix = load_pickle(eigvals_data[D][T].path)'
	const correlations_vals_matrix = load_pickle(eigvals_data[D][T].path)'

	# Calculate tau
	const tau = round(T / T_c, digits=5)
	
	md"""
	 $T =$ $(T)
	
	 $\tau =$ $(tau)
	"""
end


# ╔═╡ d0bf3d2a-ed2d-4f6c-bf6f-47b0771b61e6
md"""
## Eigenvalues ditribution
"""

# ╔═╡ b0427f47-0799-4ae4-9c3a-e589811ad6a9
hist(vec(eigvals_matrix), bins=100;
     axis=(; title=L"Eigenvalues distribution $D = %$(D)$, $\tau = %$(tau)$",
	 	 	 xlabel=L"\lambda",
	 		 #ylabel=L"\rho(\lambda)",
	  	     limits=((0, nothing), (0, nothing)),
	         yticks=make_ticks_log(0:5),
			 yscale=Makie.pseudolog10
	 ))


# ╔═╡ 5b529b4e-0f41-4f74-9f41-dff91cbc2158
md"""
## Eigenvalues spacing ditribution
"""

# ╔═╡ b722206b-8315-435c-a8d1-ca4f95ccd685
hist(vcat(map(diff, eachcol(eigvals_matrix))...), bins=100;
     axis=(; title=L"Eigenvalues spacing distribution $D = %$(D)$, $\tau = %$(tau)$",
	 	 	 xlabel=L"\Delta\lambda",
	 		 #ylabel=L"\rho(\Delta\lambda)",
	  	     limits=((0, nothing), (0, nothing)),
	         yticks=make_ticks_log(0:5),
			 yscale=Makie.pseudolog10))

# ╔═╡ 1b5dd5fd-e545-43ef-93e7-5ff6558d9ba5
md"""
## Min/max eigenvalues statistics
"""

# ╔═╡ 7534abd8-9fa2-40a0-8da5-9d3caaf30114
hist(vcat(map(first, eachcol(eigvals_matrix))...), bins=50;
     axis=(; title=L"Minimum eigenvalue distribution $D = %$(D)$, $\tau = %$(tau)$",
	 	 	 xlabel=L"min(\lambda)",
	 		 #ylabel=L"\rho(\Delta\lambda)",
	  	     limits=((0, nothing), (0, nothing)),
	         yticks=make_ticks_log(0:5),
			 yscale=Makie.pseudolog10))

# ╔═╡ f8be2c07-2a1d-42ae-bf6d-01aa63c73a3f
hist(vcat(map(last, eachcol(eigvals_matrix))...), bins=50;
     axis=(; title=L"Maximum eigenvalue distribution $D = %$(D)$, $\tau = %$(tau)$",
	 	 	 xlabel=L"max(\lambda)",
	 		 #ylabel=L"\rho(\Delta\lambda)",
	  	     #limits=((0, nothing), (0, nothing)),
	         yticks=make_ticks_log(0:5),
			 yscale=Makie.pseudolog10))

# ╔═╡ Cell order:
# ╟─97e1cec4-b47c-11ed-0f49-59ce159b2c34
# ╟─14624983-36e2-4142-ba23-c1013cb1a91e
# ╟─0a7ae038-43f6-4477-adfc-dc25a9404222
# ╟─144fd029-364a-4274-bc70-b690e7b9e463
# ╟─7362c864-9f4c-46a1-ad41-10a9ca08fbc6
# ╟─dc10501c-0fe5-43d3-90f1-4173bdd34311
# ╟─52683fbe-d51c-4257-9a18-c77e9c16204f
# ╟─e4e373cb-e6dd-4a81-9baa-a87a0ac7b3ac
# ╟─c7cbe81d-0221-42fc-95e9-24890ec2219b
# ╟─d9c5e3ca-e276-4adf-9b18-b1e307a4c23d
# ╟─175e523c-6656-46da-9076-b9da46ec30e9
# ╟─cd70d787-984b-41f0-be2a-4d41552a4912
# ╟─d61222d3-1ae7-44cd-86ef-818839291ae8
# ╟─4e4806fd-5f03-4924-b291-8588419b7c6f
# ╟─04f40158-fe3b-44f3-82d4-e5b20bd9813e
# ╟─226c1b4b-df87-411a-9e65-64483b22ddd1
# ╟─0dd764ff-7536-451e-8bed-cb2cadab5a29
# ╟─e62fe6e9-b67c-41b5-9dd7-2d9e533ef4d9
# ╟─0cb12e4d-3f9a-493c-bb61-1b72891efb68
# ╟─91e2f97c-b901-416f-beab-3844f1faa3cd
# ╟─214cec1b-823b-404a-b644-ead476400910
# ╟─fd6c247c-896a-4b25-a570-cf434affb048
# ╟─d0bf3d2a-ed2d-4f6c-bf6f-47b0771b61e6
# ╟─b0427f47-0799-4ae4-9c3a-e589811ad6a9
# ╠═5b529b4e-0f41-4f74-9f41-dff91cbc2158
# ╟─b722206b-8315-435c-a8d1-ca4f95ccd685
# ╟─1b5dd5fd-e545-43ef-93e7-5ff6558d9ba5
# ╟─7534abd8-9fa2-40a0-8da5-9d3caaf30114
# ╟─f8be2c07-2a1d-42ae-bf6d-01aa63c73a3f
