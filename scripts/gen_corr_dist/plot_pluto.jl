### A Pluto.jl notebook ###
# v0.19.18

using Markdown
using InteractiveUtils

# ╔═╡ 15e9d076-7fa5-11ed-2de4-27d5745b75fb
begin
    # Select project
	project_path = dirname(Base.current_project())
    import Pkg
    Pkg.activate(Base.current_project())
    # Libraries
    using PlutoUI, DrWatson, JLD2, DataFrames, CairoMakie, ColorSchemes
    # Custom modules
    include("/home/evf/.julia/pluto_notebooks/ingredients.jl")
    Thesis = ingredients(joinpath(project_path, "src/Thesis.jl")).Thesis
    import .Thesis.DataIO
	import .Thesis.TimeSeries
end


# ╔═╡ 27a9e09a-6279-46f0-a78b-4ccbf4203dc2
begin
	# Required parameters
	const params_req = Dict("gamma" => 1,
							"dist" => "sqeuclidean")
	# Measurement to plot (`costs` or `variance`)
	const measure = "costs"
	# Color key
	const key_name = "sigma"

	datafiles =
    	DataIO.find_datafiles(
        	datadir("gen_corr_dist/simulated_annealing/2022-12-19/sqeuclidean_long"),
        			"GenUniformCorrDistSA",
        			params_req,
        			ext="jld2")
	
	sort!(datafiles, by= x -> x.params[key_name])
end

# ╔═╡ 84a9ff45-e029-4bd7-8825-454600ab2273
begin
	# Main figure
	fig = Figure(resolution=(1024,1024))
	# Supertitle
	supertitle = Label(fig[0,:],
					   "Generate Uniform Correlation Distribution using Simulated Annealing " * join([name * " = " * string(value) for (name,value) ∈ params_req], ", "))
	# Simulated annealing axis
	ax = Axis(fig[1,1],
    	      title="Simulated Annealing ",
	    	  xlabel="iter",
    		  ylabel="Normalied cost")
	# Final distribution axis
	ax_dist = Axis(fig[2,1],
				   title="Final Correlation Distribution",
				   xlabel=L"\rho")

	# Loop on datafiles
	for (k, datafile) ∈ enumerate(datafiles)
		# Ke key value
		key_value = datafile.params[key_name]
		# Load data file
  	  	(df, M_ts) = load(datafile.path, "df", "M_ts")
		# Normalize measurement time series
	  	df[!, "norm_"*measure] = df[!, measure] ./ df[!, measure][begin]

		# Select color
		color = ColorSchemes.viridis[(k-1)/(length(datafiles)-1)]

  		lines!(ax, df[!, "norm_"*measure],
			   label=L"%$(key_value)",
			   color=(color, 0.7))

		hist!(ax_dist, TimeSeries.cross_correlation_values_norm(M_ts),
			  bins=128,
			  normalization=:pdf,
			  color=(color, 0.7))
		
	end

	# Add legend
	Legend(fig[:, 2], ax, label=L"\sigma")

	fig

end

# ╔═╡ Cell order:
# ╠═15e9d076-7fa5-11ed-2de4-27d5745b75fb
# ╠═27a9e09a-6279-46f0-a78b-4ccbf4203dc2
# ╠═84a9ff45-e029-4bd7-8825-454600ab2273
