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
    using PlutoUI, DrWatson, LinearAlgebra, Statistics, JLD2, DataFrames, CairoMakie, ColorSchemes
    # Custom modules
    include("/home/evf/.julia/pluto_notebooks/ingredients.jl")
    Thesis = ingredients(joinpath(project_path, "src/Thesis.jl")).Thesis
    import .Thesis.DataIO
	import .Thesis.TimeSeries
end


# ╔═╡ 0b0b3f06-34d7-451c-9895-acaac1bbcf2b
@inline function get_values(M::AbstractMatrix{T}) where {T<:Real}
	ax = axes(M, 1)
    axes(M,2) == ax || error("M must be square")
    n = length(ax)
    vals = Vector{T}(undef, ((n-1)*n) ÷ 2)
    @inbounds @views for (k, (i, j)) ∈ zip(eachindex(vals), (i, j) for i ∈ ax for j ∈ i+1:last(ax))
        vals[k] = M[i,j]
    end
    return vals
end

# ╔═╡ 27a9e09a-6279-46f0-a78b-4ccbf4203dc2
begin
	# Required parameters
	const params_req = Dict("dist" => "sqeuclidean")
	# Measurement to plot (`costs` or `variance`)
	const measure = "costs"
	# Color key
	const key_name = "sigma"

	datafiles =
    	DataIO.find_datafiles(
        	datadir("gen_corr_dist/simulated_annealing/2022-12-22"),
        			"GenUniformCorrDistSA",
        			params_req,
        			ext="jld2")
	
	sort!(datafiles, by= x -> x.params[key_name])

	key_values = [x.params[key_name] for x in datafiles]

	println("$key_name:\n$key_values")
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
    		  ylabel="Normalied cost",
			  yscale=log10)
	
	# Final distribution axis
	ax_dist = Axis(fig[2,1],
				   title="Final Correlation Distribution",
				   xlabel=L"\rho")

	# Eigenvalues distribution axis
	ax_eigvals = Axis(fig[3,1],
	                  title="Eigenvalues distribution",
	    			  xlabel=L"\lambda",
					  yscale=Makie.pseudolog10)

	df_final_dist_std = DataFrame(key_name => Float64[],
						      	  "std" => Float64[])

	# Loop on datafiles
	for (k, datafile) ∈ enumerate(datafiles)
		
		# Key value
		key_value = datafile.params[key_name]
		
		# Load data file
  	  	(df, M_ts) = load(datafile.path, "df", "M_ts")
		G = TimeSeries.cross_correlation_matrix(M_ts)
		corr_vals = get_values(G)
		λs = eigvals(G)

		# Normalize measurement time series
	  	df[!, "norm_"*measure] = df[!, measure] ./ df[!, measure][begin]

		# Select color
		color = ColorSchemes.viridis[(k-1)/(length(datafiles)-1)]

		# Plot simulated annealing
  		lines!(ax, df[!, "norm_"*measure],
			   label=L"%$(key_value)",
			   color=(color, 0.7))

		# Plot final distribution
		hist!(ax_dist, corr_vals,
			  bins=128,
			  normalization=:pdf,
			  color=(color, 0.7))
		# Plot eigenvalues distribution
		hist!(ax_eigvals, λs,
			  bins=512,
			  normalization=:pdf,
			  color=(color, 0.7))

		# Store final distribution
		push!(df_final_dist_std, [key_value, std(corr_vals)])
		
	end

	# Add legend
	Legend(fig[:, 2], ax, label=L"\sigma")

	fig

end

# ╔═╡ 39a8f0aa-46f9-491f-b238-096c94508eab
lines(df_final_dist_std[!,"sigma"], df_final_dist_std[!,"std"],
	  axis=(; title="Final distribution standard deviation as a function of perturbation magnitude",
	  		xlabel=L"$\sigma$ (perturabation)", ylabel=L"Final distribution  $\sigma$"))

# ╔═╡ Cell order:
# ╠═15e9d076-7fa5-11ed-2de4-27d5745b75fb
# ╠═0b0b3f06-34d7-451c-9895-acaac1bbcf2b
# ╠═27a9e09a-6279-46f0-a78b-4ccbf4203dc2
# ╠═84a9ff45-e029-4bd7-8825-454600ab2273
# ╠═39a8f0aa-46f9-491f-b238-096c94508eab
