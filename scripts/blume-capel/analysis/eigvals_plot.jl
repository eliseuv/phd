### A Pluto.jl notebook ###
# v0.19.22

using Markdown
using InteractiveUtils

# ╔═╡ 97e1cec4-b47c-11ed-0f49-59ce159b2c34
begin
    # Select project
    import Pkg
    Pkg.activate(Base.current_project())
	PROJ_ROOT = dirname(Base.current_project())
    # Libraries
    using PlutoUI, DrWatson, CSV, DataFrames, JLD2, StatsBase, CairoMakie
    # Custom modules
    include(joinpath(PROJ_ROOT, "notebooks", "ingredients.jl"))
    Thesis = ingredients(joinpath(PROJ_ROOT, "src", "Thesis.jl")).Thesis
    import .Thesis.DataIO
end


# ╔═╡ 14624983-36e2-4142-ba23-c1013cb1a91e
@quickactivate "phd"

# ╔═╡ dc10501c-0fe5-43d3-90f1-4173bdd34311
df_temperatures = DataFrame(CSV.File(joinpath(PROJ_ROOT, "tables", "butera_and_pernici_2018", "blume-capel_square_lattice.csv")))

# ╔═╡ 1713b9bd-8d8c-4b79-8062-3d73d6a4e6d3
begin
	const data_dirpath = datadir("sims", "blume-capel", "square_lattice")
	const prefix = "BlumeCapelSqLatticeCorrMatEigvals"

	# Anisotropy field
	const D = 1
	
	# Critical temperature for such systems
	const T_c = df_temperatures[
		only(findall(==(D), df_temperatures.anisotropy_field)),
		:Butera_and_Pernici]

	const params_req = Dict(
	    "dim" => 2,
	    "L" => 64,
	    "D" => D,
	    "n_runs" => 1024,
	    "n_samples" => 128,
	    "n_steps" => 512
	)
	
	const datafiles = sort(DataIO.find_datafiles(data_dirpath, prefix, params_req), by=x -> x.params["beta"])

	@show D T_c
	display(map(x -> x.params["beta"], datafiles))

end

# ╔═╡ 11dc8fd3-07dc-4872-96a9-c70044b07099
# ╠═╡ disabled = true
#=╠═╡
for datafile in datafiles

    @info datafile.params

    # Load data from file
    (corr_vals, eigvals) = load(datafile.path, "corr_vals", "eigvals")

    tau = 1.0 / (T_c * datafile.params["beta"])
    @info tau

    plt = hist(vcat(eigvals...), bins=128;
        axis=(; title=L"Eigenvalues distribution $\tau = %$(tau)$",
            yscale=Makie.pseudolog10))

    plot_path = plotsdir("blume-capel", DataIO.filename("BlumeCapelSquareLatticeEigvalsHist", params_req, "tau" => tau, ext="svg"))
    mkpath(dirname(plot_path))
    @info plot_path

    save(plot_path, plt)

end

  ╠═╡ =#

# ╔═╡ fdad1390-fab2-4103-a213-fd244e47faf7
begin
	datafile = datafiles[3]
	@info datafile.params
end

# ╔═╡ f7760564-f389-452d-8f4b-945b53cdd42d
begin
	# Load data from file
	(corr_vals, eigvals) = load(datafile.path, "corr_vals", "eigvals")
	
	tau = 1.0 / (T_c * datafile.params["beta"])
	
	plt = hist(vcat(eigvals...), bins=128;
			   axis=(; title=L"Eigenvalues distribution $\tau = %$(tau)$",
			           yscale=Makie.pseudolog10))
	
end

# ╔═╡ Cell order:
# ╠═97e1cec4-b47c-11ed-0f49-59ce159b2c34
# ╠═14624983-36e2-4142-ba23-c1013cb1a91e
# ╠═dc10501c-0fe5-43d3-90f1-4173bdd34311
# ╠═1713b9bd-8d8c-4b79-8062-3d73d6a4e6d3
# ╠═11dc8fd3-07dc-4872-96a9-c70044b07099
# ╠═fdad1390-fab2-4103-a213-fd244e47faf7
# ╠═f7760564-f389-452d-8f4b-945b53cdd42d
