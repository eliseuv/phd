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
    using PlutoUI, DrWatson, CSV, DataFrames, JLD2, StatsBase, CairoMakie
    # Custom modules
    include(joinpath(PROJ_ROOT, "notebooks", "ingredients.jl"))
    Thesis = ingredients(joinpath(PROJ_ROOT, "src", "Thesis.jl")).Thesis
    import .Thesis.DataIO
end


# ╔═╡ 14624983-36e2-4142-ba23-c1013cb1a91e
@quickactivate "phd"

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
	
	const datafiles = sort(DataIO.find_datafiles(data_dirpath, prefix, params_req), by=x -> x.params["beta"])

end

# ╔═╡ 4089cee9-8cc1-4143-9fbc-92e2261ce189
@bind D PlutoUI.Slider(unique(sort(map(x -> x.params["D"], datafiles))))

# ╔═╡ e1b9515e-1e2b-47d2-83fa-7c1bc893e033
begin
	datafiles_D = filter(x -> x.params["D"] == D, datafiles)
	
	md"""
	 $D =$ $(D)
	"""
end

# ╔═╡ 3b76757c-958b-425b-8f1b-88c3f83bd33b
begin
	# Critical temperature for such systems
	const T_c = df_temperatures[
		only(findall(==(D), df_temperatures.anisotropy_field)),
		:Butera_and_Pernici]
	md"""
	 $T_c =$ $(T_c)
	"""
end

# ╔═╡ 7d9f981e-7130-4db8-8865-dd6388445296
@bind beta PlutoUI.Slider(reverse(unique(sort(map(x -> x.params["beta"], datafiles_D)))))

# ╔═╡ becf0b50-6428-4b56-b5b4-ab4c4d2e360a
begin

	datafile = only(filter(x -> x.params["beta"] == beta, datafiles_D))

	tau = 1.0 / (T_c * beta)
	
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

# ╔═╡ f7760564-f389-452d-8f4b-945b53cdd42d
hist(vcat(eigvals...), bins=128;
     axis=(; title=L"Eigenvalues distribution $D = %$(D)$, $\tau = %$(tau)$",
			  yscale=Makie.pseudolog10))

# ╔═╡ 6b2f7168-fdbf-4d3e-9fe7-e07637849385
hist(vcat(map(diff, eigvals)...), bins=128;
     axis=(; title=L"Eigenvalues gaps $D = %$(D)$, $\tau = %$(tau)$",
			  yscale=Makie.pseudolog10))

# ╔═╡ Cell order:
# ╟─97e1cec4-b47c-11ed-0f49-59ce159b2c34
# ╟─14624983-36e2-4142-ba23-c1013cb1a91e
# ╟─dc10501c-0fe5-43d3-90f1-4173bdd34311
# ╟─1713b9bd-8d8c-4b79-8062-3d73d6a4e6d3
# ╟─4089cee9-8cc1-4143-9fbc-92e2261ce189
# ╟─e1b9515e-1e2b-47d2-83fa-7c1bc893e033
# ╟─3b76757c-958b-425b-8f1b-88c3f83bd33b
# ╟─7d9f981e-7130-4db8-8865-dd6388445296
# ╟─becf0b50-6428-4b56-b5b4-ab4c4d2e360a
# ╟─fdad1390-fab2-4103-a213-fd244e47faf7
# ╟─f7760564-f389-452d-8f4b-945b53cdd42d
# ╟─6b2f7168-fdbf-4d3e-9fe7-e07637849385
# ╟─2268a332-6c4d-42e2-9dd0-eedaa25fb7d8
# ╟─dda5413a-0088-4d78-9a64-ca76a6b2226b
# ╟─aa1d6af5-2ff1-4899-9cfd-e1ede9a28856
# ╟─08f62964-70d2-4b08-93ac-3ea532f02a17
