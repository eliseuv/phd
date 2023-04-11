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
    using PlutoUI, DrWatson, CSV, DataFrames, LaTeXStrings, CairoMakie
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
