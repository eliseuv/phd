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
    using PlutoUI, JLD2, StatsBase, CairoMakie
    # Custom modules
    include(joinpath(PROJ_ROOT, "notebooks", "ingredients.jl"))
    Thesis = ingredients(joinpath(PROJ_ROOT, "src", "Thesis.jl")).Thesis
    import .Thesis.DataIO
end


# ╔═╡ 12986577-5454-4dbe-9db9-98ba5536ee1b
joinpath(dirname(Base.current_project()), "notebooks", "ingredients.jl")

# ╔═╡ Cell order:
# ╠═97e1cec4-b47c-11ed-0f49-59ce159b2c34
