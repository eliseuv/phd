### A Pluto.jl notebook ###
# v0.19.22

using Markdown
using InteractiveUtils

# ╔═╡ 7662268a-b47d-11ed-319d-235c9fc37094
begin
    # Select project
    import Pkg
    Pkg.activate(Base.current_project())
	PROJ_ROOT = dirname(Base.current_project())
    # Libraries
    using PlutoUI # and others
    # Custom modules
    include(joinpath(PROJ_ROOT, "notebooks", "ingredients.jl"))
    Thesis = ingredients(joinpath(PROJ_ROOT, "src", "Thesis.jl")).Thesis
    import .Thesis.DataIO
end

# ╔═╡ Cell order:
# ╠═7662268a-b47d-11ed-319d-235c9fc37094
