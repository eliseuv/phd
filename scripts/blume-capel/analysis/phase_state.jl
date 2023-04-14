### A Pluto.jl notebook ###
# v0.19.24

using Markdown
using InteractiveUtils

# ╔═╡ 97e1cec4-b47c-11ed-0f49-59ce159b2c34
begin
    # Select project
    import Pkg
    Pkg.activate(Base.current_project())
	PROJ_ROOT = dirname(Base.current_project())
    # Libraries
    using PlutoUI, DrWatson, CSV, DataFrames, LaTeXStrings, Gadfly, Compose, SVG
    # Custom modules
    include(joinpath(PROJ_ROOT, "notebooks", "ingredients.jl"))
    Thesis = ingredients(joinpath(PROJ_ROOT, "src", "Thesis.jl")).Thesis
    import .Thesis.DataIO
end


# ╔═╡ 2751c4da-e9c8-4112-b0ce-0d10320056d8
df_phase_stacked = sort(dropmissing(stack(DataFrame(CSV.File(joinpath(PROJ_ROOT, "tables", "butera_and_pernici_2018", "blume-capel_square_lattice.csv"))), [:Butera_and_Pernici, :Jung_and_Kim, :Zierenberg_et_al, :Kwak_et_al, :Beale, :Silva_et_al, :Malakis_et_al, :Quian_et_al], variable_name=:source, value_name=:temperature)), [:transition_order], rev=true)

# ╔═╡ 65da14f8-78ac-483f-a07c-f3c9cff98e7a
plt = plot(df_phase_stacked,
	x=:anisotropy_field, y=:temperature, color=:transition_order,
	#Coord.cartesian(xmin=1.9, xmax=2, ymin=0.3, ymax=0.8),
	Guide.xlabel("D"), Guide.ylabel("T"),
	Guide.colorkey(title="Transition order", pos=[0.1w, 0.25h]),
	#Guide.title("Blume-Capel 2D square lattice phase space"),
	Scale.color_discrete_manual("red","blue","green"),
	Theme(point_shapes=[Shape.diamond],
	      point_size=0.75mm, alphas=[0.75],
          highlight_width = 0pt)
)

# ╔═╡ 0e8eb924-0403-4490-be7f-8639ec3af83e
draw(SVG(plotsdir("blume_capel_pickles", "phase_space.svg")), plt)

# ╔═╡ d5135661-2e32-4dbe-910c-27b317bfc596
plt_detail = plot(df_phase_stacked,
	x=:anisotropy_field, y=:temperature, color=:transition_order,
	Coord.cartesian(xmin=1.93, xmax=2, ymin=0.3, ymax=0.7),
	Guide.xlabel("D"), Guide.ylabel("T"),
	Guide.colorkey(title="Transition order", pos=[0.1w, 0.25h]),
	#Guide.title("Blume-Capel 2D square lattice phase space"),
	Scale.color_discrete_manual("red","blue","green"),
	Theme(point_shapes=[Shape.diamond],
	      point_size=0.75mm, alphas=[0.75],
          highlight_width = 0pt)
)

# ╔═╡ c2695242-8b88-4b19-9c89-2d598b87d7b5
draw(SVG(plotsdir("blume_capel_pickles", "phase_space_detail.svg")), plt_detail)

# ╔═╡ 9736dfb3-dd6a-4919-adfd-e24f93367bb3
plot(
	layer(filter(:transition_order => ==("TCP"), df_phase_stacked),
		  x=:anisotropy_field, y=:temperature, color=:transition_order),
	layer(filter(:transition_order => ==("First"), df_phase_stacked),
		  x=:anisotropy_field, y=:temperature, color=:transition_order),
	layer(filter(:transition_order => ==("Second"), df_phase_stacked),
		  x=:anisotropy_field, y=:temperature, color=:transition_order),
	Guide.xlabel("D"), Guide.ylabel("T"),
	Guide.colorkey(title="Transition order", pos=[0.1w, 0.25h]),
	Guide.title("Blume-Capel 2D square lattice"),
	Scale.color_discrete_manual("red","green","blue"),
	Theme(point_shapes=[Shape.diamond],
	      point_size=0.5mm, alphas=[0.5],
          highlight_width = 0pt)
)

# ╔═╡ Cell order:
# ╠═97e1cec4-b47c-11ed-0f49-59ce159b2c34
# ╠═2751c4da-e9c8-4112-b0ce-0d10320056d8
# ╠═65da14f8-78ac-483f-a07c-f3c9cff98e7a
# ╠═0e8eb924-0403-4490-be7f-8639ec3af83e
# ╠═d5135661-2e32-4dbe-910c-27b317bfc596
# ╠═c2695242-8b88-4b19-9c89-2d598b87d7b5
# ╠═9736dfb3-dd6a-4919-adfd-e24f93367bb3
