### A Pluto.jl notebook ###
# v0.19.14

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

# ╔═╡ b545ab73-1a5f-47e8-b518-ae1620c07e4b
begin
	import Pkg
	Pkg.activate(Base.current_project())
	using PlutoUI, Gadfly
end

# ╔═╡ bf8c69cf-ed83-43c0-908f-0aae7fd0eb9c
begin
	include("/home/evf/.julia/pluto_notebooks/ingredients.jl")
	Thesis = ingredients("../src/Thesis.jl").Thesis
	import .Thesis.Covid19Data as C19Data
end

# ╔═╡ b92ecc52-be89-42b3-8eb5-46295dc48e85
begin
	# Fetch data to update
	#fetch_jhu_c19_time_series()
	# Or just load it
	df_raw = C19Data.load_jhu_c19_time_series()
end

# ╔═╡ 78b8e182-27f0-4968-9c43-30c16ab7ed84
regions = C19Data.get_region_list(df_raw)

# ╔═╡ 306e4d73-7e70-43d0-8194-6f99e6234519
time_range = C19Data.get_time_range(df_raw)

# ╔═╡ 99e6e342-5883-4dc9-bc90-87ba2352d89d
md"""
Select region: $(@bind region Select(regions, default="Brazil"))
"""

# ╔═╡ 29e90ce8-b8ee-4cf2-8fb8-fff1106c035f
begin
	df = C19Data.get_c19_region_time_series(df_raw, region)
	
	plot(df, x=:Date, y=:confirmed,
		Geom.line)
end

# ╔═╡ Cell order:
# ╠═b545ab73-1a5f-47e8-b518-ae1620c07e4b
# ╠═bf8c69cf-ed83-43c0-908f-0aae7fd0eb9c
# ╠═b92ecc52-be89-42b3-8eb5-46295dc48e85
# ╠═78b8e182-27f0-4968-9c43-30c16ab7ed84
# ╠═306e4d73-7e70-43d0-8194-6f99e6234519
# ╠═99e6e342-5883-4dc9-bc90-87ba2352d89d
# ╠═29e90ce8-b8ee-4cf2-8fb8-fff1106c035f
