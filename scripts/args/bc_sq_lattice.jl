β_c = 1.69378
τ_vals = 2.0 .^ (-3:0.25:3)

for (beta, L) ∈ Iterators.product(β_c .* τ_vals, 2 .^ (6:11))
    println("$L $beta")
end
