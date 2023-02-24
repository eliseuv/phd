β_c = 1.0 / 1.5664
τ_vals = 2.0 .^ (-0.2:0.05:0.2)

for (beta, L) ∈ Iterators.product(β_c .* τ_vals, 2 .^ (6:6))
    println("$L $beta")
end
