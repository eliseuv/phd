const D = 0
const β_c = 1.0 / 1.69378
const τ_vals = 2.0 .^ (-0.2:0.05:0.2)

for (beta, L) ∈ Iterators.product(β_c .* τ_vals, 2 .^ (6:6))
    println("$L $D $beta")
end
