for (β, N) in Iterators.product(2 .^ range(-4, 4, 51), [2^10, 2^12, 2^14, 2^16])
    println("$N $β")
end
