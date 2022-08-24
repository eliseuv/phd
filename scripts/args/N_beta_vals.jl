for (β, z, N) in Iterators.product(2 .^ range(-4, 4, 51), [4, 6], [2^10, 2^12])
    println("$N $z $β")
end
