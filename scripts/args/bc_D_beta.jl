for (T, k) in Iterators.product(range(-0.5, 0.5, 11), [-0.1, 0.0, 0.1])
    beta = 1 / (0.60858 + T)
    D = 1.96582 + k
    println("$D $beta")
end
