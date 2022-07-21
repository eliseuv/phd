for (T, k) in Iterators.product(range(-0.5, 0.5, 11), [-0.1, 0.0, 0.1])
    D = 1.96582 + k
    beta = 1 / (0.60858 + T)
    println("$D $beta")
end
