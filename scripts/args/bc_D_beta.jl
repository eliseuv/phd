for (T, D) in Iterators.product(range(-0.5, 0.5, 11), range(-0.5, 0.5, 11))
    beta = 1 / (0.60858 + T)
    Delta = 1.96582 + D
    println("$beta $Delta")
end
