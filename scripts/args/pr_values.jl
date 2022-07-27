for (dr, p) in Iterators.product(range(-0.1, 0.1, 11), 0.3)
    r = 0.194421 + dr
    println("$p $r")
end
