for (dr, p, L) in Iterators.product(range(-0.1, 0.1, 11), 0.3, [16, 32, 64, 128, 256])
    r = 0.194421 + dr
    println("$L $p $r")
end
