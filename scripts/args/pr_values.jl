for (r, p, L) in Iterators.product(collect(0.19:0.0005:0.2), 0.3, 256)
    #r = 0.194421 + dr
    println("$L $p")
end
