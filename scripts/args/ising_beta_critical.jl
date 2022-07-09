T_C = 2 / log1p(sqrt(2))
for x in range(-0.5, 0.5, 21)
    beta = 1 / (T_C + x)
    println("$(beta)")
end
