ising_2d_temp_crit = 2 / log1p(sqrt(2))
for x in range(0.1, 3, 25)
    println(1 / (x * ising_2d_temp_crit))
end
