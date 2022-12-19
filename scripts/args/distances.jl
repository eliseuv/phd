distances_list = ["Distances.euclidean",
    "Distances.sqeuclidean",
    "Distances.cityblock",
    "Distances.chebyshev",
    "Distances.hamming",
    "Distances.rogerstanimoto",
    "Distances.jaccard",
    "Distances.braycurtis",
    "Distances.chisq_dist",
    "Distances.kl_divergence",
    "Distances.gkl_divergence",
    "Distances.renyi_divergence",
    "Distances.js_divergence",
    "Distances.bhattacharyya",
    "Distances.meanad",
    "Distances.msd",
    "Distances.rmsd",
    "Distances.nrmsd"]
for (sigma, dist_str) in Iterators.product(0.01:0.005:0.05, ["Distances.sqeuclidean"])
    println("$sigma $dist_str")
end
