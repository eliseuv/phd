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
for (sigma, n_bins, dist_str) in Iterators.product(range(0.01, 0.5, length=11), [64, 128, 256], ["Distances.sqeuclidean"])
    println("$sigma $n_bins $dist_str")
end
