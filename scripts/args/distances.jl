distances_list = ["euclidean",
    "sqeuclidean",
    "cityblock",
    "chebyshev",
    "hamming",
    "rogerstanimoto",
    "jaccard",
    "braycurtis",
    "chisq_dist",
    "kl_divergence",
    "gkl_divergence",
    "renyi_divergence",
    "js_divergence",
    "bhattacharyya",
    "meanad",
    "msd",
    "rmsd",
    "nrmsd"]
for (run, dist_str) in Iterators.product(1:32, ["nrmsd"])
    println("$run $dist_str")
end
