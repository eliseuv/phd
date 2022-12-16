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
for (sigma, dist_str) in Iterators.product(0.1:0.1:3.0, ["nrmsd"])
    println("$sigma $dist_str")
end
