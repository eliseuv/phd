for (gamma, sigma, dist_str) in Iterators.product(0.1:0.1:0.9, 0.1:0.1:0.9, ["euclidean",
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
    "nrmsd"])
    println("$gamma $sigma $dist_str")
end
