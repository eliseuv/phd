module Thesis

export
    # Square grid iterative subdivision
    square_gird_iter,
    # Finite state site
    AbstractSiteState, instance_count,
    # Abstract finite state
    AbstractFiniteState,
    set_state!, randomize_state!,
    nearest_neighbors, nearest_neighbors_sum,
    # Mean field finite state
    MeanFieldFiniteState,
    clear, split_indices, site_counts_from_split_indices,
    # Concrete finite state
    ConcreteFiniteState
# Square lattice finite state

# Finite state representations
include("finite_states.jl")

end
