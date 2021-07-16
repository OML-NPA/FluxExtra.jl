module FluxExtra

using Flux, Statistics

include("layers.jl")
include("Normalizations.jl")

export Join, Split, Addition, Activation, Flatten, Identity
export Normalizations

end
