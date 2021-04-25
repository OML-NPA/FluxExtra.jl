module FluxExtra

using Flux, CUDA
import CUDA.CuArray
include("layers.jl")
include("utilities.jl")

export Join, Split, Addition, Activation, Identity, move

end
