module FluxExtra

using Flux, CUDA
import CUDA.CuArray
include("layers.jl")

export Join, Split, Addition, Activation, Identity

end
