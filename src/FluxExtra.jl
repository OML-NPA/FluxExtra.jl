module FluxExtra

using Flux, CUDA
import CUDA.CuArray
include("layers.jl")
include("utilities.jl")

export Parallel, Catenation, Decatenation, Upscaling, Addition, Activation, Identity, move

end
