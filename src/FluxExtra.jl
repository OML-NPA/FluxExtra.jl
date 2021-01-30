module FluxExtra

using Flux, CUDA
import CUDA.CuArray
include("layers.jl")

export Parallel, Catenation, Decatenation, Upscaling, Addition, Activation, Identity

end
