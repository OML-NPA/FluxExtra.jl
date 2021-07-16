[![CI](https://github.com/OML-NPA/FluxExtra.jl/actions/workflows/CI-main.yml/badge.svg)](https://github.com/OML-NPA/FluxExtra.jl/actions/workflows/CI-main.yml)
[![codecov](https://codecov.io/gh/OML-NPA/FluxExtra.jl/branch/main/graph/badge.svg?token=JROBFGEVQN)](https://codecov.io/gh/OML-NPA/FluxExtra.jl)

# FluxExtra

Additional layers and functions for the [Flux.jl](https://github.com/FluxML/Flux.jl) machine learning library.

## Layers

### Join
```
Join(dim::Int64)
Join(dim = dim::Int64)
```
Concatenates a tuple of arrays along a dimension `dim`. A convenient and type stable way of using `x -> cat(x..., dims = dim)`.

### Split
```
Split(outputs::Int64,dim::Int64)
Split(outputs::Int64, dim = dim::Int64)
```
Breaks an array into a number of arrays which is equal to `output` along a dimension `dim`. `dim` should we divisible by `outputs` without a remainder.

### Flatten
```
Flatten()
```
Flattens an array. A convenient way of using `x -> Flux.flatten(x)`.

### Addition
```
Addition()
```
A convenient way of using `x -> sum(x)`.

### Activation
```
Activation(f::Function)
```
A convenient way of using `x -> f(x)`.

### Identity
```
Identity()
```
Returns its input without changes. Should be used with a `Parallel` layer if one wants to have a branch that does not change its input.

## Normalizations

### [0,1]
```
norm_01!(data::Vector{T}) where {F<:AbstractFloat,N,T<:Array{F,N}}
```
Rescales each feature (last dimension) to be in the range [0,1].

### [-1,1]
```
norm_negpos1(data::Vector{T}) where {F<:AbstractFloat,N,T<:Array{F,N}}
```
Rescales each feature (last dimension) to be in the range [-1,1].

### Zero center
```
norm_zerocenter!(data::Vector{T}) where {F<:AbstractFloat,N,T<:Array{F,N}}
```
Subtracts the mean of each feature (last dimension).

### Z-score
```
norm_zscore!(data::Vector{T}) where {F<:AbstractFloat,N,T<:Array{F,N}}
```
Subtracts the mean and divides by the standard deviation of each feature (last dimension).

## Other

Makes `Flux.Parallel` layer type stable when used with tuples.
