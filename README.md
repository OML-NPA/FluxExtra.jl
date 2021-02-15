# FluxExtra

Additional layers and functions for the Flux machine learning library.

## Functions

### Moving models between GPU and CPU

Allows to move complicated models between CPU and GPU. Use `cpu` or `gpu` as a target.
```
move(model,target::Union{typeof(cpu),typeof(gpu)})
```

## Layers

### Parallel
```
Parallel(layers::Tuple)
```
Allows to have multiple branches in a neural network.

### Catenation
```
Catenation(dim::Int64)
```
Concatenates an array of arrays along a dimension `dim`. A convenient way of writing `x -> cat(x..., dims = dim)`.

### Decatenation
```
Decatenation(output::Int64,dim::Int64)
```
Breaks an array into a number of arrays which is equal to `output` along a dimension `dim`.

### Addition
```
Addition()
```
A convenient way of using `x -> sum(x)`

### Upscaling
```
Upscaling(multiplier::Int64,dims::Union{Int64,Tuple{Int64,Int64},Tuple{Int64,Int64,Int64}}))
```
Bileniar upscaling. Scales the input array by `multiplier`. Requires a user to input a new size as a second argument and from 1 to 3 dimensions that should be scaled.

### Activation
```
Activation(f::Function)
```
A convenient way of using `x -> some_activation_function`.

### Identity
```
Identity()
```
Returns its input without changes. Should be used with a Parallel layer if one wants to have a branch that does not change its input.
