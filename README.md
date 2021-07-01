# FluxExtra

Additional layers for the [Flux.jl](https://github.com/FluxML/Flux.jl) machine learning library.

## Layers

### Join
```
Join(dim::Int64)
```
Concatenates an array of arrays along a dimension `dim`. A convenient way of using `x -> cat(x..., dims = dim)`.

### Split
```
Split(output::Int64,dim::Int64)
```
Breaks an array into a number of arrays which is equal to `output` along a dimension `dim`.

### Addition
```
Addition()
```
A convenient way of using `x -> sum(x)`

### Activation
```
Activation(f::Function)
```
A convenient way of using `x -> some_activation_function`.

### Identity
```
Identity()
```
Returns its input without changes. Should be used with a `Parallel` layer if one wants to have a branch that does not change its input.
