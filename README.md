# FluxExtra

Additional layers for the Flux machine learning library.

### Parallel
```
Parallel(layers::Tuple)
```
### Catenation
```
Catenation(dim::Int64)
```
### Decatenation
```
Decatenation(output::Int64,dim::Int64)
```
### Addition
```
Addition()
```
### Upscaling
```
Upscaling(multiplier::Float64,new_size::Tuple{Int64,Int64,Int64},dims::Union{Int64,Tuple{Int64,Int64},Tuple{Int64,Int64,Int64}}))
```
### Activation
```
Activation(f::Function)
```
### Identity
```
Identity()
```
Returns its input without changes. Should be used with a Parallel layer if one wants to have a branch that does not change its input.
