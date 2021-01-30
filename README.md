# FluxExtra

Additional layers for the Flux machine learning library.

### Parallel
```
Parallel(x::AbstractArray{Float32,4},layers::Tuple)
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
