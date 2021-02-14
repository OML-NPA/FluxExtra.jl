
# Parallel layer
struct Parallel
    layers::NTuple{N,Any} where N
end
function Parallel(x::T,layers::NTuple{N,Any}) where{T<:AbstractArray{<:AbstractFloat,4},N}
    result= map(fun -> fun(x), layers)
    return result
end
function Parallel(x::NTuple{N,T},layers::NTuple{N,Any}) where {T<:AbstractArray{<:AbstractFloat,4},N}
    result = map((fun,xn) -> fun(xn), layers,x)
    return result
end
(m::Parallel)(x) = Parallel(x, m.layers)
Flux.@functor Parallel

# Catenation layer
struct Catenation
    dim::Int64
end
(m::Catenation)(x::NTuple{N,T}) where {T,N} = cat(x...,dims=m.dim)::T

# Decatenation layer
struct Decatenation
    outputs::Int64
    dim::Int64
end
function Decatenation_func(x::T,outputs::Int64, 
        dim::Int64) where T<:AbstractArray{<:AbstractFloat,4}
    step_var = Int64(size(x, dim) / outputs)
    f = i::Int64 -> (1+(i-1)*step_var):(i)*step_var
    vals = ntuple(i -> i, outputs)
    inds_vec = map(f,vals)
    if dim == 1
        x_out = map(inds -> x[inds,:,:,:],inds_vec)
    elseif dim == 2
        x_out = map(inds -> x[:,inds,:,:],inds_vec)
    else
        x_out = map(inds -> x[:,:,inds,:],inds_vec)
    end
    return x_out
end
(m::Decatenation)(x) = Decatenation_func(x,m.outputs,m.dim)

# Addition layer
struct Addition end
(m::Addition)(x) = sum(x)

# Upscaling layer
struct Upscaling
    multiplier::Int64
    dims::Union{Int64,Tuple{Int64,Int64},Tuple{Int64,Int64,Int64}}
end
function Upscaling_func(x::AbstractArray{T,4}, multiplier::Int64,
        dims::Union{Int64,Tuple{Int64,Int64},Tuple{Int64,Int64,Int64}}) where T<:AbstractFloat
    if dims == 1
        ratio = (multiplier,1,1,1)
    elseif dims == 2
        ratio = (1,multiplier,1,1)
    elseif dims == 3
        ratio = (1,1,multiplier,1)
    elseif dims == (1,2)
         ratio = (multiplier,multiplier,1,1)
    elseif dims == (1,2,3)
        ratio = (multiplier,multiplier,multiplier,1)
    end
    return upscale(x,ratio)
end
function upscale(x::Array{T,4},ratio::Tuple{Int64,Int64,Int64,Int64}) where T<:AbstractFloat
    s = size(x)
    h,w,c,n = s
    s_ratio = (ratio[1], 1, ratio[2], 1, ratio[3], 1)
    y = ones(T,s_ratio)
    z = reshape(x, (1, h, 1, w, 1, c, n))  .* y
    new_x = reshape(z, s .* ratio)
    return new_x
end
function upscale(x::CuArray{T,4},ratio::Tuple{Int64,Int64,Int64,Int64}) where T<:AbstractFloat
    s = size(x)
    h,w,c,n = s
    s_ratio = (ratio[1], 1, ratio[2], 1, ratio[3], 1)
    y = cu(ones(T,s_ratio))
    z = reshape(x, (1, h, 1, w, 1, c, n))  .* y
    new_x = reshape(z, s .* ratio)
    return new_x
end
(m::Upscaling)(x) = Upscaling_func(x,m.multiplier,m.dims)

# Activation layer
struct Activation
    f::Function
end
(m::Activation)(x) = m.f.(x)

# Identity layer
struct Identity
end
(m::Identity)(x) = x