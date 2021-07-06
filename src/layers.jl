
# Join layer
struct Join
    dim::Int64
end

function cat_dispatcher(x::NTuple{N,T},dim::Int64) where {N,T<:AbstractMatrix}
    if dim==1
        return vcat(x...)
    else
        msg::String = string("Only dimensions 1 is supported. Given ",dim,".")
        err::DimensionMismatch = DimensionMismatch(msg)
        throw(err)
    end
end

function cat_dispatcher(x::NTuple{N,T},dim::Int64) where {N,Y,T<:AbstractArray{Y,4}}
    if dim==1
        return vcat(x...)
    elseif dim==2
        return hcat(x...)
    elseif dim==3
        return cat(x...,dims=Val(3))
    else
        msg::String = string("Only dimensions 1,2 and 3 are supported. Given ",dim,".")
        err::DimensionMismatch = DimensionMismatch(msg)
        throw(err)
    end
end

(m::Join)(x::NTuple{N,T}) where {T<:AbstractArray,N} = cat_dispatcher(x,m.dim)

# Split layer
struct Split
    outputs::Int64
    dim::Int64
end
function Split_func(x::T,outputs::Int64, 
        dim::Int64) where T<:AbstractArray{<:AbstractFloat,2}
    if dim!=1
        throw(DimensionMismatch("Dimension should be 1."))
    end
    step_var = Int64(size(x, dim) / outputs)
    f = i::Int64 -> (1+(i-1)*step_var):(i)*step_var
    vals = ntuple(i -> i, outputs)
    inds_vec = map(f,vals)
    x_out = map(inds -> x[inds,:],inds_vec)
    return x_out
end
function Split_func(x::T,outputs::Int64, 
        dim::Int64) where T<:AbstractArray{<:AbstractFloat,4}
    if dim>3
        throw(DimensionMismatch("Dimension should be 1, 2 or 3."))
    end
    step_var = Int64(size(x, dim) / outputs)
    f = i::Int64 -> (1+(i-1)*step_var):(i)*step_var
    vals = ntuple(i -> i, outputs)
    inds_vec = map(f,vals)
    if dim == 1
        x_out = map(inds -> x[inds,:,:,:],inds_vec)
    elseif dim == 2
        x_out = map(inds -> x[:,inds,:,:],inds_vec)
    else # dim == 3
        x_out = map(inds -> x[:,:,inds,:],inds_vec)
    end
    return x_out
end
(m::Split)(x) = Split_func(x,m.outputs,m.dim)

# Addition layer
struct Addition 
end
(m::Addition)(x) = sum(x)

# Activation layer
struct Activation
    f::Function
end
(m::Activation)(x) = m.f.(x)

# Flatten
struct Flatten 
end
(m::Flatten)(x) = Flux.flatten(x)

# Identity layer
struct Identity
end
(m::Identity)(x) = x