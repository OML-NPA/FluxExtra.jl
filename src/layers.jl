
# Join layer
struct Join
    dim::Int64
end
(m::Join)(x::NTuple{N,T}) where {T,N} = cat(x...,dims=m.dim)::T

# Split layer
struct Split
    outputs::Int64
    dim::Int64
end
function Split_func(x::T,outputs::Int64, 
        dim::Int64) where T<:AbstractArray{<:AbstractFloat,2}
    if dim>2
        throw(BoundsError(x,dim))
    end
    step_var = Int64(size(x, dim) / outputs)
    f = i::Int64 -> (1+(i-1)*step_var):(i)*step_var
    vals = ntuple(i -> i, outputs)
    inds_vec = map(f,vals)
    if dim == 1
        x_out = map(inds -> x[inds,:],inds_vec)
    else # dim == 2
        x_out = map(inds -> x[:,inds],inds_vec)
    end
    return x_out
end
function Split_func(x::T,outputs::Int64, 
        dim::Int64) where T<:AbstractArray{<:AbstractFloat,4}
    if dim>2
        throw(BoundsError(x,dim))
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
struct Addition end
(m::Addition)(x) = sum(x)

# Activation layer
struct Activation
    f::Function
end
(m::Activation)(x) = m.f.(x)

# Identity layer
struct Identity
end
(m::Identity)(x) = x