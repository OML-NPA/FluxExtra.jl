
# Join layer
struct Join{D}
    dim::Int64
    function Join(dim)
        if dim>3
            throw(DimensionMismatch("Dimension should be 1, 2 or 3."))
        end
        new{dim}(dim)
    end
end
(m::Join{D})(x::NTuple{N,T}) where {D,N,T<:AbstractArray} = cat(x...,dims = Val(D))


# Split layer
struct Split{D}
    outputs::Int64
    dim::Int64
    function Split(outputs,dim)
        if dim>3
            throw(DimensionMismatch("Dimension should be 1, 2 or 3."))
        end
        new{outputs}(outputs,dim)
    end
end
function Split_func(x::T,m::Split{D}) where {D,T<:AbstractArray{<:AbstractFloat,2}}
    dim = m.dim
    if dim!=1
        throw(DimensionMismatch("Dimension should be 1."))
    end
    step_var = Int64(size(x, dim) / D)
    f = i::Int64 -> (1+(i-1)*step_var):(i)*step_var
    vals = ntuple(i -> i, D)
    inds_vec = map(f,vals)
    x_out = map(inds -> x[inds,:],inds_vec)
    return x_out
end
function Split_func(x::T,m::Split{D}) where {D,T<:AbstractArray{<:AbstractFloat,4}}
    dim = m.dim
    step_var = Int64(size(x, dim) / D)
    f = i::Int64 -> (1+(i-1)*step_var):(i)*step_var
    vals = ntuple(i -> i, D)
    inds_vec = map(f,vals)
    if dim == 1
        x_out = map(inds -> x[inds,:,:,:],inds_vec)
        return x_out
    elseif dim == 2
        x_out = map(inds -> x[:,inds,:,:],inds_vec)
        return x_out
    else # dim == 3
        x_out = map(inds -> x[:,:,inds,:],inds_vec)
        return x_out
    end
end
(m::Split{D})(x::T) where {D,T<:AbstractArray} = Split_func(x,m)

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