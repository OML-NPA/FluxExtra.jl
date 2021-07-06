
# Join layer
struct Join # Not type stable inside
    dim::Int64
    function Join(dim)
        if dim>3
            throw(DimensionMismatch("Dimension should be 1, 2 or 3."))
        end
        new(dim)
    end
end
(m::Join)(x::NTuple{N,T}) where {T<:AbstractArray,N} = cat(x...,dims = Val(m.dim))::T

struct Join1 # Type stable
end
(m::Join1)(x::NTuple{N,T}) where {T<:AbstractArray,N} = vcat(x...)

struct Join2 # Type stable
end
(m::Join2)(x::NTuple{N,T}) where {T<:AbstractArray,N} = hcat(x...)

struct Join3 # Type stable
end
(m::Join3)(x::NTuple{N,T}) where {T<:AbstractArray,N} = cat(x...,dims = Val(3))


# Split layer
struct Split
    outputs::Int64
    dim::Int64
    function Split(outputs,dim)
        if dim>3
            throw(DimensionMismatch("Dimension should be 1, 2 or 3."))
        end
        new(outputs,dim)
    end
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