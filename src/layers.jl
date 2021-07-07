
# Join layer
"""
    Join(dim::Int64)
    Join(dim = dim::Int64)

Concatenates a tuple of arrays along a dimension `dim`. A convenient and type stable way of using `x -> cat(x..., dims = dim)`.
"""
struct Join{D}
    dim::Int64
    function Join(dim)
        if dim>3
            throw(DimensionMismatch("Dimension should be 1, 2 or 3."))
        end
        new{dim}(dim)
    end
    function Join(;dim)
        if dim>3
            throw(DimensionMismatch("Dimension should be 1, 2 or 3."))
        end
        new{dim}(dim)
    end
end
(m::Join{D})(x::NTuple{N,AbstractArray}) where {D,N} = cat(x...,dims = Val(D))

function Base.show(io::IO, l::Join)
    print(io, "Join(", "dim = ",l.dim, ")")
end


# Split layer
"""
    Split(outputs::Int64, dim::Int64)
    Split(outputs::Int64, dim = dim::Int64)

Breaks an array into a number of arrays which is equal to `outputs` along a dimension `dim`. `dim` should we divisible by `outputs` without a remainder.
"""
struct Split{O,D}
    outputs::Int64
    dim::Int64
    function Split(outputs,dim)
        if dim>3
            throw(DimensionMismatch("Dimension should be 1, 2 or 3."))
        end
        new{outputs,dim}(outputs,dim)
    end
    function Split(outputs;dim)
        if dim>3
            throw(DimensionMismatch("Dimension should be 1, 2 or 3."))
        end
        new{outputs,dim}(outputs,dim)
    end
end

function Split_func(x::T,m::Split{O,D}) where {O,D,T<:AbstractArray{<:AbstractFloat,2}}
    if D!=1
        throw(DimensionMismatch("Dimension should be 1."))
    end
    step_var = Int64(size(x, D) / O)
    f = i::Int64 -> (1+(i-1)*step_var):(i)*step_var
    vals = ntuple(i -> i, O)
    inds_vec = map(f,vals)
    x_out = map(inds -> x[inds,:],inds_vec)
    return x_out
end

function get_part(x::T,inds::NTuple{O,UnitRange{Int64}},d::Type{Val{1}}) where {O,T<:AbstractArray{<:AbstractFloat,4}}
    x_out = map(inds -> x[inds,:,:,:],inds)
end
function get_part(x::T,inds::NTuple{O,UnitRange{Int64}},d::Type{Val{2}}) where {O,T<:AbstractArray{<:AbstractFloat,4}}
    x_out = map(inds -> x[:,inds,:,:],inds)
end
function get_part(x::T,inds::NTuple{O,UnitRange{Int64}},d::Type{Val{3}}) where {O,T<:AbstractArray{<:AbstractFloat,4}}
    x_out = map(inds -> x[:,:,inds,:],inds)
end
function Split_func(x::T,m::Split{O,D}) where {O,D,T<:AbstractArray{<:AbstractFloat,4}}
    step_var = Int64(size(x, D) / O)
    f = i::Int64 -> (1+(i-1)*step_var):(i)*step_var
    vals = ntuple(i -> i, O)
    inds_tuple = map(f,vals)
    x_out = get_part(x,inds_tuple,Val{D})
    return x_out
end
(m::Split{O,D})(x::T) where {O,D,T<:AbstractArray} = Split_func(x,m)

function Base.show(io::IO, l::Split)
    print(io, "Split(", l.outputs,", dim = ",l.dim, ")")
end

# Makes Parallel layer type stable when used after Split
(m::Parallel)(xs::NTuple{N,AbstractArray}) where N = map((f,x) -> f(x), m.layers,xs)


# Addition layer
"""
    Addition()

A convenient way of using `x -> sum(x)`.
"""
struct Addition 
end
(m::Addition)(x::NTuple{N,AbstractArray}) where N = sum(x)


# Activation layer
"""
    Activation(f::Function)

A convenient way of using `x -> f(x)`.
"""
struct Activation{F}
    f::F
    Activation(f) = new{typeof(f)}(f)
end
(m::Activation)(x::AbstractArray) = m.f.(x)

function Base.show(io::IO, l::Activation)
    print(io, "Activation(",l.f, ")")
end


# Flatten layer
"""
    Flatten()

Flattens an array. A convenient way of using `x -> Flux.flatten(x)`.
"""
struct Flatten 
end
(m::Flatten)(x::AbstractArray) = Flux.flatten(x)


# Identity layer
"""
    Identity()

Returns its input without changes. Should be used with a `Parallel` layer if one wants to have a branch that does not change its input.
"""
struct Identity
end
(m::Identity)(x::AbstractArray) = x