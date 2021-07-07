
# Join layer
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
struct Addition 
end
(m::Addition)(x) = sum(x)

# Activation layer
struct Activation{F}
    f
    Activation(f) = new{f}(f)
end
(m::Activation{F})(x) where F = F.(x)

function Base.show(io::IO, l::Activation)
    print(io, "Activation(",l.f, ")")
end

# Flatten layer
struct Flatten 
end
(m::Flatten)(x) = Flux.flatten(x)

# Identity layer
struct Identity
end
(m::Identity)(x) = x