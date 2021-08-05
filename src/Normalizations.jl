
module Normalizations

using Statistics

function norm_range!(data::T,min_vals::T, max_vals::T,new_min::F,new_max::F) where {F<:AbstractFloat,N,T<:Array{F,N}}
    data .= ((data .- min_vals)./(max_vals .- min_vals)).*(new_max .- new_min) .+ new_min
    return nothing
end

function norm_range!(data::Vector{T},new_min::F,new_max::F) where {F<:AbstractFloat,N,T<:Array{F,N}}
    num = size(data[1],N)
    min_vals = T(undef,ntuple(x->1,Val(N-1))...,num)
    max_vals = T(undef,ntuple(x->1,Val(N-1))...,num)
    for i = 1:num
        min_vals[i] = minimum(cat(selectdim.(data, N, i)...,dims=Val(N)))
        max_vals[i] = maximum(cat(selectdim.(data, N, i)...,dims=Val(N)))
    end
    map(x -> norm_range!(x,min_vals, max_vals,new_min,new_max), data)
    return min_vals, max_vals
end


"""
    norm_01!(data::T,min_vals::T, max_vals::T) where {F<:AbstractFloat,N,T<:Array{F,N}}

Rescales each feature (last dimension) to be in the range [0,1].
"""
function norm_01!(data::T,min_vals::T, max_vals::T) where {F<:AbstractFloat,N,T<:Array{F,N}}
    norm_range!(data,min_vals, max_vals,zero(F),one(F))
    return nothing
end

"""
    norm_01!(data::Vector{T}) where {F<:AbstractFloat,N,T<:Array{F,N}}

Rescales each feature (last dimension) to be in the range [0,1]. Returns min and max values for each feature.
"""
function norm_01!(data::Vector{T}) where {F<:AbstractFloat,N,T<:Array{F,N}}
    min_vals, max_vals = norm_range!(data,zero(F),one(F))
    return min_vals, max_vals
end



"""
    norm_negpos1(data::T,min_vals::T,max_vals::T) where {F<:AbstractFloat,N,T<:Array{F,N}}

Rescales each feature (last dimension) to be in the range [-1,1].
"""
function norm_negpos1!(data::T,min_vals::T,max_vals::T) where {F<:AbstractFloat,N,T<:Array{F,N}}
    norm_range!(data,min_vals, max_vals,-one(F),one(F))
    return nothing
end

"""
    norm_negpos1(data::Vector{T}) where {F<:AbstractFloat,N,T<:Array{F,N}}

Rescales each feature (last dimension) to be in the range [-1,1]. Returns min and max values for each feature.
"""
function norm_negpos1!(data::Vector{T}) where {F<:AbstractFloat,N,T<:Array{F,N}}
    min_vals, max_vals = norm_range!(data,-one(F),one(F))
    return min_vals, max_vals
end


"""
    norm_zerocenter!(data::T,mean_vals::T) where {F<:AbstractFloat,N,T<:Array{F,N}}

Subtracts the mean of each feature (last dimension).
"""
function norm_zerocenter!(data::T,mean_vals::T) where {N,T<:Array{<:AbstractFloat,N}}
    data .= data .- mean_vals
    return nothing
end

"""
    norm_zerocenter!(data::Vector{T}) where {F<:AbstractFloat,N,T<:Array{F,N}}

Subtracts the mean of each feature (last dimension). Returns a mean value for each feature.
"""
function norm_zerocenter!(data::Vector{T}) where {N,T<:Array{<:AbstractFloat,N}}
    num = size(data[1],N)
    mean_vals = T(undef,ntuple(x->1,Val(N-1))...,num)
    for i = 1:num
        mean_vals[i] = mean(cat(selectdim.(data, N, i)...,dims=Val(N)))
    end
    map(x -> norm_zerocenter!(x,mean_vals), data)
    return mean_vals
end


"""
    norm_zscore!(data::T,mean_vals::T,std_vals::T) where {F<:AbstractFloat,N,T<:Array{F,N}}

Subtracts the mean and divides by the standard deviation of each feature (last dimension).
"""
function norm_zscore!(data::T,mean_vals::T,std_vals::T) where {N,T<:Array{<:AbstractFloat,N}}
    data .= (data .- mean_vals)./std_vals
    return nothing
end

"""
    norm_zscore!(data::Vector{T}) where {F<:AbstractFloat,N,T<:Array{F,N}}

Subtracts the mean and divides by the standard deviation of each feature (last dimension). 
Returns mean and standard deviation values for each feature.
"""
function norm_zscore!(data::Vector{T}) where {N,T<:Array{<:AbstractFloat,N}}
    num = size(data[1],N)
    mean_vals = T(undef,ntuple(x->1,Val(N-1))...,num)
    std_vals = T(undef,ntuple(x->1,Val(N-1))...,num)
    for i = 1:num
        mean_vals[i] = mean(cat(selectdim.(data, N, i)...,dims=Val(N)))
        std_vals[i] = std(cat(selectdim.(data, N, i)...,dims=Val(N)))
    end
    map(x -> norm_zscore!(x,mean_vals,std_vals), data)
    return mean_vals, std_vals
end


export norm_01!, norm_negpos1!, norm_zerocenter!, norm_zscore!

end