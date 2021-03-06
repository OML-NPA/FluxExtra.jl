
using Flux, CUDA
path = joinpath(dirname(@__DIR__),"src","FluxExtra.jl")
include(path)
using .FluxExtra

function test(model::Chain,x::T,y::T) where T<:AbstractArray{<:AbstractFloat,4}
    losses = Vector{Float32}(undef,2)
    for i = 1:2
        local loss_val
        ps = Flux.Params(Flux.params(model))
        gs = gradient(ps) do
            predicted = model(x)
            loss_val = loss(predicted,y)
        end
        losses[i] = loss_val
        # Update weights
        Flux.Optimise.update!(opt,ps,gs)
    end
    if losses[1]==losses[2]
        ex = ErrorException("Gradient not updating")
    end
    return nothing
end

opt = Descent(0.1)
loss = Flux.Losses.mse
test_layer = Conv((3, 3), 1=>2,pad=SamePad())
test_layer2 = Chain(Conv((3, 3), 1=>2,pad=SamePad()))

# Test Parallel and Catenation layers
x = rand(Float32,6,6,1,1)
y = rand(Float32,6,6,4,1)
model = Chain(Parallel((test_layer,test_layer2)),Catenation(3))
test(model,x,y)
test(move(model,gpu),gpu(x),gpu(y))

# Test Decatenation layer
x = rand(Float32,6,6,2,1)
y = rand(Float32,6,6,4,1)
model = Chain(Decatenation(2,3),Parallel((test_layer,test_layer)),Catenation(3))
test(model,x,y)
test(move(model,gpu),gpu(x),gpu(y))

# Test Addition layer
x = rand(Float32,6,6,2,1)
y = rand(Float32,6,6,1,1)
model = Chain(Decatenation(2,3),Parallel((test_layer,test_layer)),Addition())
test(model,x,y)
test(move(model,gpu),gpu(x),gpu(y))

# Test Upscaling layer
x = rand(Float32,6,6,1,1)
# Dimension 1
y = rand(Float32,12,6,2,1)
model = Chain(test_layer,Upscaling(2,1))
test(model,x,y)
test(move(model,gpu),gpu(x),gpu(y))
# Dimension 2
y = rand(Float32,6,12,2,1)
model = Chain(test_layer,Upscaling(2,2))
test(model,x,y)
test(move(model,gpu),gpu(x),gpu(y))
# Dimension 1,2
y = rand(Float32,12,12,2,1)
model = Chain(test_layer,Upscaling(2,(1,2)))
test(model,x,y)
test(move(model,gpu),gpu(x),gpu(y))

# Dimension 3
y = rand(Float32,6,6,4,1)
model = Chain(test_layer,Upscaling(2,3))
test(model,x,y)
test(move(model,gpu),gpu(x),gpu(y))

# Test Activation layer
x = rand(Float32,4,4,1,1)
y = rand(Float32,4,4,2,1)
model = Chain(test_layer,Activation(tanh))
test(model,x,y)
test(move(model,gpu),gpu(x),gpu(y))

# Test Identity layer
x = rand(Float32,4,4,1,1)
y = rand(Float32,4,4,3,1)
model = Chain(Parallel((test_layer,Identity())),Catenation(3))
test(model,x,y)
test(move(model,gpu),gpu(x),gpu(y))