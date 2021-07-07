
using Flux, CUDA, Test, FluxExtra

function test_training(model,x,y)
    opt = ADAM()
    loss = Flux.Losses.mse
    losses = Vector{Float32}(undef,2)
    for i = 1:2
        local loss_val
        ps = Flux.Params(Flux.params(model))
        gs = gradient(ps) do
            predicted = model(x)
            loss_val = loss(predicted,y)
        end
        losses[i] = loss_val
        Flux.Optimise.update!(opt,ps,gs)
    end
    if losses[1]==losses[2]
        error("Parameters not updating.")
    end
    return nothing
end

function test(model,x,y)
    @inferred model(x)
    test_training(model,x,y)
    @inferred gpu(model)(gpu(x))
    test_training(gpu(model),gpu(x),gpu(y))
end

#---Convolution-----------------------------------------------------------

test_layer = Conv((3, 3), 1=>2,pad=SamePad())
test_layer2 = Conv((3, 3), 1=>2,pad=SamePad())

# Test Join layer
x = ones(Float32,6,6,1,1)
y = ones(Float32,12,6,2,1)
model = Chain(Parallel(tuple,(test_layer,test_layer2)),Join(1))
test(model,x,y)

x = ones(Float32,6,6,1,1)
y = ones(Float32,6,12,2,1)
model = Chain(Parallel(tuple,(test_layer,test_layer2)),Join(2))
test(model,x,y)

x = ones(Float32,6,6,1,1)
y = ones(Float32,6,6,4,1)
model = Chain(Parallel(tuple,(test_layer,test_layer2)),Join(dim = 3))
test(model,x,y)

try
    Join(4)
catch e
    if !(e isa DimensionMismatch)
        error("Wrong error returned.")
    end
end

try
    Join(dim = 4)
catch e
    if !(e isa DimensionMismatch)
        error("Wrong error returned.")
    end
end

Base.show(IOBuffer(),Join(1))


# Test Split layer
x = ones(Float32,6,6,1,1)
y = ones(Float32,6,6,2,1)
model = Chain(Split(2,dim = 1),Parallel(tuple,(test_layer,test_layer)),Join(1))
test(model,x,y)

x = ones(Float32,6,6,1,1)
y = ones(Float32,6,6,2,1)
model = Chain(Split(2,2),Parallel(tuple,(test_layer,test_layer)),Join(2))
test(model,x,y)

x = ones(Float32,6,6,2,1)
y = ones(Float32,6,6,4,1)
model = Chain(Split(2,3),Parallel(tuple,(test_layer,test_layer)),Join(3))
test(model,x,y)

try
    Split(2,4)
catch e
    if !(e isa DimensionMismatch)
        error("Wrong error returned.")
    end
end

try
    Split(2,dim = 4)
catch e
    if !(e isa DimensionMismatch)
        error("Wrong error returned.")
    end
end

try
    Split(1,1)
catch e
    if !(e isa DomainError)
        error("Wrong error returned.")
    end
end

Base.show(IOBuffer(),Split(2,3))


# Test Addition layer
x = ones(Float32,6,6,2,1)
y = ones(Float32,6,6,1,1)
model = Chain(Split(2,3),Parallel(tuple,(test_layer,test_layer)),Addition())
test(model,x,y)


# Test Activation layer
x = ones(Float32,4,4,1,1)
y = ones(Float32,4,4,2,1)
model = Chain(test_layer,Activation(tanh))
test(model,x,y)

# Test Flatten layer
x = ones(Float32,4,4,1,1)
y = ones(Float32,32,1)
model = Chain(test_layer,Flatten())
test(model,x,y)

Base.show(IOBuffer(),Activation(tanh))

# Test Identity layer
x = ones(Float32,4,4,1,1)
y = ones(Float32,4,4,3,1)
model = Chain(Parallel(tuple,(test_layer,Identity())),Join(3))
test(model,x,y)

#---Dense---------------------------------------------------------------

test_layer = Dense(2,3)
test_layer2 = Dense(2,3)

# Test Join layer
x = ones(Float32,2,1)
y = ones(Float32,6,1)
model = Chain(Parallel(tuple,(test_layer,test_layer2)),Join(1))
test(model,x,y)


# Test Split layer
x = ones(Float32,4,1)
y = ones(Float32,6,1)
model = Chain(Split(2,1),Parallel(tuple,(test_layer,test_layer)),Join(1))
test(model,x,y)

x = ones(Float32,2,1)
layer = Split(2,2)
try
    layer(x)
catch e
    if !(e isa DimensionMismatch)
        error("Wrong error returned.")
    end
end


# Test Addition layer
x = ones(Float32,4,1)
y = ones(Float32,3,1)
model = Chain(Split(2,1),Parallel(tuple,(test_layer,test_layer)),Addition())
test(model,x,y)


# Test Activation layer
x = ones(Float32,2,1)
y = ones(Float32,3,1)
model = Chain(test_layer,Activation(tanh))
test(model,x,y)


# Test Identity layer
x = ones(Float32,2,1)
y = ones(Float32,5,1)
model = Chain(Parallel(tuple,(test_layer,Identity())),Join(1))
test(model,x,y)
