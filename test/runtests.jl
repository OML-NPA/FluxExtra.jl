
using Flux, CUDA, Test, FluxExtra, FluxExtra.Normalizations

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
    return true
end

function test_model(model,x,y) 
    test_exp = quote
        @test begin
            @inferred $model($x) 
            true 
        end
        @test test_training($model,$x,$y)
        @test begin 
            @inferred gpu($model)(gpu($x)) 
            true 
        end
        @test test_training(gpu($model),gpu($x),gpu($y))
    end
    return test_exp
end


#---Tests begin-------------------------------------------------------------

@testset verbose=true "Conv as input" begin
    test_layer = Conv((3, 3), 1=>2,pad=SamePad())
    test_layer2 = Conv((3, 3), 1=>2,pad=SamePad())

    @testset "Join" begin
        @testset "Join(1)" begin
            x = ones(Float32,6,6,1,1)
            y = ones(Float32,12,6,2,1)
            model = Chain(Parallel(tuple,(test_layer,test_layer2)),Join(1))
            eval(test_model(model,x,y))
        end
        
        @testset "Join(2)" begin
            x = ones(Float32,6,6,1,1)
            y = ones(Float32,6,12,2,1)
            model = Chain(Parallel(tuple,(test_layer,test_layer2)),Join(2))
            eval(test_model(model,x,y))
        end
        
        @testset "Join(dum = 3)" begin
            x = ones(Float32,6,6,1,1)
            y = ones(Float32,6,6,4,1)
            model = Chain(Parallel(tuple,(test_layer,test_layer2)),Join(dim = 3))
            eval(test_model(model,x,y))
        end
        
        @testset "Join errors" begin
            @test try
                    Join(4)
                catch e
                    if !(e isa DimensionMismatch)
                        @error "Wrong error returned."
                        return e
                    else
                        true
                    end
                end

            @test try
                    Join(dim = 4)
                catch e
                    if !(e isa DimensionMismatch)
                        @error "Wrong error returned."
                        return e
                    else
                        true
                    end
                end
        end
        
        @testset "Printing" begin
            @test begin 
                Base.show(IOBuffer(),Join(1))
                true
            end
        end
    end

    @testset "Split" begin
        @testset "Split(_,1)" begin
            x = ones(Float32,6,6,1,1)
            y = ones(Float32,6,6,2,1)
            model = Chain(Split(2,1),Parallel(tuple,(test_layer,test_layer)),Join(1))
            eval(test_model(model,x,y))
        end
        
        @testset "Split(_,2)" begin
            x = ones(Float32,6,6,1,1)
            y = ones(Float32,6,6,2,1)
            model = Chain(Split(2,2),Parallel(tuple,(test_layer,test_layer)),Join(2))
            eval(test_model(model,x,y))
        end

        @testset "Split(_,dim = 3)" begin
            x = ones(Float32,6,6,2,1)
            y = ones(Float32,6,6,4,1)
            model = Chain(Split(2,dim = 3),Parallel(tuple,(test_layer,test_layer)),Join(3))
            eval(test_model(model,x,y))
        end

        @testset "Split errors" begin
            @test try
                    Split(2,4)
                catch e
                    if !(e isa DimensionMismatch)
                        @error "Wrong error returned."
                        return e
                    else
                        true
                    end
                end

            @test try
                    Split(2,dim = 4)
                catch e
                    if !(e isa DimensionMismatch)
                        @error "Wrong error returned."
                        return e
                    else
                        true
                    end
                end

            @test try
                    Split(1,1)
                catch e
                    if !(e isa DomainError)
                        @error "Wrong error returned."
                        return e
                    else
                        true
                    end
                end

            @test try
                    Split(1,dim = 1)
                catch e
                    if !(e isa DomainError)
                        @error "Wrong error returned."
                        return e
                    else
                        true
                    end
                end
        end

        @testset "Printing" begin
            @test begin
                Base.show(IOBuffer(),Split(2,3)) 
                true
            end
        end
    end

    @testset "Addition" begin
        x = ones(Float32,6,6,1,1)
        y = ones(Float32,6,6,1,1)
        model = Chain(Parallel(tuple,(test_layer,test_layer)),Addition())
        eval(test_model(model,x,y))
    end

    @testset "Activation" begin
        x = ones(Float32,4,4,1,1)
        y = ones(Float32,4,4,2,1)
        model = Chain(test_layer,Activation(tanh))
        eval(test_model(model,x,y))

        @testset "Printing" begin
            @test begin 
                Base.show(IOBuffer(),Activation(tanh))
                true
            end
        end
    end

    @testset "Flatten" begin
        x = ones(Float32,4,4,1,1)
        y = ones(Float32,32,1)
        model = Chain(test_layer,Flatten())
        eval(test_model(model,x,y))
    end

    @testset "Identity" begin
        x = ones(Float32,4,4,1,1)
        y = ones(Float32,4,4,3,1)
        model = Chain(Parallel(tuple,(test_layer,Identity())),Join(3))
        eval(test_model(model,x,y))
    end
end


@testset verbose=true "Dense as input" begin
    test_layer = Dense(2,3)
    test_layer2 = Dense(2,3)

    @testset "Join" begin
        @testset "Join(1)" begin
            x = ones(Float32,2,1)
            y = ones(Float32,6,1)
            model = Chain(Parallel(tuple,(test_layer,test_layer2)),Join(1))
            eval(test_model(model,x,y))
        end
    end

    @testset "Split" begin
        @testset "Split(_,1)" begin
            x = ones(Float32,4,1)
            y = ones(Float32,6,1)
            model = Chain(Split(2,1),Parallel(tuple,(test_layer,test_layer)),Join(1))
            eval(test_model(model,x,y))
        end

        @testset "Split errors" begin
            x = ones(Float32,2,1)
            layer = Split(2,2)
            @test try
                    layer(x)
                catch e
                    if !(e isa DimensionMismatch)
                        error("Wrong error returned.")
                        return e
                    else
                        true
                    end
                end
        end
    end

    @testset "Addition" begin
        x = ones(Float32,4,1)
        y = ones(Float32,3,1)
        model = Chain(Split(2,1),Parallel(tuple,(test_layer,test_layer)),Addition())
        eval(test_model(model,x,y))
    end

    @testset "Activation" begin
        x = ones(Float32,2,1)
        y = ones(Float32,3,1)
        model = Chain(test_layer,Activation(tanh))
        eval(test_model(model,x,y))
    end

    @testset "Identity" begin
        x = ones(Float32,2,1)
        y = ones(Float32,5,1)
        model = Chain(Parallel(tuple,(test_layer,Identity())),Join(1))
        eval(test_model(model,x,y))
    end
end

@testset "Normalizations" begin
    data = repeat([rand(Float32,5,5,3)],2)
    @test begin
        min_vals,max_vals = norm_01!(data)
        norm_01!(data,min_vals,max_vals)
        min_vals,max_vals = norm_negpos1!(data)
        norm_negpos1!(data,min_vals,max_vals)
        mean_vals = norm_zerocenter!(data)
        mean_vals,std_vals = norm_zscore!(data)
        norm_zscore!(data,mean_vals,std_vals)
        true
    end
end
