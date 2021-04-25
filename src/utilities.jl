
# Move model between CPU and GPU
function move(model,target::Union{typeof(cpu),typeof(gpu)})
    model_moved = []
    if model isa Chain
        for i = 1:length(model)
            # If model branches out, then apply function also to each branch
            if model[i] isa Parallel
                layers = model[i].layers
                new_layers = Array{Any}(undef,length(layers))
                for i = 1:length(layers)
                    new_layers[i] = move(layers[i],target)
                end
                new_layers = (new_layers...,)
                push!(model_moved,target(Parallel(tuple,new_layers)))
            else
                push!(model_moved,target(model[i]))
            end
        end
    else
        push!(model_moved,target(model))
    end
    # If model contains more than one layer, then form a chain
    if length(model_moved)==1
        model_moved = model_moved[1]
    else
        model_moved = target(Chain(model_moved...))
    end
    return model_moved
end