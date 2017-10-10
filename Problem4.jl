
# Just to make sure you have installed all the packages that you will need in Lab-3
for p in ("Knet","ArgParse","Compat","GZip")
    Pkg.installed(p) == nothing && Pkg.add(p)
end

using Knet   
using ArgParse # To work with command line argumands
using Compat,GZip # Helpers to read the MNIST (Like lab-2)
using Knet: relu_dot

function main(args="")
    #=
    In the macro, options and positional arguments are specified within a begin...end block
    by one or more names in a line, optionally followed by a list of settings. 
    So, in  the below, there are five options: epoch,batchsize,hidden size of mlp, 
    learning rate, weight initialization constant
    =#
    s = ArgParseSettings()
    @add_arg_table s begin
        ("--epochs"; arg_type=Int; default=10; help="number of epoch ")
        ("--batchsize"; arg_type=Int; default=100; help="size of minibatches")
        ("--hidden"; nargs='*'; arg_type=Int; help="sizes of hidden layers, e.g. --hidden 128 64 for a net with two hidden layers")
        ("--lr"; arg_type=Float64; default=0.5; help="learning rate")
        ("--winit"; arg_type=Float64; default=0.1; help="w initialized with winit*randn()")
    end

    #=
    the actual argument parsing is performed via the parse_args function the result 
    will be a Dict{String,Any} object.In our case, it will contain the keys "epochs", 
    "batchsize", "hidden" and "lr", "winit" so that e.g. o["lr"] or o[:lr] 
     will yield the value associated with the positional argument.
     For more information: http://argparsejl.readthedocs.io/en/latest/argparse.html
    =#
    o = parse_args(s; as_symbols=true)

    # Some global configs do not change here 
    println("opts=",[(k,v) for (k,v) in o]...)
    o[:seed] = 123
    srand(o[:seed])

    # initalize weights of your model 
    w = weights(o[:hidden]; winit=o[:winit])

    # load the mnist data
    xtrnraw, ytrnraw, xtstraw, ytstraw = loaddata()
    
    xtrn = convert(Array{Float32}, reshape(xtrnraw ./ 255, 28*28, div(length(xtrnraw), 784)))
    ytrnraw[ytrnraw.==0]=10;
    ytrn = convert(Array{Float32}, sparse(convert(Vector{Int},ytrnraw),1:length(ytrnraw),one(eltype(ytrnraw)),10,length(ytrnraw)))
    
    xtst = convert(Array{Float32}, reshape(xtstraw ./ 255, 28*28, div(length(xtstraw), 784)))
    ytstraw[ytstraw.==0]=10;
    ytst = convert(Array{Float32}, sparse(convert(Vector{Int},ytstraw),1:length(ytstraw),one(eltype(ytstraw)),10,length(ytstraw)))
    # seperate it into batches. 
    dtrn = minibatch(xtrn, ytrn, o[:batchsize]) 
    dtst = minibatch(xtst, ytst, o[:batchsize])

    # helper function to see how your training goes on.
    report(epoch)=println((:epoch,epoch,:trn,accuracy(w,dtrn),:tst,accuracy(w,dtst)))
    
    report(0)    
    # Main part of our training process 
    @time for epoch=1:o[:epochs]
        train(w, dtrn,o[:lr])
        report(epoch)
    end
    return w
end

# Create weight matrix that will be used in our MLP.
# your weights should be  Float32 type.
function weights(h;winit=0.1)
    w = Any[]    
    x = 28*28
    for y in [h..., 10]
        push!(w, convert(Array{Float32}, winit*randn(y,x)))
        push!(w, convert(Array{Float32}, zeros(y, 1)))
        x = y
    end
    return w
end

# Create the minibatches on Mnist data. You will do exactly the same thing you did in Lab-2
#takes raw input (X) and gold labels (Y)
#returns list of minibatches (x, y)
function minibatch(X, Y,bs)
    #takes raw input (X) and gold labels (Y)
    #returns list of minibatches (x, y)
    cols_X = size(X,1)
    rows_X = size(X,2)
    data = Any[]
    # MY CODE HERE
    for batch_start=1:bs:rows_X-bs+1
        batch_end=batch_start+bs-1
        push!(data,(X[:,batch_start:batch_end],Y[:,batch_start:batch_end]))
    end
    #MY CODE ENDS HERE
    return data
end

#=
predict function takes model parameters (w) and data (x) and 
makes forward calculation. Fill below function accordingly. It
should return #ofclasses x batchsize size matrix as a result. 
Use ReLU nonlinearty at each layer except the last one. 
=#
function predict(w,x)
    for i=1:2:length(w)
        x = w[i]*x .+ w[i+1]
        if i<length(w)-1
            x = relu_dot(x)
        end
    end
    return x
end

#=
loss function takes model parameters(w), a batch of instance (x) and
their gold labels. Please fill the loss function so that it returns 
the loss based on your predictions and gold data
Hint : You may want to use predict function here to get your predictions 
Hint2: Take a loot at logp function defined in knet
=#
function loss(w,x,ygold)
    ypred = predict(w,x)
    #logp returns normalized log probabilities
    ylogp_norm = logp(ypred,1)
    -sum(ygold .* ylogp_norm) / size(ygold,2) #sum(abs2,ypred-ygold)
end

#=
  As you notice, we did not do anything to calculate gradients of our loss function.
  We know that Knet will do it on behalf of us. However, we need to provide our loss function
  to Knet so that it can record all necessary operations. The line below does this.
=#
#Just doing a gradient on this is enough
lossgradient = grad(loss)  # your code here [just 1 line]

#=
This is the main function you need to use to train your model.
It takes model of parameter, all training data, learning rate and 
epoch which determines the number of times you will go through your
whole data. Inside this function, you need to decide your prediction
,measure the loss between  gold labels and your predictions. And finally
update your weights with the gradients Knet gives  you based on the loss.
Hint: You did everything required in this function (by implementing loss and predict functions) except the last step. Here you just need to call higher order procedure given you by Knet and update the weights accordingly.
=#
function train(w, dtrn,lr)
    for (x,y) in dtrn
        g = lossgradient(w, x, y)
        for i in 1:length(w)
            w[i] = (-lr)*g[i]+w[i]
        end
    end
    return w
end

#=
Function for measuring current accuracy of the model.  Again you need to find your predictions first and then measure how many of them correct. You also need to measure average loss your model does at test dataset after each epoch.
=#
using Compat
import Compat.String

function accuracy(w,dtst,pred=predict)
    total_rows = 0
    got_right = 0
    got_wrong = 0
    for (x, ygold) in dtst
        ypred = pred(w, x)
        ynorm = logp(ypred,1) 
        got_wrong += -sum(ygold .* ynorm)
        got_right += sum(ygold .* (ypred .== maximum(ypred,1)))
        total_rows += size(ygold,2)
    end
    return (got_right/total_rows, got_wrong/total_rows)
end
function loaddata()
	info("Loading MNIST...")
	xtrn = gzload("train-images-idx3-ubyte.gz")[17:end]
	xtst = gzload("t10k-images-idx3-ubyte.gz")[17:end]
	ytrn = gzload("train-labels-idx1-ubyte.gz")[9:end]
	ytst = gzload("t10k-labels-idx1-ubyte.gz")[9:end]
	return (xtrn, ytrn, xtst, ytst)
end

function gzload(file; path="$file", url="http://yann.lecun.com/exdb/mnist/$file")
	isfile(path) || download(url, path)
	f = gzopen(path)
	a = @compat read(f)
	close(f)
	return(a)
end
main()

# Implementation details

# For the default configuration 

# min epoch correctly predicted labels accuracy: 0.9049f0
# max epoch correctly predicted labels accuracy: 0.9151f0
# min epoch wrongly predicted labels accuracy: 0.29300820530955446
# max epoch wrongly predicted labels accuracy: 0.3238402942176069


# With hidden size --32 

# min epoch correctly predicted labels accuracy: 0.9282f0
# max epoch correctly predicted labels accuracy: 0.9595f0
# min epoch wrongly predicted labels accuracy: 0.12928239247059078
# max epoch wrongly predicted labels accuracy: 0.21917876140175835


# With hidden size --64

# min epoch correctly predicted labels accuracy: 0.9383f0
# max epoch correctly predicted labels accuracy: 0.9682f0
# min epoch wrongly predicted labels accuracy: 0.10537905940245612
# max epoch wrongly predicted labels accuracy: 0.19680877156377527


# With hidden size --64 64 64 64

# min epoch correctly predicted labels accuracy: 0.1161f0
# max epoch correctly predicted labels accuracy: 0.968f0
# min epoch wrongly predicted labels accuracy: 0.18178857633542703
# max epoch wrongly predicted labels accuracy: 2.3000147f0

# 2.1 Hidden layers vs Performance

# We can observe that increasing the number of hidden layers doesn't increase the accuracy
# In fact the perfomance can worsen as we can see from the results our training. In some epochs the accuracy 
# has fallen to min 0.116 for correctly predicted labels. 

# 2.2 Configurations vs Overfitting

# Overfitting is when the model fits the training data so well that it performs badly on the test/new
# dataset. I looked at the implementation results to find the instance in which there is a decrease 
# in accuracy between training and testing.And for the configuration 64 64 64 64 hidden size, there 
# is a significantly bigger difference in accuracy during training and testing compared to rest
# of the configurations. Having way too many hidden layers could result in overfitting

# 2.3 Learning rate set to 4.0

# I noticed that setting learning rate to 4.0 has improved my training time to 47 seconds while rest
# of the configurations took around 200s or 150s . But the training rate has dropped significantly
# to around 89% compared to 95+% or above in the rest of the configurations

# Intuition for why this might be the case : Seems like learning rate during the training step
# is intended as a way to determine how fast the model should react to a weird data outlier
# in the training dataset.The higher the learning rate the faster (the more erratic) the model 
# is during training. Which is a good thing when we want the model to be sensitive to data changes
# I think we want a good balance between a low learning rate and a high learning rate
# so that there is a good balance between the model finishing training to a good accuracy and 
# also being able to be responsive

