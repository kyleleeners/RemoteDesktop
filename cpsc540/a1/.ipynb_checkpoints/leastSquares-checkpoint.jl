include("misc.jl")
using LinearAlgebra

function leastSquares(X,y)

	# Add bias column
	n = size(X,1)
	Z = [ones(n,1) X]

	# Find regression weights minimizing squared error
	w = (Z'*Z)\(Z'*y)

	# Make linear prediction function
	predict(Xtilde) = [ones(size(Xtilde,1),1) Xtilde]*w

	# Return model
	return LinearModel(predict,w)
end

function leastSquaresRBFL2(X,y,σ,λ)
    
    # create rbf
    rbf = rbfBasis(X,X,σ)
    
    # Add bias column
	n = size(rbf,1)
	Z = [ones(n,1) rbf]
    
    # Find regression weights minimizing squared error
    w = (Z'*Z + I*λ)\(Z'*y)
    
    # Make linear prediction function
    predict(Xtilde) = [ones(size(Xtilde,1),1) rbfBasis(Xtilde,X,σ)]*w
    
	# Return model
	return LinearModel(predict,w)
end

function leastSquaresRBFL2CV(X,y)
    
    # split data into training and test
    Xtrain, ytrain, Xvalid, yvalid = partitionTrainTest(X,y) 
  
    # find best hyperparams
    σ,λ = gridSearch(Xtrain, ytrain, Xvalid, yvalid)
    
    # print σ,λ
    @printf "σ = %f ,λ = %f\n" σ λ
    
    return leastSquaresRBFL2(Xtrain,ytrain,σ,λ)
end

function gridSearch(Xtrain,ytrain,Xtest,ytest)
    
    # set defaults
    bestError = typemax(Int32)
    bestLamb = nothing
    bestSig = nothing
    
    # iterate over possible values of σ,λ
    for curSig in exp10.(range(-2, stop=2, length=10))
        for curLamb in exp10.(range(-3, stop=3, length=10))
            
            # compute weights
            model = leastSquaresRBFL2(Xtrain,ytrain,curSig,curLamb)
            
            # calculate error
            yhat = model.predict(Xtest)
            curError = sum((yhat - ytest).^2)/t
            
            # update defaults
            if curError < bestError 
                bestError = curError
                bestLamb = curLamb
                bestSig = curSig
            end
        end
    end
    return bestSig, bestLamb
end

