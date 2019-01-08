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

function leastSquaresRBFL2(X,y, σ, λ)
    
    # create rbf
    rbf = rbfBasis(X, σ)
    
    # Add bias column
	n = size(rbf,1)
	Z = [ones(n,1) rbf]
    
    # Idenity * lambda
    I = Diagonal(ones(n+1,n+1)) * λ
    
    # Find regression weights minimizing squared error
    w = (Z'*Z - I)\(Z'*y)
    
    # Make linear prediction function
    println(size(w))
    println(size(Z))
    
    predict(Xtilde) = [ones(size(Xtilde,1),1) Xtilde]*w
    
	# Return model
	return LinearModel(predict,w)
end
