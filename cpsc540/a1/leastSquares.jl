include("misc.jl")
using LinearAlgebra
using MathProgBase, GLPKMathProgInterface

function leastAbsolutes(X,y)

	# resphape y
	y = reshape(y, length(y))

	# Add bias column
	n = size(X,1)
	Z = [ones(n,1) X]
	m = size(Z,2)

	# prepare LP
	A = [Z -I ; -Z -I]
	c = [zeros(m); ones(n)]
	b = [y ; -y]
	d = -Inf*ones(size(A,1))
	lb = -Inf*ones(n+m)
	ub = Inf*ones(n+m)

	# run through solver
	solution = linprog(c, A, d, b, lb, ub, GLPKSolverLP())
	w = solution.sol[1:m]

	# Make linear prediction function
	predict(Xtilde) = [ones(size(Xtilde,1),1) Xtilde]*w

	# Return model
	return LinearModel(predict,w)
end

function leastMax(X,y)

	# resphape y
	y = reshape(y, length(y))

	# Add bias column
	n = size(X,1)
	Z = [ones(n,1) X]
	m = size(Z,2)
	res = -ones(size(y))

	# prepare LP
	A = [Z res; -Z res]
	c = [zeros(m); 1]
	b = [y ; -y]
	d = -Inf*ones(size(b))

	# run through solver
	solution = linprog(c, A, d, b, -Inf, Inf, GLPKSolverLP())
	w = solution.sol[1:m]

	# Make linear prediction function
	predict(Xtilde) = [ones(size(Xtilde,1),1) Xtilde]*w

	# Return model
	return LinearModel(predict,w)
end

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

## Compute gaussian rbf of given dataset with given variance
function rbfBasis(Xi, Xj, σ)
    return exp.(-distancesSquared(Xi,Xj) / 2σ)
end

function leastSquaresRBFL2CV(X,y)

    # split data into training and test
    Xtrain, ytrain, Xvalid, yvalid = partitionTrainTest(X,y)

    # find best hyperparams
    σ,λ = gridSearch(Xtrain, ytrain, Xvalid, yvalid)

    # print σ,λ
    @printf "σ = %f, λ = %f\n" σ λ

    return leastSquaresRBFL2(Xtrain,ytrain,σ,λ)
end

## Randomly split data
function partitionTrainTest(data, y, train_perc = 0.7)
    n = size(data,1)
    mid = Int(ceil(n/2))
    idx = collect(1:n)
    rand_idx = idx[randperm(length(idx))]
    trainIdxs = rand_idx[1:mid]
    testIdxs = rand_idx[mid+1:end]
    return data[trainIdxs,:], y[trainIdxs,:], data[testIdxs,:], y[testIdxs,:]
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
			t = size(Xtest,1)
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
