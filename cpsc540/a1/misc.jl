using Random

mutable struct LinearModel
	predict # Funcntion that makes predictions
	w # Weight vector
end

# Return squared Euclidean distance all pairs of rows in X1 and X2
function distancesSquared(X1,X2)
	(n,d) = size(X1)
	(t,d2) = size(X2)
	@assert(d==d2)
	return X1.^2*ones(d,t) + ones(n,d)*(X2').^2 - 2X1*X2'
end

### A function to compute the gradient numerically
function numGrad(func,x)
	n = length(x);
	delta = 2*sqrt(1e-12)*(1+norm(x));
	g = zeros(n);
	e_i = zeros(n)
	for i = 1:n
		e_i[i] = 1;
		(fxp,) = func(x + delta*e_i)
		(fxm,) = func(x - delta*e_i)
		g[i] = (fxp - fxm)/2delta;
		e_i[i] = 0
	end
	return g
end

### Check if number is a real-finite number
function isfinitereal(x)
	return (imag(x) == 0) & (!isnan(x)) & (!isinf(x))
end

## Compute gaussian rbf of given dataset with given variance
function rbfBasis(Xi, Xj, σ)
    return exp.(-distancesSquared(Xi,Xj) / 2σ)
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
