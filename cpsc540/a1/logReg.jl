include("misc.jl")
include("findMin.jl")

function logReg(X,y)

	(n,d) = size(X)

	# Initial guess
	w = zeros(d,1)

	# Function we're going to minimize (and that computes gradient)
	funObj(w) = logisticObj(w,X,y)

	# Solve least squares problem
	w = findMin(funObj,w,derivativeCheck=true)

	# Make linear prediction function
	predict(Xhat) = sign.(Xhat*w)

	# Return model
	return LinearModel(predict,w)
end

function logisticObj(w,X,y)
	yXw = y.*(X*w)
	f = sum(log.(1 .+ exp.(-yXw)))
	g = -X'*(y./(1 .+ exp.(yXw)))
	return (f,g)
end

# Multi-class one-vs-all version (assumes y_i in {1,2,...,k})
function logRegOnevsAll(X,y)
	(n,d) = size(X)
	k = maximum(y)

	# Each column of 'w' will be a logistic regression classifier
	W = zeros(d,k)

	for c in 1:k
		yc = ones(n,1) # Treat class 'c' as +1
		yc[y .!= c] .= -1 # Treat other classes as -1

		# Each binary objective has the same features but different lables
		funObj(w) = logisticObj(w,X,yc)

		W[:,c] = findMin(funObj,W[:,c],verbose=false)
	end

	# Make linear prediction function
	predict(Xhat) = mapslices(argmax,Xhat*W,dims=2)

	return LinearModel(predict,W)
end

function softmaxClassifier(X,y)
    
end

function softmaxObj(w,X,y)
    XW = dot(X,W)
    Z = sum(exp.(XW))
    
    f = -sum(XW[y] - log.(Z))
end


    def funObj(self, w, X, y):
        n, d = X.shape
        k = self.n_classes

        W = np.reshape(w, (d, k))

        y_binary = np.zeros((n, k)).astype(bool)
        y_binary[np.arange(n), y] = 1

        XW = np.dot(X, W)
        Z = np.sum(np.exp(XW), axis=1)

        # Calculate the function value
        f = - np.sum(XW[y_binary] - np.log(Z))

        # Calculate the gradient value
        g = X.T.dot(np.exp(XW) / Z[:,np.newaxis] - y_binary)

        return f, g.ravel()

    def fit(self,X, y):
        n, d = X.shape
        k = np.unique(y).size

        self.n_classes = k
        self.w = np.zeros(d*k)

        # Initial guess
        utils.check_gradient(self, X, y)
        (self.w, f) = findMin.findMin(self.funObj, self.w.ravel(),
                                      self.maxEvals, X, y, verbose=self.verbose)

        self.w = np.reshape(self.w, (d, k))

    def predict(self, X):
        yhat = np.dot(X, self.w)

        return np.argmax(yhat, axis=1)