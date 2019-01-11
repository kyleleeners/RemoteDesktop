# Load X and y variable
using JLD, Statistics, Printf, PyPlot

function reportErrorAndPlot(model, X, Xtest, y, ytest, modelName)
    # Evaluate training error
    yhat = model.predict(X)
    trainError = mean((yhat - y).^2)
    @printf("Squared train Error with %s: %.3f\n",modelName, trainError)

    # Evaluate test error
    yhat = model.predict(Xtest)
    testError = mean((yhat - ytest).^2)
    @printf("Squared test Error with %s: %.3f\n\n",modelName, testError)

    # Plot model
    figure()
    plot(X,y,"b.")
    Xhat = minimum(X):.01:maximum(X)
    yhat = model.predict(Xhat)
    plot(Xhat,yhat,"g")
    show()
end

data = load("outliersData.jld")
(X,y,Xtest,ytest) = (data["X"],data["y"],data["Xtest"],data["ytest"])

# Fit a least squares model
include("leastSquares.jl")
leastSquaresModel = leastSquares(X,y)
leastAbsolutesModel = leastAbsolutes(X,y)
leastMaxModel = leastMax(X,y)

reportErrorAndPlot(leastSquaresModel, X, Xtest, y, ytest, "least squares")
reportErrorAndPlot(leastAbsolutesModel, X, Xtest, y, ytest, "least absolutes")
reportErrorAndPlot(leastMaxModel, X, Xtest, y, ytest, "least max")
