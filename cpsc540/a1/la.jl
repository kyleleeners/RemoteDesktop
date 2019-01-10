include("misc.jl")
using MathProgBase, GLPKMathProgInterface

function leastAbsolutes(X,y)

	y = reshape(y, length(y))

	# Add bias column
	n = size(X,1)
	Z = [X ones(n,1)]
	f = size(Z,2)

	# Set up constraint matrix
	A = [Z -I; -Z -I]
	b = [y; -y]

	# set up coefficent vector.
	c = [zeros(f); ones(n)]

	# set up constraint vectors.
	d = -Inf*ones(2*n)
	lb = -Inf*ones(f+n)
	ub = Inf*ones(f+n)

	#Solve the LP
	solution = linprog(c,A,d,b,lb,ub, GLPKSolverLP())
	z = solution.sol

	#Get Parameters.
	w = z[1:f]

	# Make linear prediction function
	predict(Xtilde) = [Xtilde ones(size(Xtilde,1),1)]*w

	# Return model
	return LinearModel(predict,w)
end
