using Pkg
Pkg.add("JLD")
Pkg.add("PyPlot")
Pkg.build("HDF5")
Pkg.add("MathProgBase")
Pkg.add("GLPKMathProgInterface")

include("example_nonLinear.jl")
include("example_multiClass.jl")
include("example_outliers.jl")
