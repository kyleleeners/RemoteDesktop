using Pkg
Pkg.add("JLD")
Pkg.add("PyPlot")
Pkg.build("HDF5")

include("example_nonLinear.jl")
include("example_multiClass.jl")
