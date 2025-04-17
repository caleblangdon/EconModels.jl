module EconModels

using Distributions
using UnPack
using ProgressMeter
using Tullio
using Plots
using LinearAlgebra
using DataFrames

abstract type Params end
abstract type Model end
abstract type EstimationProcedure end

parameters(m::Model) = m.p


include("ggv.jl")
export ParamsGGV, ModelGGV, solve!

function test_func()
    println("Test 2")
end
export test_func

end
