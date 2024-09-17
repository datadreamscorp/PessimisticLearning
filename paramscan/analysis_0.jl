#############
#ANALYSIS 5
@everywhere using Pkg
@everywhere Pkg.activate("..")
#@everywhere Pkg.instantiate()

@everywhere include("../src/pessimistic_learning_ABM.jl")

@everywhere using CSV, Distributed
@everywhere using Agents, Random, Distributions, Statistics, StatsBase

@everywhere total_ticks = 0

@everywhere begin #INCLUDE MODEL CODE AND NECESSARY LIBRARIES

	parameters = Dict( #ALTER THIS DICTIONARY TO DEFINE PARAMETER DISTRIBUTIONS
    :N => 10000,
    :n => [5, 10, 25],
    :T => [1000],
    :t => [2, 25],
    :λ => [1.5, 3.0, 6.0],
    :aleph => [0.65, 0.8, 0.95],
    :init_sens => [0.0, 0.1],
    :init_soc_h => 0.0:0.05:1.0|>collect,
    :seed => 1:5|>collect
)

	mdata = [
        :Vbar
        ]

end

#USE THIS LINE AFTER DEFINITIONS TO BEGIN PARAMETER SCANNING
_, mdf = paramscan(
            parameters, initialize_pessimistic_learning;
            mdata=mdata,
            n = total_ticks,
			parallel=true,
			when_model = 0:total_ticks|>collect,
			showprogress = true
	)

CSV.write("../data/analysis_0.csv", mdf)
