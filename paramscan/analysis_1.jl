#############
#ANALYSIS 1
@everywhere using Pkg
@everywhere Pkg.activate("..")

@everywhere include("../src/pessimistic_learning_ABM.jl")

@everywhere using CSV, Distributed
@everywhere using Agents, Random, Distributions, Statistics, StatsBase

@everywhere total_ticks = 0

@everywhere begin #INCLUDE MODEL CODE AND NECESSARY LIBRARIES

	parameters = Dict( #ALTER THIS DICTIONARY TO DEFINE PARAMETER DISTRIBUTIONS
    :N => 10000,
    :n => [5, 25],
    :T => [1000],
    :t => [2, 25],
    :λ => [1.5, 3.0, 6.0],
    :aleph => [0.65, 0.8, 0.95],
    :init_sens => 0.0:0.005:0.3|>collect,
    :seed => 1:5|>collect
)

	mdata = [
        :opt_s, 
        :opt_payoff, 
        :s_median, 
        :s_lerror, 
        :s_herror, 
        :Vbar
        ]

end

#USE THIS LINE AFTER DEFINITIONS TO BEGIN PARAMETER SCANNING
_, mdf = paramscan(
            parameters, initialize_pessimistic_learning;
            mdata=mdata,
            n = total_ticks,
			parallel=true,
			when_model = [total_ticks],
			showprogress = true
	)

CSV.write("../data/analysis_1.csv", mdf)
