#############
#ANALYSIS 2 ALT
@everywhere using Pkg
@everywhere Pkg.activate("..")
@everywhere using PessimisticLearning, Agents, CSV

@everywhere total_ticks = 10

@everywhere begin #INCLUDE MODEL CODE AND NECESSARY LIBRARIES

	parameters = Dict( #ALTER THIS DICTIONARY TO DEFINE PARAMETER DISTRIBUTIONS
    :N => 10000,
    :n => [25],
    :T => [1000],
    :t => [25],
    :λ => [1.5, 3.0, 6.0],
    :aleph => 0.05:0.05:0.95|>collect,
    :init_sens => [0.1],
    :strategies => ["UB"],
    :init_soc_v => 0:0.05:1|>collect,
    :mu_soc_v => [0.0],
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
			when_model = 0:total_ticks|>collect,
			showprogress = true
	)

CSV.write("../data/analysis_2_5.csv", mdf)
