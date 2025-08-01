#############
#ANALYSIS 0 - NO ELDER INFLUENCE
using Pkg
Pkg.activate("..")

using Distributed
addprocs(80)

@everywhere begin 
    
    using PessimisticLearning, Agents, CSV, Random
    
    total_gens = 2500

    seeds = rand(Xoshiro(465826433645), 1:100000, 25)

	parameters = Dict( #ALTER THIS DICTIONARY TO DEFINE PARAMETER DISTRIBUTIONS
    :N => 750,
    :n => 10,
    :T => 500,
    :t => 10,
    :u => [0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
    :aleph => 0.05:0.05:0.95|>collect,
    :mu_std => 0.01,
    :mu_soc_h => 0.001,
    :mu_soc_v => 0.0,
    :mu_sens => 0.001,
    :mu_L => 0.0,
    :strategies => ["UB"],
    :selection => true,
    :mixed => false,
    :seed => seeds
)

	mdata = [
        :Vbar,
        :soc_h_median,
        :soc_h_lerror,
        :soc_h_herror,
        :sens_median,
        :sens_lerror,
        :sens_herror,
        :s_median,
        :s_mean,
        :s_lerror, 
        :s_herror,
        :s_ltail,
        :s_htail,
        :s_young_median,
        :s_young_lerror, 
        :s_young_herror,
        :sbar,
        :mean_increment,
        :concentration,
        :concentration_lerror,
        :concentration_herror
        ]

end

#USE THIS LINE AFTER DEFINITIONS TO BEGIN PARAMETER SCANNING
_, mdf = paramscan(
            parameters, initialize_pessimistic_learning;
            mdata = mdata,
            n = total_gens,
			parallel = true,
			when_model = [total_gens],
			showprogress = true
	)

    
CSV.write("../data/analysis_0.csv", mdf)
