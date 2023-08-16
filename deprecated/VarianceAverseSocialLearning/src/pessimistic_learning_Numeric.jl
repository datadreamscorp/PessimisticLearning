using Distributions
using Agents
using Base.Threads

function calculate_kelly(estimated_p)
    k = 2*estimated_p - 1
    k > 0 ? k : 0
end

function simulate_success_rates(; u=0.7, λ=1000, n=100)
    [
        u / rand( Pareto(λ) )
        for i in 1:n
    ]
end


function rdeu_power(array::Vector, δ::Number; w=[])

	l = length(array)

	if length(w) == 0
		w = repeat( [1/l], l )
	end

	sorted = sort(array, rev=true)

	rdeu_mean = 0

	@threads for i in 1:l
		if i == 1
			ρ = w[1]^δ
		else
			ρ = sum( w[1:i] )^δ - sum( w[1:(i-1)] )^δ
		end
		rdeu_mean += ρ*sorted[i]
	end

	return rdeu_mean

end


function rdeu_linear(array::Vector, δ::Number; w=[])

	l = length(array)

	if length(w) == 0
		w = repeat( [1/l], l )
	end

	sorted = sort(array, rev=true)

	rdeu_mean = 0

	for i in 1:l
		if i == 1
			ρ = w[1]^δ
		else
			ρ = sum( w[1:i] )^δ - sum( w[1:(i-1)] )^δ
		end
		rdeu_mean += ρ*sorted[i]
	end

	return rdeu_mean

end


function calculate_estimates(n, tries, sim_rates)
	estimates = []
	for i in 1:n
		simmed_throws = [
			rand() < sim_rates[i] ? 1 : 0
			for j in 1:tries
		]
		est = sum(simmed_throws) / length(simmed_throws)
		push!(estimates, est)
	end
	return estimates
end


function calculate_estimates02(n, tries, λ)
	estimates = []
	for i in 1:n
		simmed_throws = [
			rand() < (1 / rand(Pareto(λ))) ? 1 : 0
			for j in 1:tries
		]
		est = sum(simmed_throws) / length(simmed_throws)
		push!(estimates, est)
	end
	return estimates
end


calculate_stake(est) = 2*est - 1 > 0 ? 2*est - 1 : 0


function simulate_gambles(u, λ, stake;
	log_capital=0,
	log_benefit=0,
	seasons=100,
	rounds=10
)
	for i in 1:seasons
		rate = u / rand( Pareto(λ) )
		for j in 1:rounds
			if rand() < rate
				log_capital = log(1 + stake) + log_benefit + log_capital
			else
				log_capital = log(1 - stake) + log_capital
			end
		end
	end

	return log_capital

end

function simulate_gambles02(u, λ, stake;
	log_capital=0,
	log_benefit=0,
	seasons=100,
	rounds=10
)
	for i in 1:seasons
		rate = u / rand( Pareto(λ) )
		if rand() < rate
			log_capital = log(1 + stake) + log_benefit + log_capital
		else
			log_capital = log(1 - stake) + log_capital
		end
	end

	return log_capital

end
