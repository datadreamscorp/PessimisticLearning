using Random, Distributions, StatsBase, LaTeXStrings
using Base.Threads

g(x; l=0.75) = x == 1.0 ? 0.0 : exp( ( l*log(1 + x) ) + ( (1 - l)*log(1 - x) ) )

g_ruin(s; ℵ=0.5, u=0.55) = (1 - probruin(u, ℵ, s)) * g(s, l=u)

function probruin(u, ℵ, s; VB=1)
	V0 = VB/(1 - ℵ)
	
	# Compute increments
	a = log(1 + s)
	b = log(1 - s)
	
	# Mean and variance
	μ = u*a + (1-u)*b
	σ² = u*(a - μ)^2 + (1-u)*(b - μ)^2
	
	p_ruin = (VB / V0)^( (2*μ)/σ² )
	# Clamp for safety, though it should be in (0,1)
	return clamp(p_ruin, 0.0, 1.0)
end

probruin_numeric(u, א, s; Vb=1, seasons=500, n=1000) = 1 - ( (filter(x -> x > 0, [simulate_gambles_num(u, א, s, Vb=Vb, seasons=seasons) for i in 1:n]) |> length)/n )

s_star(u, ℵ) = (0.001:0.001:0.999|>collect)[findmax( (1 .- probruin.(u, ℵ, 0.001:0.001:0.999)) .* g.(0.001:0.001:0.999, l=u) )[2]]

s_star_numeric(u, ℵ) = (0.001:0.001:0.999|>collect)[findmax( (1 .- probruin_numeric.(u, ℵ, 0.001:0.001:0.999)) .* g.(0.001:0.001:0.999, l=u) )[2]]

function simulate_gambles_num(u, aleph, stake;
	Vb=1,
	log_benefit=0,
	seasons=1000,
	rounds=1,
	abarrier=true
	)

	if aleph < 1
		if aleph > 0
			if Vb > 0
				barrier = log(Vb)
			else
				abarrier = false
			end
			init_capital = log( Vb/(1 - aleph) )
		else
			error("only non-negative values of aleph allowed")
		end
	else
		abarrier = false
		init_capital = 1
	end
	
	log_capital = init_capital
	for i in 1:seasons

		rate = u# / rand( Pareto(λ) )

		for j in 1:rounds
			if rand() < rate
				log_capital = log(1 + stake) + log_benefit + log_capital
			else
				log_capital = log(1 - stake) + log_capital
			end
			if abarrier
				if log_capital < barrier
					log_capital = -Inf
					break
				end
			end
		end

		if abarrier
			if log_capital < barrier
				log_capital = -Inf
				break
			end
		end

	end

	return log_capital #- init_capital

end

function sim_payoffs(
	l, S, aleph;
	abarrier=true, 
	seasons=1000, 
	rounds=1, 
	n=10000
	)

	payoffs = ( [
			exp.( [simulate_gambles_num(l, aleph, s, abarrier=abarrier, seasons=seasons, rounds=rounds) for i in 1:n] ./ (seasons) )
			for s in S
		] )
	
	surv = [filter(v -> v > 1, p) for p in payoffs]
	fullmean = mean.(payoffs)
	survmean = mean.(surv)
	prob_surv = [length( filter(v -> v >= 1, p) ) / length(p) for p in payoffs]

	return(survmean, fullmean, prob_surv)
end