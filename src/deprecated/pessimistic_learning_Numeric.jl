using Random, Distributions, StatsBase, Plots
using Base.Threads


transform_one(a) = a == 0.0 ? 1.0 : a

function simulate_success_rates(; u=1.0, λ=1000, n=100)
	[
        u / rand( Pareto(λ) )
        for i in 1:n
    ]
end

#=
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
=#

function calculate_estimates(n, tries, λ)
	estimates = []
	for i in 1:n
		simmed_throws = [
			rand() < (1 / rand( Pareto(λ) ) ) ? 1 : 0
			for j in 1:tries
		]
		est = sum(simmed_throws) / length(simmed_throws)
		push!(estimates, est)
	end
	return estimates
end

#=
function calculate_estimates02(n, tries, sim_rates)
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
=#


calculate_stake(est) = 2*est - 1 > 0 ? 2*est - 1 : 0

expected_stake(λ) = ( 2*( λ*( 1 - ( 1/2^(λ+1) ) ) / ( (λ+1)*( 1 - (1/2^λ) ) ) ) - 1 ) * ( 1 - (1/2^λ) )
	
optimal_stake(λ) = 2*(λ/(1+λ)) - 1

optimal_fraction(λ) = 1 / ( 1 + ( 2^(-λ) / (λ - 1) ) )


stakemean(n, tries, λ) = calculate_stake( mean( calculate_estimates(n, tries, λ) ) )


meanstake(n, tries, λ) = mean( calculate_stake.( calculate_estimates(n, tries, λ) ) )


function simulate_gambles(u, λ, stake;
	init_capital=1,
	log_benefit=0,
	seasons=2000,
	rounds=1,
	abarrier=true
	)

	log_capital = init_capital
	for i in 1:seasons

		rate = u / rand( Pareto(λ) )

		for j in 1:rounds
			if rand() < rate
				log_capital = log(1 + stake) + log_benefit + log_capital
			else
				log_capital = log(1 - stake) + log_capital
			end
			if abarrier
				if log_capital < 0.0
					log_capital = -Inf
					break
				end
			end
		end

		if abarrier
			if log_capital < 0.0
				log_capital = -Inf
				break
			end
		end

	end

	return log_capital

end


sigmoid(x; b=0.1, a=-3) = 1 / (1 + exp(-( a + b*x )))

function simulate_gambles_adaptive(λ, stake, n, b, a;
	u=1.0,
	init_capital=1.0,
	log_benefit=0,
	seasons=2000,
	rounds=1,
	abarrier=true
	)

	stakes = repeat([stake], n)
	log_capitals = repeat([init_capital], n)
	
	for k in 1:n
		
		for i in 1:seasons

			rate = u / rand( Pareto(λ) )
		
			for j in 1:rounds
				if rand() < rate
					log_capitals[k] = log(1 + stakes[k]) + log_benefit + log_capitals[k]
				else
					log_capitals[k] = log(1 - stakes[k]) + log_capitals[k]
				end
				if abarrier
					if log_capitals[k] < 0.0
						log_capitals[k] = -Inf
						break
					end
				end
			end
	
			if abarrier
				if log_capitals[k] < 0.0
					log_capitals[k] = -Inf
					break
				end
			end

			stakes[k] = sigmoid(log_capitals[k], b=b, a=a)

		end

	end

	return (log_capitals, stakes)

end


function simulate_gambles_alt(λ, stake, n;
	sens=0.01,
	u=1.0,
	init_capital=1.0,
	log_benefit=0,
	seasons=2000,
	rounds=1,
	abarrier=true,
	succ_bias=true
	)

	sens = length(sens) == 1 ? repeat([sens], n) : sens

	stakes = repeat([stake], n)
	log_capitals = repeat([init_capital], n)

	for k in 1:n

		for i in 1:seasons

			rate = u / rand( Pareto(λ) )
		
			for j in 1:rounds
				if rand() < rate
					log_capitals[k] = log(1 + stakes[k]) + log_benefit + log_capitals[k]
				else
					log_capitals[k] = log(1 - stakes[k]) + log_capitals[k]
				end
				if abarrier
					if log_capitals[k] < 0.0
						log_capitals[k] = -Inf
						break
					end
				end
			end
	
			if abarrier
				if log_capitals[k] < 0.0
					log_capitals[k] = -Inf
					stakes = clamp.(stakes .- rand.(Exponential.(sens)), 0.0, 1.0)
					break
				end
			end

		end

	end
	
	if succ_bias
		log_capitals = filter(a -> exp(a) > 0, log_capitals)
	end

	return (log_capitals, stakes)

end


function simulate_gambles_alt(λ, stake, n;
	sens=0.01,
	u=1.0,
	init_capital=1.0,
	log_benefit=0,
	seasons=2000,
	rounds=1,
	abarrier=true,
	succ_bias=true
	)

	sens = length(sens) == 1 ? repeat([sens], n) : sens

	stakes = repeat([stake], n)
	log_capitals = repeat([init_capital], n)

	for i in 1:seasons

		results = [
			rand() < u / rand( Pareto(λ) ) ? log(1 + stakes[k]) : log(1 - stakes[k]) 
			for k in 1:n
		]
		
		log_capitals = log_capitals .+ results
	
		if abarrier
			
			for c in log_capitals
				if c != -Inf
					if c < 0.0
						c = -Inf
						stakes = clamp.(stakes .- rand.(Exponential.(sens)), 0.0, 1.0)
					end
				end
			end
		end

	end

	
	if succ_bias
		dead = findall([c -> c == -Inf, log_capitals])
		log_capitals = log_capitals[1:end .!= dead]
		stakes = stakes[1:end .!= dead]
	end

	return (log_capitals, stakes)

end


#=

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

=#


g(x; l=5.0) = x == 1.0 ? 0.0 : exp( ( ( l / (l+1) )*log(1 + x) ) + ( (1 - ( l / (l+1) ) )*log(1 - x) ) )

geo_mean(λ, s; u=1.0, cap=1, seasons=2000, rounds=1) = exp( simulate_gambles(u, λ, s, init_capital=cap, seasons=seasons, rounds=1) ./ seasons )

function get_power_samples(λ, times, n)
	all_times = []
	estimates = []
	for T in times
		for i in 1:n
			est = 0
			for t in 1:T
				u = 1 / rand( Pareto(λ) )
				est += rand() < u ? 1 : 0
			end
			est = 2*(est / T) - 1
			est = est < 0 ? 0.0 : est
			push!(estimates, est)
			push!(all_times, T)
		end
	end
	return (all_times, estimates)
end

#=

function sim_payoffs(λ, S; u=1.0, seasons=2000, rounds=1, tries=10, init_capital=0, abarrier=false, n=10000)
	
	payoffs = [
		exp.( [simulate_gambles(u, λ, s, seasons=seasons, rounds=rounds, init_capital=init_capital, abarrier=abarrier) for i in 1:n] ./ seasons )
		for s in S
	]
	
	surv = [filter(v -> v >= 1, p) for p in payoffs]

	mean_payoffs = [mean(p) for p in payoffs]
	surv_payoffs = mean.(surv)
	prop_surv = length.(surv) ./ length.(payoffs)

	return (mean_payoffs, surv_payoffs, prop_surv)
	
end
=#


function sim_payoffs(
	l, S, cap; u=1.0, 
	abarrier=true, seasons=2000, rounds=1, n=1000
	)

	payoffs = ( [
			exp.( [simulate_gambles(u, l, s, init_capital=cap, abarrier=abarrier, seasons=seasons, rounds=rounds) for i in 1:n] ./ seasons )
			for s in S
		] )
	
	surv = [filter(v -> v > 1, p) for p in payoffs]
	fullmean = mean.(payoffs)
	survmean = mean.(surv)
	prob_surv = [length( filter(v -> v >= 1, p) ) / length(p) for p in payoffs]

	return(survmean, fullmean, prob_surv)
end


function simpop(
	λ, c; 
	α=1, β=1, n=1000, 
	T = 2000, n_samples=1,
	w_constant=false,  w=0.5, 
	succ_bias=true
	)

	betaweights = w_constant ? w : rand( Beta(α,β), n )
	
	pop = []
	counter = 1
	for d in betaweights
		if w_constant
			for s in 1:n_samples
				pay = sim_payoffs(
					λ, d, c, 
					n=1, seasons=T
					)[2]
				if succ_bias
					if pay > 0.0
						push!(pop, (d, pay))
					end
				else
					push!(pop, (d, pay))
				end
			end
			counter += 1
		else
			pay = sim_payoffs(
				λ, d, c, 
				n=1, seasons=T
				)[2]
			if succ_bias
				if pay > 0.0
					push!(pop, (d, pay))
				end
			else
				push!(pop, (d, pay))
			end
			counter += 1
		end
	end

	return (betaweights, pop)
	
end


penalty_func(w, med; S=0.05) = exp( (-( w - med )^2)/ S )


function compare_conformity(λ, c; opt_surv=50, var=0.25, T=2000, w_constant=false, w=0:0.05:1|>collect, n=1000, n_samples=1)
	mu = (0:0.01:1|>collect)[opt_surv]
	var = clamp(var, 0, mu*(1 - mu)-eps())
	α = ( ( (1 - mu) / var ) - ( 1 / mu ) ) * mu^2 
	β = α * ( ( 1 / mu ) - 1 )

	betaweights, pop = simpop(λ, c, T=T, α=α, β=β, w_constant=w_constant, w=w, n=n, n_samples=n_samples)
	med = median([a[1] for a in pop])
	noconf = [(a[1], penalty_func(a[1], med, S=10)*a[2]) for a in pop]
	conf = [(a[1], penalty_func(a[1], med, S=0.01)*a[2]) for a in pop]

	return(noconf, conf, pop)
	
end


allcombinations(v...) = vec(collect(Iterators.product(v...)))


function get_seeds(v1, v2; seed=12345)
	combs = allcombinations(v1, v2)
	combs = zip( combs, rand(Xoshiro(seed), 1:100000, length(combs)) )
	Dict(
		[
			(comb[1][1],comb[1][2]) => comb[2] 
			for comb in combs
				]...
	)
end


function plot_powerdist(λ; n=1000, lw=3, legend=false)
	U = [ 1/rand( Pareto(λ) ) for i in 1:n ]
	
	hist = histogram(
		U,
		legend=legend,
		xlab="λ = $λ",
		normalize=true,
		color="white",
		titlefontsize=10
	)
	plot!(
		0:0.001:1,
		λ.*(0:0.001:1).^(λ - 1),
		lw=3,
		dpi=300,
		normalize=true,
		color="black"
	)
	vline!([mean(U)], lw=lw, color="green")

	return hist
end


function estimate_plots(λ, tries, n; title=false, label=false)
	
	estimates = calculate_estimates(n, tries, λ)
	estimated_stakes = calculate_stake.(estimates)
	
	estimateplot = histogram(
		estimates, 
		legend=false, 
		dpi=300, 
		title= title ? "success rate estimates (ū)" : "",
		color="white",
		ylab="λ = $λ"
	)
	vline!(
		[mean(estimates)], 
		#labels= label ? "mean" : "", 
		lw=3, 
		color="grey",
		alpha=0.75
	)
	
	stakeplot = histogram(
		estimated_stakes,
		legend=:topright,
		title= title ? "estimated stakes (s̄)" : "",
		label="",
		#color="light blue",
		bins=50,
		size=(700,500),
		dpi=300,
		color="white"
	)
	vline!(
		[mean(estimated_stakes)], 
		label= label ? "mean" : "",
		lw=3, 
		color="black",
		ls=:dash,
		alpha=0.75
	)
	vline!(
		[2*mean(estimates)-1],
		label= label ? "optimal stake" : "",
		lw=3,
		color="grey",
		alpha=0.75
	)
	
	return [estimateplot, stakeplot]
	
end


function simplots(λ, s, abarrier, n, init_capital, legend, xlab)

	barrier_payoffs = [
	sim_payoffs(l, s, init_capital, abarrier=abarrier, n=n)[2]
	for l in λ
	]
	
	sim_plot = plot( 
		s, 
		barrier_payoffs[1],
		label="$(λ[1])",
		legendtitle="λ",
		legendfontsize=6,
		legendtitlefontsize=8,
		lw=2,
		alpha=0.5,
		ylab="ν = $init_capital",
		xlab=xlab ? "stake (s)" : "",
		ylim=(0.9, 1.15),
		legend=legend
	)

	b_counter = 2
	for p in barrier_payoffs[2:length(barrier_payoffs)]

		plot!( 
			s, 
			barrier_payoffs[b_counter],
			label="$(λ[b_counter])",
			lw=2,
			alpha=0.5
		)
		scatter!(
			(s[findmax(barrier_payoffs[b_counter])[2]],
			findmax(barrier_payoffs[b_counter])[1]),
			label="",
			color="black"
		)
		b_counter += 1
		
	end

	scatter!(
		(s[findmax(barrier_payoffs[1])[2]],
		findmax(barrier_payoffs[1])[1]),
		label="",
		color="black"
	)
	hline!([1.0], color="black", alpha=0.2, label="")
	
	return sim_plot
	
end


function plot_delta_payoffs(delta_payoffs, vals; legend=false, mincap=0, maxcap=5, minl=1.5)
	xaxis = 0:0.01:1
	#max_payoff = findmax(delta_payoffs[1].*delta_payoffs[3])
	#max_prob = findmin(transform_one.(delta_payoffs[2] .- delta_payoffs[1]))
	
	plot(
		xaxis,
		delta_payoffs[2],
		lw=2,
		color="black",
		legend=legend,
		legendfontsize=5,
		label="mean payoff",
		#ylim=(0.9, findmax(delta_payoffs[2,1][2])[1]+0.01),
		alpha=0.5,
		xlab=vals[2]==mincap ? "λ = $(vals[1])" : ( vals[2]==maxcap ? "stake (s)" : "" ),
		guide_position=vals[2]==mincap ? :top : :bottom,
		ylab=vals[1]==minl ? "ν = $(vals[2])" : "",
		
	)
	plot!(
		xaxis,
		delta_payoffs[1],
		lw=2,
		ls=:dash,
		color="black",
		label="mean success-biased payoff"
	)
	
end


function plot_meanconf(λ, C)
	vals = (1:length(λ)|>collect, λ, C)
	v_barrier = [(i, j) for i in vals[2], j in vals[3]]
	
	deltas = [ 
		α
		for α in 0:0.01:1, l in vals[2]
	]
	
	delta_payoffs = [
		sim_payoffs(
		vals[2][l], 
		0:0.01:1,
		vals[3][C],
		)
		for l in vals[1], C in vals[1]
	]

	minl = first(vals[2])
	mincap = first(vals[3])
	maxcap = last(vals[3])

	plot( 
		vec(plot_delta_payoffs.(delta_payoffs, v_barrier))..., 
		layout=(3,3), 
		link=:all, 
		ylim=(0.9, 1.3),
		plot_title="mean payoff under demographic filtering"
	)
end


function scatterconf(noconf, conf, pop, opt_surv; legend=false, xlab="", ylab="", gp=:top, alpha=0.05)
	
	weightspop = [w[1] for w in pop]
	poppay = [c[2] for c in pop]
	maxpop = findmax(poppay)

	weightsnoconf = [w[1] for w in noconf]
	noconf = [c[2] for c in noconf]
	maxnoconf = findmax(noconf)

	weightsconf = [w[1] for w in conf]
	conf = [c[2] for c in conf]
	maxconf = findmax(conf)
	
	confplot = scatter(
			weightspop,
			poppay,
			alpha=alpha,
			xlim=(0,1),
			ylim=(0.95, maxnoconf[1]+0.01),
			markershape=:circle,
			color="black",
			label="",
			legend=legend,
			xlab=xlab,
			guide_position=gp,
			ylab=ylab
		)

	plot!(
		[weightsnoconf[maxnoconf[2]], weightsnoconf[maxnoconf[2]]],
		[-1.0, maxnoconf[1]],
		lw=2,
		ls=:dot,
		color="black",
		alpha=0.7,
		label="pure payoff",
		legendfontsize=6
	)

	plot!(
		[weightsconf[maxconf[2]], weightsconf[maxconf[2]]],
		[-1.0, maxconf[1]],
		color="black",
		alpha=0.5,
		lw=2,
		label="pure conformity"
	)

	plot!(
		[(0:0.01:1)[opt_surv], (0:0.01:1)[opt_surv]],
		[0.0, 0.96],
		lw=3, color="dark blue", alpha=1.0, label="")
	
	return confplot

end


function plot_scatterconf(λ, c; T=5000, n=1000, var=0.2, w_constant=false, w=0:0.05:1|>collect, n_samples=1, T_opt=5000, legend=true, xlab=false, ylab=false, gp=:top, alpha=0.05)
	opt_surv = findmax(sim_payoffs(λ, 0:0.01:1, c, seasons=T_opt)[2])[2]
	noconf, conf, pop = compare_conformity(λ, c, opt_surv=opt_surv, var=var, T=T, n_samples=n_samples, n=n, w_constant=w_constant, w=w)
	scatterconf(noconf, conf, pop, opt_surv, legend=legend, xlab=xlab, ylab=ylab, gp=gp, alpha=alpha)
end

