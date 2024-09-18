using Random, Distributions, StatsBase, LaTeXStrings
using Base.Threads

g(x; l=5.0) = x == 1.0 ? 0.0 : exp( ( ( l / (l+1) )*log(1 + x) ) + ( (1 - ( l / (l+1) ) )*log(1 - x) ) )

transform_one(a) = a == 0.0 ? 1.0 : a

sigmoid(x; b=0.1, a=-3) = 1 / (1 + exp(-( a + b*x )))

function simulate_success_rates(; u=1.0, λ=1000, n=100)
	[
        u / rand( Pareto(λ) )
        for i in 1:n
    ]
end

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

calculate_stake(est) = 2*est - 1 > 0 ? 2*est - 1 : 0

expected_stake(λ) = ( 2*( λ*( 1 - ( 1/2^(λ+1) ) ) / ( (λ+1)*( 1 - (1/2^λ) ) ) ) - 1 ) * ( 1 - (1/2^λ) )
	
optimal_stake(λ) = 2*(λ/(1+λ)) - 1

optimal_fraction(λ) = 1 / ( 1 + ( 2^(-λ) / (λ - 1) ) )

stakemean(n, tries, λ) = calculate_stake( mean( calculate_estimates(n, tries, λ) ) )

meanstake(n, tries, λ) = mean( calculate_stake.( calculate_estimates(n, tries, λ) ) )

function probruin(λ, א, s)
	u = λ/(λ+1)
	μ = u*log(1+s) + (1-u)*log(1-s)
	σ²= ( u*(log(1+s) - μ)^2 ) + ( (1-u)*(log(1-s) - μ)^2 )

	return clamp( (1 - א)^(2*μ/(σ² + μ^2)), 0, 1 ) 
end

probruin_numeric(λ, א, s; seasons=10000, n=10000) = 1 - ( (filter(x -> x != -Inf, [simulate_gambles(λ, א, s, seasons=seasons) for i in 1:n]) |> length)/n )

function simulate_gambles_num(λ, aleph, stake;
	u=1,
	Vb=1,
	log_benefit=0,
	seasons=2000,
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

		rate = u / rand( Pareto(λ) )

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

	return log_capital - init_capital

end

function get_power_samples(λ, times, n)
	estimates = []
	for T in times
		est_T = []
		for i in 1:n
			est = 0
			for t in 1:T
				u = 1 / rand( Pareto(λ) )
				est += rand() < u ? 1 : 0
			end
			est = 2*(est / T) - 1
			est = est < 0 ? 0.0 : est
			push!(est_T, est)
		end
		push!(estimates, est_T)
	end
	return estimates
end

function sim_payoffs(
	l, S, aleph;
	abarrier=true, 
	seasons=1000, 
	rounds=1, 
	n=1000
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


function simpop(
	λ, aleph; 
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
					λ, d, aleph, 
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

#=
function pull_stake(stakes, sens)
	for i in 1:length(stakes)
		if sens[i] > 0.0
			stakes[i] = clamp(
				stakes[i] - rand( Exponential(sens[i]) ),
				0.0, 1.0
			)
		end
	end
	return stakes
end

function simulate_gambles_sensitive(λ, stake, n;
	sens=0.05,
	u=1.0,
	init_capital=1.0,
	seasons=2000,
	rounds=1,
	abarrier=true,
	succ_bias=false
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
			
			for i in 1:n
				if log_capitals[i] != -Inf
					if log_capitals[i] < 0.0
						log_capitals[i] = -Inf
						stakes = pull_stake(stakes, sens)
					end
				end
			end
		end

	end

	
	if succ_bias
		dead = findall(c -> c == -Inf, log_capitals)
		log_capitals = [log_capitals[i] for i in 1:n if i ∉ dead]
		stakes = [stakes[i] for i in 1:n if i ∉ dead]
	end

	return (log_capitals, stakes)

end
=#

#=
penalty_func(w, med; S=0.05) = exp( (-( w - med )^2)/ S )

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
=#

#=
function plot_payoff_soclearn(cultrans, n, ν, sens, T, t; seed=1, legend=false)
	miao1 = cultrans[cultrans.strategies .== "CB" .&& cultrans.n .== n .&& cultrans.ν .== ν .&& cultrans.init_sens .== sens .&& cultrans.T .== T .&& cultrans.seed .== seed .&& cultrans.t .== t .&& cultrans.time .== 0, :]
	
	miao2 = cultrans[cultrans.strategies .== "CB" .&& cultrans.n .== n .&& cultrans.ν .== ν .&& cultrans.init_sens .== sens .&& cultrans.T .== T .&& cultrans.seed .== seed .&& cultrans.t .== t .&& cultrans.time .== 10, :]

	miao3 = cultrans[cultrans.strategies .== "PB" .&& cultrans.n .== n .&& cultrans.ν .== ν .&& cultrans.init_sens .== sens .&& cultrans.T .== T .&& cultrans.seed .== seed .&& cultrans.t .== t .&& cultrans.time .== 10, :]

	miao4 = cultrans[cultrans.strategies .== "UL" .&& cultrans.n .== n .&& cultrans.ν .== ν .&& cultrans.init_sens .== sens .&& cultrans.T .== T .&& cultrans.seed .== seed .&& cultrans.t .== t .&& cultrans.time .== 10, :]
	
	payplot = plot(
		miao1.λ,
		miao1.Vbar,
		c="black", alpha=0.75, ls=:dash, lw=2,
		legend=false, xticks=1.5:0.5:5.0, xrotation=90,
		title="ν = $(ν)",
		ylabel=ν == 1 ? "v̄" : ""
	)
	plot!(
		miao4.λ,
		miao4.Vbar,
		c="pink", lw=2
	)
	plot!(
		miao2.λ,
		miao2.Vbar,
		c="purple", alpha=0.75, lw=2,
	)
	plot!(
		miao3.λ,
		miao3.Vbar,
		c="dark blue", alpha=0.75, lw=2,
	)
	plot!(
		miao3.λ,
		miao3.opt_payoff,
		c="black", ls=:dot
	)

	return payplot
end

function run_ABM_mixed_strategy_plot(;lamb = [2.5, 6.0], nu = [1, 3],  mixed_freq=0.5, mixed_L=[1,3], mixed_λ_shift=[6.0, 6.0], mixed_ν_shift=[1, 3], t = 2, envshift = 10, lambda_shift = 6.0, n = 25, mu=0.0, T=100, total=20, parochial=true,
	legend=:bottomright)
	
		model1 = initialize_pessimistic_learning(;
			N=10000, n=n, seed=654123,  init_sens=0.1, 
			mixed_λ=lamb, mixed=true, mixed_ν=nu, 
			mixed_ν_shift=mixed_ν_shift, mixed_freq=mixed_freq, 
			t=t, mixed_L=mixed_L, envshift=envshift, 
			mixed_λ_shift=mixed_λ_shift, mu_soc_v=0.0, 
			init_soc_v=0.5, T=T, parochial=false
		)
	
		_, mdata1 = run!(
			model1,
			total, mdata=[:opt_s, :opt_payoff, :s_median, :s_lerror, :s_herror, :Vbar, :Vbar_conf, :Vbar_g0, :Vbar_g1]
		)
	
		model2 = initialize_pessimistic_learning(;
			N=10000, n=n, seed=654123,  init_sens=0.1, 
			mixed_λ=lamb, mixed=true, mixed_ν=nu, 
			mixed_ν_shift=mixed_ν_shift, mixed_freq=mixed_freq, 
			t=t, mixed_L=mixed_L, envshift=envshift, 
			mixed_λ_shift=mixed_λ_shift, mu_soc_v=0.0, 
			init_soc_v=1.0, T=T, parochial=false
		)
	
		_, mdata2 = run!(
			model2,
			total, mdata=[:opt_s, :opt_payoff, :s_median, :s_lerror, :s_herror, :Vbar, :Vbar_conf, :Vbar_g0, :Vbar_g1]
		)
	
		model3 = initialize_pessimistic_learning(;
			N=10000, n=n, seed=654123,  init_sens=0.1, 
			mixed_λ=lamb, mixed=true, mixed_ν=nu, 
			mixed_ν_shift=mixed_ν_shift, mixed_freq=mixed_freq, 
			t=t, mixed_L=mixed_L, envshift=envshift, 
			mixed_λ_shift=mixed_λ_shift, mu_soc_v=0.0, 
			init_soc_v=0.5, T=T, parochial=parochial
		)
	
		_, mdata3 = run!(
			model3,
			total, mdata=[:opt_s, :opt_payoff, :s_median, :s_lerror, :s_herror, :Vbar, :Vbar_conf, :Vbar_g0, :Vbar_g1]
		)
	
		model4 = initialize_pessimistic_learning(;
			N=10000, n=n, seed=654123,  init_sens=0.1, 
			mixed_λ=lamb, mixed=true, mixed_ν=nu, 
			mixed_ν_shift=mixed_ν_shift, mixed_freq=mixed_freq, 
			t=t, mixed_L=mixed_L, envshift=envshift, 
			mixed_λ_shift=mixed_λ_shift, mu_soc_v=0.0, 
			init_soc_v=1.0, T=T, parochial=parochial
		)
	
		_, mdata4 = run!(
			model4,
			total, mdata=[:opt_s, :opt_payoff, :s_median, :s_lerror, :s_herror, :Vbar, :Vbar_conf, :Vbar_g0, :Vbar_g1]
		)
	
		timeplot = plot(
			mdata1.time,
			mdata1.Vbar_g0,
			color=palette(:Dark2_5)[3],
			lw=2,
			legend=legend,
			legendfontsize=6,
			label="mixed learning (lower class)",
			xticks=0:maximum(mdata1.time),
			xlabel="time",
			ylabel="average growth rate"
		)
		plot!(
			mdata2.time,
			mdata2.Vbar_g0,
			color=palette(:Dark2_5)[1],
			#ls=:dot,
			lw=2,
			label="fully conservative (lower class)"
		)
		plot!(
			mdata3.time,
			mdata3.Vbar_g0,
			color=palette(:Dark2_5)[3],
			ls=:dash,
			lw=2,
			label="mixed parochial learning (lower class)"
		)
		plot!(
			mdata4.time,
			mdata4.Vbar_g0,
			color=palette(:Dark2_5)[1],
			ls=:dashdotdot,
			lw=2,
			label="conservative parochial (lower class)"
		)
		plot!(
			mdata1.time,
			mdata1.Vbar_g1,
			color=palette(:Dark2_5)[2],
			#ls=:dash,
			lw=2,
			label="mixed payoff bias (high class)"
		)
		plot!(
			mdata2.time,
			mdata2.Vbar_g1,
			color=palette(:Dark2_5)[4],
			#ls=:dash,
			lw=2,
			label="conservative payoff bias (high class)"
		)
		plot!(
			mdata3.time,
			mdata3.Vbar_g1,
			color=palette(:Dark2_5)[2],
			ls=:dash,
			lw=2,
			label="mixed parochial (high class)"
		)
		plot!(
			mdata4.time,
			mdata4.Vbar_g1,
			color=palette(:Dark2_5)[4],
			ls=:dashdotdot,
			lw=2,
			label="conservative parochial (high class)"
		)
		vline!([envshift], c="black", lw=1, alpha=0.5, label="")
		hline!([1], c="black", lw=1, ls=:dash, alpha=0.5, label="")
		
		return timeplot
	end

	function run_ABM_mixed_plot(;lamb = [2.5, 6.0], nu = [1, 3],  mixed_freq=0.5, t = 2, lambda_shift = 6.0, n = 25, mu=0.0, T=1000, total=10)
	
	model_CB = initialize_pessimistic_learning(;
		N=10000, n=n, seed=654123,  init_sens=0.1, 
		mixed_λ=lamb, mixed=true, mixed_ν=nu, mixed_freq=mixed_freq, t=t, strategies="UB", lambda_shift=lambda_shift, mu_soc_v=0.0, T=T
	)
	model_PB = initialize_pessimistic_learning(;
		N=10000, n=n, seed=654123,  init_sens=0.1, 
		mixed_λ=lamb, mixed=true, mixed_ν=nu, mixed_freq=mixed_freq, t=t, strategies="PB", lambda_shift=lambda_shift, mu_soc_v=0.0, T=T
	)
	model_UL = initialize_pessimistic_learning(;
		N=10000, n=n, seed=654123,  init_sens=0.1, 
		mixed_λ=lamb, mixed=true, mixed_ν=nu, mixed_freq=mixed_freq, t=t, strategies="UB", lambda_shift=lambda_shift, mu_soc_v=0.0, init_soc_v=0.5, T=T
	)
	model_PEER = initialize_pessimistic_learning(;
		N=10000, n=n, seed=654123,  init_sens=0.1, 
		mixed_λ=lamb, mixed=true, mixed_ν=nu, mixed_freq=mixed_freq, t=t, strategies="HORIZONTAL", lambda_shift=lambda_shift, init_soc_v = 0.0, T=T
	)
	
	adataCB, mdataCB = run!(
		model_CB,
		total, mdata=[:opt_s, :opt_payoff, :s_median, :s_lerror, :s_herror, :Vbar, :Vbar_conf, :Vbar_g0, :Vbar_g1]
	)
	adataPB, mdataPB = run!(
		model_PB,
		total, mdata=[:opt_s, :opt_payoff, :s_median, :s_lerror, :s_herror, :Vbar, :Vbar_pay, :Vbar_g0, :Vbar_g1]
	)
	adataUL, mdataUL = run!(
		model_UL,
		total, mdata=[:opt_s, :opt_payoff, :s_median, :s_lerror, :s_herror, :Vbar, :Vbar_ub, :Vbar_g0, :Vbar_g1]
	)
	adataPEER, mdataPEER = run!(
		model_PEER,
		total, mdata=[:opt_s, :opt_payoff, :s_median, :s_lerror, :s_herror, :Vbar, :Vbar_g0, :Vbar_g1]
	)
	
	timeplot = plot(
		mdataCB.time,
		mdataCB.Vbar_g0,
		color=palette(:Dark2_5)[1],
		lw=2,
		legend=:bottomright,
		legendfontsize=6,
		label="fully conservative",
		xticks=0:maximum(mdataCB.time),
		xlabel="time",
		ylabel="mean growth rate"
	)
	plot!(
		mdataCB.time,
		mdataCB.Vbar_g1,
		color=palette(:Dark2_5)[1],
		ls=:dash,
		lw=2,
		label=""
	)
	plot!(
		mdataPB.time,
		mdataPB.Vbar_g0,
		lw=2,
		color=palette(:Dark2_5)[2],
		#alpha=0.5,
		label="conservative payoff bias"
	)
	plot!(
		mdataPB.time,
		mdataPB.Vbar_g1,
		ls=:dash,
		lw=2,
		#alpha=0.5,
		color=palette(:Dark2_5)[2],
		label=""
	)
	plot!(
		mdataUL.time,
		mdataUL.Vbar_g0,
		lw=2,
		label="mixed learning",
		color=palette(:Dark2_5)[3]
	)
	plot!(
		mdataUL.time,
		mdataUL.Vbar_g1,
		lw=2,
		ls=:dash,
		label="",
		color=palette(:Dark2_5)[3]
	)
	plot!(
		mdataPEER.time,
		mdataPEER.Vbar_g0,
		lw=2,
		label="fully explorative",
		#alpha=0.5,
		color=palette(:Dark2_5)[4]
	)
	plot!(
		mdataPEER.time,
		mdataPEER.Vbar_g1,
		lw=2,
		ls=:dash,
		#alpha=0.5,
		label="",
		color=palette(:Dark2_5)[4]
	)
	hline!([1], c="black", alpha=0.5, ls=:dash, lw=1, label="")

	return timeplot
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

function plot_change_soclearn(cultrans, n, ν, sens, T, t; seed=1, legend=false)
	
	miao1 = cultrans[cultrans.strategies .== "CB" .&& cultrans.n .== n .&& cultrans.ν .== ν .&& cultrans.init_sens .== sens .&& cultrans.T .== T .&& cultrans.seed .== seed .&& cultrans.t .== t .&& cultrans.time .== 0, :]
	
	miao2 = cultrans[cultrans.strategies .== "CB" .&& cultrans.n .== n .&& cultrans.ν .== ν .&& cultrans.init_sens .== sens .&& cultrans.T .== T .&& cultrans.seed .== seed .&& cultrans.t .== t .&& cultrans.time .== 10, :]

	miao3 = cultrans[cultrans.strategies .== "PB" .&& cultrans.n .== n .&& cultrans.ν .== ν .&& cultrans.init_sens .== sens .&& cultrans.T .== T .&& cultrans.seed .== seed .&& cultrans.t .== t .&& cultrans.time .== 10, :]

	miao4 = cultrans[cultrans.strategies .== "UL" .&& cultrans.n .== n .&& cultrans.ν .== ν .&& cultrans.init_sens .== sens .&& cultrans.T .== T .&& cultrans.seed .== seed .&& cultrans.t .== t .&& cultrans.time .== 10, :]
	
	changeplot = plot(
		[miao1.λ[1] .- 0.05, miao3.λ[1] .+ 0.05],
		[miao1.s_median[1], miao3.s_median[1]],
		c="dark blue",
		legend=legend,
		label="",
		alpha=0.35,
		xrotation=90,
		xlabel="environmental uncertainty (λ)",
		xlabelfontsize=8,
		ylabel=ν==1 ? "stake (s)" : "",
		dpi=600
	)
	for i in 2:length(miao1.λ)
		plot!(
			[miao1.λ[i] .- 0.05, miao3.λ[i] .+ 0.05],
			[miao1.s_median[i], miao3.s_median[i]],
			c="dark blue",
			label="",
			alpha=0.35
		)
	end
	for i in 1:length(miao1.λ)
		plot!(
			[miao1.λ[i] .- 0.05, miao2.λ[i] .+ 0.05],
			[miao1.s_median[i], miao2.s_median[i]],
			c="purple",
			label="",
			alpha=0.35
		)
	end
	for i in 1:length(miao1.λ)
		plot!(
			[miao1.λ[i] .- 0.05, miao4.λ[i] .+ 0.05],
			[miao1.s_median[i], miao4.s_median[i]],
			c="pink",
			label="",
			alpha=0.35
		)
	end
	scatter!(
		miao1.λ .- 0.05, miao1.s_median, yerror=(miao1.s_lerror, miao1.s_herror), xticks=1:0.5:5, c="black", label="first generation", markersize=3,
	)
	scatter!(
		miao3.λ .+ 0.05, miao3.s_median, yerror=(miao3.s_lerror, miao3.s_herror),
		c="dark blue", markerstrokecolor="dark blue", markerstrokewidth=1, label="payoff bias", markersize=3,
	)
	scatter!(
		miao2.λ .+ 0.05, miao2.s_median, yerror=(miao2.s_lerror, miao2.s_herror), c="purple", markerstrokecolor="purple", markerstrokewidth=1, label="conformity", markersize=3,
	)
	scatter!(
		miao4.λ .+ 0.05, miao4.s_median, yerror=(miao4.s_lerror, miao4.s_herror), c="pink", markerstrokecolor="pink", markerstrokewidth=1, label="blending", markersize=3,
	)

	return changeplot
end


function plot_comparison(n, sens, T, t; seed=1)
	cultrans = CSV.read("../data/analysis_2.csv", DataFrame)
	plot(
		plot( 
			[plot_payoff_soclearn(cultrans, n, i, sens, T, t, seed=seed) for i in [1, 2, 3]]...,
			layout=(1,3), link=:all
		),
		plot( 
			[plot_change_soclearn(cultrans, n, i, sens, T, t, seed=seed) for i in [1, 2, 3]]...,
			layout=(1,3), link=:all
		),
		layout=(2,1), size=(700, 400), margin=2Plots.mm
	)
end

function plot_scatterconf(λ, c; T=5000, n=1000, var=0.2, w_constant=false, w=0:0.05:1|>collect, n_samples=1, T_opt=5000, legend=true, xlab=false, ylab=false, gp=:top, alpha=0.05)
	opt_surv = findmax(sim_payoffs(λ, 0:0.01:1, c, seasons=T_opt)[2])[2]
	noconf, conf, pop = compare_conformity(λ, c, opt_surv=opt_surv, var=var, T=T, n_samples=n_samples, n=n, w_constant=w_constant, w=w)
	scatterconf(noconf, conf, pop, opt_surv, legend=legend, xlab=xlab, ylab=ylab, gp=gp, alpha=alpha)
end


=#

#==
function simulate_gambles_wealth(λ, stake, n, b, a;
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
==#