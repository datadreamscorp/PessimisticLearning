using StatsBase, Random, Distributions, Agents

@agent struct Peep(NoSpaceAgent)
    ###
    #Heritable characteristics
    s::Float64 #risk attidude
    soc_h::Float64 #weight of horizontal social information
    soc_v::Float64 #weight of vertical and oblique social information
    L::Int64 #learning strategy for vertical and oblique social information
    sens::Float64 #sensitivity to observed ruin
	group::Int64 #initial wealth
    ###
    #Other dynamic characteristics
    log_payoff::Float64
    models::Vector{Int64}
    old_models::Vector{Int64}
    s_old::Float64
    soc_h_old::Float64
    soc_v_old::Float64
    L_old::Int64
    sens_old::Float64
end


Base.@kwdef mutable struct Parameters
    #model parameters
    N::Int64
    n::Int64
    T::Int64
    t::Int64
    λ::Float64
	ν::Float64
    init_soc_h::Float64
    init_soc_v::Float64
    init_sens::Float64
    mu_soc_h::Float64
    mu_soc_v::Float64
    mu_sens::Float64
    strat_pool::Vector{Int64}
	trimean::Int64
    abarrier::Bool
    demographic_filter::Bool
    selection::Bool
    total_ticks::Int64
    seed::Int64
    steps::Int64
	envshift::Int64
	lambda_shift::Float64
	mixed::Bool
	mixed_freq::Float64
	mixed_aleph::Vector{Float64}
	mixed_aleph_shift::Vector{Float64}
	mixed_λ::Vector{Float64}
	mixed_λ_shift::Vector{Float64}
	mixed_L::Vector{Float64}
	parochial::Bool
	periodic::Bool
    #data
    s_dist::Vector{Float64}
    s_median::Float64
    s_lerror::Float64
    s_herror::Float64
    V_dist::Vector{Float64}
    V_median::Float64
    V_probsurv::Float64
    Vbar::Float64
	Vbar_vec::Vector{Float64}
	gVbar::Float64
	Vbar_g0::Float64
	Vbar_g1::Float64
    opt_s::Float64
    opt_payoff::Float64
    tick::Int64
end

function median_error(X; f="median", l=0.1, h=0.9)
	med = f == "median" ? median(X) : mean(X)
	ϵ⁻ = med - quantile(X, l)
	ϵ⁺ = quantile(X, h) - med
	return (med, ϵ⁻, ϵ⁺)
end

function aleph_transform(aleph; Vb=1)
	return Vb/(1 - aleph)
end

function simulate_gambles(λ, א, s;
	Vb = 1,
	seasons=5000,
	rounds=1,
	abarrier=true
	)

	log_capital = log( aleph_transform(א, Vb=Vb) )
	for i in 1:seasons

		rate = 1 / rand( Pareto(λ) )

		for j in 1:rounds
			if rand() < rate
				log_capital = log(1 + s)  + log_capital
			else
				log_capital = log(1 - s) + log_capital
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

#=
function sim_payoffs(
	λ, S, א; 
	abarrier=true, seasons=1000, rounds=1, n=1000
	)

	payoffs = ( [
			exp.( [simulate_gambles(λ, א, s, abarrier=abarrier, seasons=seasons, rounds=rounds) for i ∈ 1:n] ./ seasons )
			for s ∈ S
		] )
	
	surv = [filter(v -> v >= 1.0, p) for p in payoffs]
	fullmean = mean.(payoffs)
	survmean = mean.(surv)
	prob_surv = length.(surv) ./ n

	return(survmean, fullmean, prob_surv)
end
=#

function sample_environment!(model)
	for a ∈ allagents(model)
        est = 0
        for t ∈ 1:model.t
			if model.mixed
				est += rand(abmrng(model)) < 1 / rand( abmrng(model), Pareto(model.mixed_λ[a.group]) ) ? 1 : 0
			else
            	est += rand(abmrng(model)) < 1 / rand( abmrng(model), Pareto(model.λ) ) ? 1 : 0
			end
        end
        est = est / model.t
        a.s = 2*est - 1 > 0 ? 2*est - 1 : 0.0
    end
end

function pool!(model)
    for a ∈ shuffle(abmrng(model), allagents(model)|>collect)
        models = filter(x -> x.id ∈ a.models, allagents(model)|>collect)
        if model.parochial
			models = filter(x -> x.group == a.group, models)
		end
		if length(models) > 0
			model_stakes = [b.s for b ∈ models]
        	a.s = (1 - a.soc_h)*a.s + (a.soc_h)*mean(model_stakes)
		end
    end
end

function learn_from_olds!(model)
	for a in allagents(model)
		a.old_models = sample(abmrng(model), 1:model.N, model.n)
		old_models = filter(x -> x.id ∈ a.old_models, allagents(model)|>collect)

		if model.demographic_filter
			old_models = filter(x -> x.log_payoff > 0.0, old_models)
		end
		
		if model.mixed && model.parochial
			old_models = filter(x -> x.group == a.group, old_models)
		end

        if length(old_models) > 0
            old_s = [b.s_old for b in old_models]
            old_payoffs = [b.log_payoff for b in old_models]
            if a.L == 1
				trim = clamp( model.trimean, 0, ceil(model.n/2) - 1 )
				old_s = sort(old_s)
				for i in 1:trim
					pop!(old_s)
					popfirst!(old_s)
				end
				learned_s = mean(old_s)
            elseif a.L == 2
                learned_s = median(old_s)
            elseif a.L == 3
                learned_s = old_s[ findmax(old_payoffs)[2] ]
            end
            
            a.s = (1 - a.soc_v)*a.s + a.soc_v*learned_s
        end
	end
end

function pull_stake!(model, agent)
	if agent.sens > 0.0
		agent.s = clamp(
			agent.s - rand( abmrng(model), Exponential(agent.sens) ),
			0.0, 1.0
		)
	end
end

function play!(model)

	for a ∈ allagents(model)
		a.log_payoff = model.mixed ? log(model.mixed_aleph[a.group]) : log(model.ν)
	end
	
	for i in 1:model.T

		for a ∈ allagents(model)
			if model.mixed
				a.log_payoff += rand(abmrng(model)) < 1 / rand(abmrng(model), Pareto(model.mixed_λ[a.group]) ) ? log(1 + a.s) : log(1 - a.s) 
			else
				a.log_payoff += rand(abmrng(model)) < 1 / rand(abmrng(model), Pareto(model.λ) ) ? log(1 + a.s) : log(1 - a.s) 
			end
		end
	
		if model.abarrier
			for a ∈ allagents(model)
				if a.log_payoff != -Inf

					if a.log_payoff < 0.0
						a.log_payoff = -Inf
                        for b ∈ filter(
						x -> a.id ∈ x.models, 
						allagents(model)|>collect
						)
						    if b.log_payoff > 0.0 && abs(b.s - a.s) < model.init_sens
                                b.s = pull_stake!(model, b)
                            end
                        end
					end

				end
			end
		end

	end

end

function pass_the_torch!(model)
	for a in allagents(model)|>collect
		a.s_old = a.s
	    a.soc_h_old = a.soc_h
	    a.soc_v_old = a.soc_v
	    a.L_old = a.L
	    a.sens_old = a.sens
		a.soc_v = rand(abmrng(model)) < model.mu_soc_v ? abs(a.soc_v - 1.0) : a.soc_v
	end
end

function initialize_pessimistic_learning(;
	N = 10000,
    n = 25,
	T = 1000,
	t = 25,
    λ = 2.5,
    aleph = 0.5,
	init_soc_h = 1.0,
    init_soc_v = 1.0,
	init_sens = 0.1,
    mu_soc_h = 0.0,
    mu_soc_v = 0.0,
    mu_sens = 0.0,
	strategies = "UB",
	trimean = 0,
    abarrier = true,
	demographic_filter = true,
	selection = false,
	seed = 123456789,
	total_ticks = 5,
    steps = 3,
	envshift = 1000,
	lambda_shift = 6.0,
	mixed = false,
	mixed_freq = 0.5,
	mixed_aleph1 = 0.65,
	mixed_aleph2 = 0.95,
	mixed_aleph1_shift = 0.95,
	mixed_aleph2_shift = 0.95,
	mixed_λ1 = 2.5,
	mixed_λ2 = 6.0,
	mixed_λ1_shift = 6.0,
	mixed_λ2_shift = 6.0,
	mixed_L1 = 1,
	mixed_L2 = 1,
	parochial = false,
	periodic = false
)
	rng = Xoshiro(seed)
	
	if strategies == "UB"
		strat_pool = [1]

	elseif strategies == "CB"
		strat_pool = [2]

	elseif strategies == "PB"
		strat_pool = [3]

	elseif strategies == "UB&CB"
		strat_pool = [1, 2]

	elseif strategies == "UB&PB"
		strat_pool = [1, 3]

	elseif strategies == "CB&PB"
		strat_pool = [2, 3]

	elseif strategies == "ALL"
		strat_pool = [1, 2, 3]

	else
		error(raw"Invalid learning strategy pool.")
	end

	#survmean, fullmean, probsurv = sim_payoffs(λ, 0:0.005:1, ν)
	#opt_payoff, opt_idx = findmax( filter(x -> !isnan(x), probsurv.*survmean ) )
	#opt_s = (0:0.005:1|>collect)[ opt_idx ]
	
	properties = Parameters(
		#model parameters
	    N,
	    n,
        T,
        t,
        λ,
        aleph_transform(aleph),
        init_soc_h,
        init_soc_v,
		init_sens,
	    mu_soc_h,
	    mu_soc_v,
		mu_sens,
	    strat_pool,
		trimean,
        abarrier,
		demographic_filter,
		selection,
        total_ticks,
        seed,
        steps,
		envshift,
		lambda_shift,
		mixed,
		mixed_freq,
		[aleph_transform(mixed_aleph1), aleph_transform(mixed_aleph2)],
		[aleph_transform(mixed_aleph1_shift), aleph_transform(mixed_aleph2_shift)],
		[mixed_λ1, mixed_λ2],
		[mixed_λ1_shift, mixed_λ2_shift],
		[mixed_L1, mixed_L2],
		parochial,
		periodic,
        #data
        Vector{Float64}(),
        0.0,
        0.0,
        0.0,
        Vector{Float64}(),
        0.0,
        0.0,
		0.0,
		Vector{Float64}(),
		0.0,
		0.0,
		0.0,
		0.0,
		0.0,
        0
	)

	model = StandardABM( 
		Peep, 
		nothing;
		properties = properties,
        model_step! = model_step!,
		rng = rng
	)

	for a in 1:N
		
		group = rand(abmrng(model)) < model.mixed_freq ? 1 : 2

		agent = Peep( 
			a,
            0.0,
			model.init_soc_h, 
			init_soc_v,
			model.mixed ? model.mixed_L[group] : rand(abmrng(model), model.strat_pool),
            init_sens,
			group,
			model.mixed ? model.mixed_aleph[group] : model.ν,
            sample(
				abmrng(model), 
				deleteat!(1:N|>collect, a), 
				model.n, 
				replace=false
			),
			Vector{Int64}(),
            0.0,
            0.0,
            0.0,
            0,
            0.0
		)
        new_a = add_agent!(agent, model)
		
	end

	
    if model.steps == 1
        sample_environment!(model)
    elseif model.steps == 2
        sample_environment!(model)
        pool!(model)
    elseif model.steps >= 3
        sample_environment!(model)
        pool!(model)
        play!(model)    
    end

	model.s_dist = [a.s for a ∈ allagents(model)|>collect]
    model.s_median, model.s_lerror, model.s_herror = median_error(model.s_dist)

	model.V_dist = [exp(a.log_payoff/(model.T+1)) for a ∈ allagents(model)|>collect]
	model.V_median = mean(
			filter(
				x -> x > 0.0,
				[exp(a.log_payoff ./ (model.T+1)) for a ∈ allagents(model)|>collect]
				)
		)

	probruin = length(
		filter(
			x -> x == 0, 
			[exp(a.log_payoff/(model.T+1)) for a ∈ allagents(model)|>collect] 
		)
	) / model.N

	model.V_probsurv = 1 - probruin
	model.Vbar = model.V_probsurv * model.V_median
	push!(model.Vbar_vec, model.Vbar)

	if model.mixed
		model.Vbar_g0 = mean( [exp(a.log_payoff/(model.T+1)) for a ∈ filter(x -> x.group == 1, allagents(model)|>collect)] )
		model.Vbar_g1 = mean( [exp(a.log_payoff/(model.T+1)) for a ∈ filter(x -> x.group == 2, allagents(model)|>collect)] )
	end

	pass_the_torch!(model)
	
	return model
		
end

function model_step!(model)

	if model.tick != 0 && model.tick % model.envshift == 0
		if model.periodic
			if model.mixed
				current1 = [model.mixed_λ[1], model.mixed_aleph[1]]
				current2 = (model.mixed_λ[2], model.mixed_aleph[2])

				model.mixed_λ[1] = model.mixed_λ_shift[1]
				model.mixed_λ[2] = model.mixed_λ_shift[2]
				model.mixed_λ_shift[1] = current1[1]
				model.mixed_λ_shift[2] = current2[1]

				model.mixed_aleph[1] = model.mixed_aleph_shift[1]
				model.mixed_aleph[2] = model.mixed_aleph_shift[2]
				model.mixed_aleph_shift[1] = current1[2]
				model.mixed_aleph_shift[2] = current2[2]
			else
				current = model.λ
				model.λ = model.lambda_shift
				model.lambda_shift = current
			end
		else
			if model.mixed
				model.mixed_λ[1] = model.mixed_λ_shift[1]
				model.mixed_λ[2] = model.mixed_λ_shift[2]

				model.mixed_aleph[1] = model.mixed_aleph_shift[1]
				model.mixed_aleph[2] = model.mixed_aleph_shift[2]
			else
				model.λ = model.lambda_shift
			end
		end
	end

	model.tick += 1

	sample_environment!(model)
	pool!(model)
	model.tick > 0 && learn_from_olds!(model)
	play!(model) 

	##DATA COLLECTION
	model.s_dist = [a.s for a ∈ allagents(model)|>collect]
	model.s_median, model.s_lerror, model.s_herror = median_error(model.s_dist)
	

    demo_payoffs = filter(
        x -> x > 0.0,
        [exp(a.log_payoff / (model.T)) for a ∈ allagents(model)|>collect]
        )
    model.V_median = length(demo_payoffs) > 0 ? mean(demo_payoffs) : 0.0

	probruin = length(
		filter(
			x -> x == 0, 
			[exp(a.log_payoff / (model.T)) for a ∈ allagents(model)|>collect] 
		)
	) / model.N

	model.V_probsurv = 1 - probruin
	model.Vbar = model.V_probsurv * model.V_median
	push!(model.Vbar_vec, model.Vbar)
	model.gVbar = geomean(model.Vbar_vec)

	if model.mixed
		model.Vbar_g0 = mean( [exp(a.log_payoff/(model.T+1)) for a ∈ filter(x -> x.group == 1, allagents(model)|>collect)] )
		model.Vbar_g1 = mean( [exp(a.log_payoff/(model.T+1)) for a ∈ filter(x -> x.group == 2, allagents(model)|>collect)] )
	end

	pass_the_torch!(model)
	
end
