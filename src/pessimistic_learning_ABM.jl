using StatsBase, Random, Distributions, Agents

function kelly_stake(u)
	u > 0.5 ? 2*u - 1 : 0.0
end

function median_error(X; f="median", l=0.25, h=0.75)
	med = f == "median" ? median(X) : mean(X)
	ϵ⁻ = quantile(X, l)
	ϵ⁺ = quantile(X, h)
	return (med, ϵ⁻, ϵ⁺)
end

function trauma(x, delta)
    if delta == 0
        return x == 0 ? 1.0 : 0.0  # Dirac delta at x = 0
    elseif delta == 0.5
        return 1.0 - x  # Linear decay
    elseif delta == 1
        return x == 1 ? 0.0 : 1.0  # Inverted Dirac delta at x = 1
    elseif delta > 0 && delta < 0.5
        p = 1.0 / (2.0 * delta)
        return (1.0 - x) ^ p
    elseif delta > 0.5 && delta < 1
        q = 1.0 / (2.0 * (1.0 - delta))
        return 1.0 - x ^ q
    else
        error("delta must be in [0, 1]")
    end
end

function aleph_transform(aleph; Vb=1)
	return Vb/(1 - aleph)
end

function sample_environment!(model)
	for a ∈ allagents(model)
        a.α_young = 1.0
		a.β_young = 1.0
        for t ∈ 1:model.t
			if model.mixed
                if rand( abmrng(model) ) < model.mixed_λ[a.group]
					a.α_young += 1
				else
					a.β_young += 1
				end
			else
                if rand( abmrng(model) ) < model.λ
                #if rand( abmrng(model) ) < 1 / rand(abmrng(model), Pareto(model.λ/(1 - model.λ)))
					a.α_young += 1
				else
            		a.β_young += 1
				end
			end
        end
    end
end

function pool!(model)
    for a ∈ shuffle(abmrng(model), allagents(model)|>collect)

        a.models = sample(abmrng(model), deleteat!((1:model.N)|>collect, a.id), model.n, replace = false)
        models = filter(x -> x.id ∈ a.models, allagents(model)|>collect)
        
		if model.mixed && a.parochial
			models = filter(x -> x.group == a.group, models)
		end

		if length(models) > 0

			model_alphas = [b.α_young for b ∈ models]
			model_betas = [b.β_young for b ∈ models]

            if sum(model_alphas) > 1 && sum(model_betas) > 1

                a.α_young += (a.soc_h)*sum( model_alphas )
                a.β_young += (a.soc_h)*sum( model_betas )

            end
            
		end

        a.α = a.α_young
		a.β = a.β_young

		mean_beta = a.α / (a.α + a.β)
        a.s_child = kelly_stake(mean_beta)
        a.s = a.s_child

    end
end

function learn_from_olds!(model)

	for a ∈ shuffle(abmrng(model), allagents(model)|>collect)

		a.old_models = sample(abmrng(model), 1:model.N, model.m, replace = false)
		old_models = filter(x -> x.id ∈ a.old_models, allagents(model)|>collect)

		if model.demographic_filter
			old_models = filter(x -> x.log_payoff > 0.0, old_models)
		end
		
		if model.mixed && a.parochial
			old_models = filter(x -> x.group == a.group, old_models)
		end

        if length(old_models) > 0

			old_alphas = [b.α_old for b ∈ old_models]
			old_betas = [b.β_old for b ∈ old_models]
            old_payoffs = [b.log_payoff for b ∈ old_models]

            if a.L == 1
				learned_α = mean(old_alphas)
				learned_β = mean(old_betas)
            elseif a.L == 2
                learned_α = rand(abmrng(model), old_alphas)
				learned_β = rand(abmrng(model), old_betas)
            elseif a.L == 3
				highest_payoff = findmax(old_payoffs)[2]
                learned_α = old_alphas[ highest_payoff ]
				learned_β = old_betas[ highest_payoff ]
            end
            
            a.α_young = (1 - a.soc_v)*a.α_young + a.soc_v*learned_α
			a.β_young = (1 - a.soc_v)*a.β_young + a.soc_v*learned_β

        end

        a.α = a.α_young
		a.β = a.β_young

		mean_beta = a.α_young / (a.α_young + a.β_young)
		a.s_young = kelly_stake(mean_beta)
        a.s = a.s_young

	end
end

function pull_stake!(model, agent, other)
    normalized_difference = abs(other.s - agent.s) / maximum( [agent.s, 1 - agent.s] )
	agent.β += trauma( normalized_difference, agent.sens )*agent.β
	mean_beta = agent.α / (agent.α + agent.β)
	agent.s = kelly_stake(mean_beta)
end

function play!(model)

	for a ∈ allagents(model)
		a.log_payoff = model.mixed ? log(aleph_transform(model.mixed_aleph[a.group])) : log(aleph_transform(model.aleph))
        a.s_vec = []
    end
	
	for i ∈ 1:model.T

		for a ∈ allagents(model)
			
			u = rand( abmrng(model), Beta(a.α, a.β) )
			s = kelly_stake(u)

			if model.mixed
                if rand( abmrng(model) ) < model.mixed_λ[a.group]
					a.log_payoff += log(1 + s)
					a.α += 1
				else
					a.log_payoff += log(1 - s)
					a.β += 1
				end
			else
                if rand( abmrng(model) ) < model.λ
                #if rand( abmrng(model) ) < 1 / rand(abmrng(model), Pareto(model.λ/(1 - model.λ)))
					a.log_payoff += log(1 + s)
					a.α += 1
				else
					a.log_payoff += log(1 - s)
					a.β += 1
				end
			end

			mean_beta = a.α / (a.α + a.β)
			a.s = kelly_stake(mean_beta)

            if a.log_payoff != -Inf

                if a.log_payoff < 0.0
                    a.log_payoff = -Inf
                    for b ∈ filter(
                        x -> a.id ∈ x.models, 
                        allagents(model)|>collect
                        )
                        if b.log_payoff > 0.0
                            pull_stake!(model, b, a)
                        end
                    end
                end

            end

            push!(a.s_vec, a.s)
            a.s_mean = mean(a.s_vec)
            #a.s_median = median(a.s_vec)

		end

	end

	for a ∈ allagents(model)|>collect
		a.log_payoff -= model.mixed ? log(aleph_transform(model.mixed_aleph[a.group])) : log(aleph_transform(model.aleph))
        a.avg_payoff = exp( a.log_payoff / model.T )
	end

end

function pass_the_torch!(model)
	for a in allagents(model)|>collect
		a.α_old = a.α
		a.β_old = a.β
		a.s_old = a.s
	    a.soc_h_old = a.soc_h
	    a.soc_v_old = a.soc_v
	    a.L_old = a.L
	    a.sens_old = a.sens
	end
end

function selection!(model)

	peeps = allagents(model)|>collect

	if !model.mixed

		fitness = [clamp(exp(a.log_payoff / (model.T)), eps(), Inf)^model.b_coeff for a ∈ peeps]
		total_fitness = sum(fitness)
		fitness_weights = weights(fitness ./ total_fitness)

		soc_h_vec = [a.soc_h_old for a ∈ peeps]
		soc_v_vec = [a.soc_v_old for a ∈ peeps]
		sens_vec = [a.sens_old for a ∈ peeps]
		L_vec = [a.L_old for a ∈ peeps]

		for peep ∈ peeps

			new_soc_h = rand(abmrng(model)) > model.mu_soc_h ? sample(abmrng(model), soc_h_vec, fitness_weights) : rand(abmrng(model))
			peep.soc_h = model.mu_soc_h > 0.0 ? new_soc_h : peep.soc_h
            peep.soc_h = model.mu_soc_h > 0.0 ? clamp( rand(abmrng(model), Normal(peep.soc_h, model.mu_std)) , 0, 1 ) : peep.soc_h

			new_soc_v = rand(abmrng(model)) > model.mu_soc_v ? sample(abmrng(model), soc_v_vec, fitness_weights) : rand(abmrng(model))
			peep.soc_v = model.mu_soc_v > 0.0 ? new_soc_v : peep.soc_v
            peep.soc_v = model.mu_soc_v > 0.0 ? clamp( rand(abmrng(model), Normal(peep.soc_v, model.mu_std)), 0, 1 ) : peep.soc_v

			new_sens = rand(abmrng(model)) > model.mu_sens ? sample(abmrng(model), sens_vec, fitness_weights) : rand(abmrng(model))
			peep.sens = model.mu_sens > 0.0 ? new_sens : peep.sens
            peep.sens = model.mu_sens > 0.0 ? clamp( rand(abmrng(model), Normal(peep.sens, model.mu_std)), 0, 1 ) : peep.sens

			new_L = rand(abmrng(model)) > model.mu_L ? sample(abmrng(model), L_vec, fitness_weights) : rand(abmrng(model), model.strat_pool)
			peep.L = model.mu_L > 0.0 ? new_L : peep.L

		end

	else
		
		g0 = filter(x -> x.group == 1, peeps)
		g1 = filter(x -> x.group == 2, peeps)

		fitness_g0 = [clamp(exp(a.log_payoff / (model.T)), eps(), Inf)^model.b_coeff for a ∈ g0]
		fitness_g1 = [clamp(exp(a.log_payoff / (model.T)), eps(), Inf)^model.b_coeff for a ∈ g1]

		fitness_weights_g0 = weights(fitness_g0 ./ sum(fitness_g0))
		fitness_weights_g1 = weights(fitness_g1 ./ sum(fitness_g1))

		soc_h_vec_g0 = [a.soc_h_old for a ∈ g0]
		soc_v_vec_g0 = [a.soc_v_old for a ∈ g0]
		sens_vec_g0 = [a.sens_old for a ∈ g0]
		L_vec_g0 = [a.L_old for a ∈ g0]
        parochial_vec_g0 = [a.parochial for a ∈ g0]

		soc_h_vec_g1 = [a.soc_h_old for a ∈ g1]
		soc_v_vec_g1 = [a.soc_v_old for a ∈ g1]
		sens_vec_g1 = [a.sens_old for a ∈ g1]
		L_vec_g1 = [a.L_old for a ∈ g1]
        parochial_vec_g1 = [a.parochial for a ∈ g1]

		for peep ∈ g0

			new_soc_h = rand(abmrng(model)) > model.mu_soc_h ? sample(abmrng(model), soc_h_vec_g0, fitness_weights_g0) : rand(abmrng(model))
			peep.soc_h = model.mu_soc_h > 0.0 ? new_soc_h : peep.soc_h
            peep.soc_h = model.mu_soc_h > 0.0 ? clamp( rand(abmrng(model), Normal(peep.soc_h, model.mu_std)) , 0, 1 ) : peep.soc_h

			new_soc_v = rand(abmrng(model)) > model.mu_soc_v ? sample(abmrng(model), soc_v_vec_g0, fitness_weights_g0) : rand(abmrng(model))
			peep.soc_v = model.mu_soc_v > 0.0 ? new_soc_v : peep.soc_v
            peep.soc_v = model.mu_soc_v > 0.0 ? clamp( rand(abmrng(model), Normal(peep.soc_v, model.mu_std)), 0, 1 ) : peep.soc_v

			new_sens = rand(abmrng(model)) > model.mu_sens ? sample(abmrng(model), sens_vec_g0, fitness_weights_g0) : rand(abmrng(model))
			peep.sens = model.mu_sens > 0.0 ? new_sens : peep.sens
            peep.sens = model.mu_sens > 0.0 ? clamp( rand(abmrng(model), Normal(peep.sens, model.mu_std)), 0, 1 ) : peep.sens

			new_L = rand(abmrng(model)) > model.mu_L ? sample(abmrng(model), L_vec_g0, fitness_weights_g0) : rand(abmrng(model), model.strat_pool)
			peep.L = model.mu_L > 0.0 ? new_L : peep.L

            new_par = rand(abmrng(model)) > model.mu_parochial ? sample(abmrng(model), parochial_vec_g0, fitness_weights_g0) : rand(abmrng(model), [true, false])
			peep.parochial = model.mu_parochial > 0.0 ? new_par : peep.parochial

		end

		for peep ∈ g1

			new_soc_h = rand(abmrng(model)) > model.mu_soc_h ? sample(abmrng(model), soc_h_vec_g1, fitness_weights_g1) : rand(abmrng(model))
			peep.soc_h = model.mu_soc_h > 0.0 ? new_soc_h : peep.soc_h
            peep.soc_h = model.mu_soc_h > 0.0 ? clamp( rand(abmrng(model), Normal(peep.soc_h, model.mu_std)) , 0, 1 ) : peep.soc_h

			new_soc_v = rand(abmrng(model)) > model.mu_soc_v ? sample(abmrng(model), soc_v_vec_g1, fitness_weights_g1) : rand(abmrng(model))
			peep.soc_v = model.mu_soc_v > 0.0 ? new_soc_v : peep.soc_v
            peep.soc_v = model.mu_soc_v > 0.0 ? clamp( rand(abmrng(model), Normal(peep.soc_v, model.mu_std)), 0, 1 ) : peep.soc_v

			new_sens = rand(abmrng(model)) > model.mu_sens ? sample(abmrng(model), sens_vec_g1, fitness_weights_g1) : rand(abmrng(model))
			peep.sens = model.mu_sens > 0.0 ? new_sens : peep.sens
            peep.sens = model.mu_sens > 0.0 ? clamp( rand(abmrng(model), Normal(peep.sens, model.mu_std)), 0, 1 ) : peep.sens
            
			new_L = rand(abmrng(model)) > model.mu_L ? sample(abmrng(model), L_vec_g1, fitness_weights_g1) : rand(abmrng(model), model.strat_pool)
			peep.L = model.mu_L > 0.0 ? new_L : peep.L

            new_par = rand(abmrng(model)) > model.mu_parochial ? sample(abmrng(model), parochial_vec_g1, fitness_weights_g1) : rand(abmrng(model), [true, false])
			peep.parochial = model.mu_parochial > 0.0 ? new_par : peep.parochial

		end
	end
end

@agent struct Peep(NoSpaceAgent)
    ###
    #Heritable and developmental characteristics
	α_young::Float64 #positive impression of environment during juvenile
	β_young::Float64 #negative impression of environment during juvenile
	α::Float64 #final positive impression of environment
	β::Float64 #final negative impression of environment
    s_child::Float64 #childhood stake
    s_young::Float64 #juvenile stake
    s_vec::Vector{Float64} #stake vector
	s::Float64 #end of lifetime stake
    s_mean::Float64
    s_median::Float64
    soc_h::Float64 #weight of horizontal social information
    soc_v::Float64 #weight of vertical and oblique social information
    L::Int64 #learning strategy for vertical and oblique social information
    sens::Float64 #sensitivity to observed ruin
	group::Int64 #group identity
    parochial::Bool
    ###
    #Other dynamic characteristics
    log_payoff::Float64
    avg_payoff::Float64
    models::Vector{Int64}
    old_models::Vector{Int64}
	α_old::Float64
	β_old::Float64
    s_old::Float64
    soc_h_old::Float64
    soc_v_old::Float64
    L_old::Int64
    sens_old::Float64
end

Base.@kwdef mutable struct Parameters
    # Model parameters
    N::Int64
    n::Int64
    m::Int64
    T::Int64
    t::Int64
    λ::Float64
    aleph::Float64
    soc_h::Float64
    soc_v::Float64
    sens::Float64
    mu_std::Float64
    mu_soc_h::Float64
    mu_soc_v::Float64
    mu_sens::Float64
    mu_L::Float64
    strat_pool::Vector{Int64}
    mu_parochial::Float64
    abarrier::Bool
    steps::Int64
    envshift::Int64
    λ_shift::Float64
    aleph_shift::Float64
    peg_aleph::Bool
    peg_lambda::Bool
    mixed::Bool
    mixed_freq::Float64
    mixed_aleph::Vector{Float64}
    mixed_aleph_shift::Vector{Float64}
    mixed_λ::Vector{Float64}
    mixed_λ_shift::Vector{Float64}
    mixed_L::Vector{Int64}
    parochial::Bool
    periodic::Bool
    demographic_filter::Bool
    selection::Bool
    b_coeff::Float64
    seed::Int64
    tick::Int64 = 0
    total_ticks::Int64 = 2500
    # Data fields with default values
    s_mean::Float64 = 0.0
    s_median::Float64 = 0.0
    s_lerror::Float64 = 0.0
    s_herror::Float64 = 0.0
    s_ltail::Float64 = 0.0
    s_htail::Float64 = 0.0
    s_end_mean::Float64 = 0.0
    s_end_median::Float64 = 0.0
    s_end_lerror::Float64 = 0.0
    s_end_herror::Float64 = 0.0
    s_young_mean::Float64 = 0.0
    s_young_median::Float64 = 0.0
    s_young_lerror::Float64 = 0.0
    s_young_herror::Float64 = 0.0
    s_child_mean::Float64 = 0.0
    s_child_median::Float64 = 0.0
    s_child_lerror::Float64 = 0.0
    s_child_herror::Float64 = 0.0
    sbar::Float64 = 0.0
    mean_increment::Float64 = 0.0
    concentration::Float64 = 0.0
    concentration_lerror::Float64 = 0.0
    concentration_herror::Float64 = 0.0
    s_mean_g0::Float64 = 0.0
    s_median_g0::Float64 = 0.0
    s_lerror_g0::Float64 = 0.0
    s_herror_g0::Float64 = 0.0
    s_ltail_g0::Float64 = 0.0
    s_htail_g0::Float64 = 0.0
    s_end_mean_g0::Float64 = 0.0
    s_end_median_g0::Float64 = 0.0
    s_end_lerror_g0::Float64 = 0.0
    s_end_herror_g0::Float64 = 0.0
    s_young_mean_g0::Float64 = 0.0
    s_young_median_g0::Float64 = 0.0
    s_young_lerror_g0::Float64 = 0.0
    s_young_herror_g0::Float64 = 0.0
    s_child_mean_g0::Float64 = 0.0
    s_child_median_g0::Float64 = 0.0
    s_child_lerror_g0::Float64 = 0.0
    s_child_herror_g0::Float64 = 0.0
    concentration_g0::Float64 = 0.0
    concentration_lerror_g0::Float64 = 0.0
    concentration_herror_g0::Float64 = 0.0
    s_mean_g1::Float64 = 0.0
    s_median_g1::Float64 = 0.0
    s_lerror_g1::Float64 = 0.0
    s_herror_g1::Float64 = 0.0
    s_ltail_g1::Float64 = 0.0
    s_htail_g1::Float64 = 0.0
    s_end_mean_g1::Float64 = 0.0
    s_end_median_g1::Float64 = 0.0
    s_end_lerror_g1::Float64 = 0.0
    s_end_herror_g1::Float64 = 0.0
    s_young_mean_g1::Float64 = 0.0
    s_young_median_g1::Float64 = 0.0
    s_young_lerror_g1::Float64 = 0.0
    s_young_herror_g1::Float64 = 0.0
    s_child_mean_g1::Float64 = 0.0
    s_child_median_g1::Float64 = 0.0
    s_child_lerror_g1::Float64 = 0.0
    s_child_herror_g1::Float64 = 0.0
    concentration_g1::Float64 = 0.0
    concentration_lerror_g1::Float64 = 0.0
    concentration_herror_g1::Float64 = 0.0
    Vbar::Float64 = 0.0
    Vbar_g0::Float64 = 0.0
    Vbar_g1::Float64 = 0.0
    freq_ub::Float64 = 0.0
    freq_pb::Float64 = 0.0
    freq_cb::Float64 = 0.0
    freq_ub_g0::Float64 = 0.0
    freq_pb_g0::Float64 = 0.0
    freq_cb_g0::Float64 = 0.0
    freq_parochial_g0::Float64 = 0.0
    freq_ub_g1::Float64 = 0.0
    freq_pb_g1::Float64 = 0.0
    freq_cb_g1::Float64 = 0.0
    freq_parochial_g1::Float64 = 0.0
    # New fields for soc_h statistics
    soc_h_median::Float64 = 0.0
    soc_h_lerror::Float64 = 0.0
    soc_h_herror::Float64 = 0.0
    soc_h_median_g0::Float64 = 0.0
    soc_h_lerror_g0::Float64 = 0.0
    soc_h_herror_g0::Float64 = 0.0
    soc_h_median_g1::Float64 = 0.0
    soc_h_lerror_g1::Float64 = 0.0
    soc_h_herror_g1::Float64 = 0.0
    # New fields for soc_v statistics
    soc_v_median::Float64 = 0.0
    soc_v_lerror::Float64 = 0.0
    soc_v_herror::Float64 = 0.0
    soc_v_median_g0::Float64 = 0.0
    soc_v_lerror_g0::Float64 = 0.0
    soc_v_herror_g0::Float64 = 0.0
    soc_v_median_g1::Float64 = 0.0
    soc_v_lerror_g1::Float64 = 0.0
    soc_v_herror_g1::Float64 = 0.0
    # New fields for sens statistics
    sens_median::Float64 = 0.0
    sens_lerror::Float64 = 0.0
    sens_herror::Float64 = 0.0
    sens_median_g0::Float64 = 0.0
    sens_lerror_g0::Float64 = 0.0
    sens_herror_g0::Float64 = 0.0
    sens_median_g1::Float64 = 0.0
    sens_lerror_g1::Float64 = 0.0
    sens_herror_g1::Float64 = 0.0
end



function initialize_pessimistic_learning(; 
    N = 1000,
    n = 15,
    m = 15,
    T = 100,
    t = 15,
    u = 0.65,
    aleph = 0.05,
    soc_h = 0.0,
    soc_v = 0.0,
    sens = 0.0,
    mu_std = 0.0,
    mu_soc_h = 0.0,
    mu_soc_v = 0.0,
    mu_sens = 0.0,
    mu_L = 0.0,
    strategies = "UB",
    mu_parochial = 0.0,
    abarrier = true,
    steps = 3,
    envshift = 5000,
    u_shift = 6.0,
    aleph_shift = 0.95,
    peg_aleph = false,
    peg_lambda = false,
    mixed = false,
    mixed_freq = 0.5,
    mixed_aleph1 = 0.05,
    mixed_aleph2 = 0.95,
    mixed_aleph1_shift = 0.95,
    mixed_aleph2_shift = 0.95,
    mixed_u1 = 0.6,
    mixed_u2 = 0.75,
    mixed_λ1_shift = 0.65,
    mixed_λ2_shift = 0.65,
    mixed_L1 = 1,
    mixed_L2 = 1,
    parochial = false,
    periodic = true,
    demographic_filter = true,
    selection = false,
    b_coeff = 1.0,
    seed = 123456789
)
    rng = Xoshiro(seed)

    # Determine the strategy pool
    strat_pool = strategies == "UB"      ? [1] :
                 strategies == "CB"      ? [2] :
                 strategies == "PB"      ? [3] :
                 strategies == "UB&CB"   ? [1, 2] :
                 strategies == "UB&PB"   ? [1, 3] :
                 strategies == "CB&PB"   ? [2, 3] :
                 strategies == "ALL"     ? [1, 2, 3] :
                 error("Invalid learning strategy pool.")

    # Initialize model properties with default values for data fields
    properties = Parameters(
        N = N,
        n = n,
        m = m,
        T = T,
        t = t,
        λ = u,
        aleph = aleph,
        soc_h = soc_h,
        soc_v = soc_v,
        sens = sens,
        mu_std = mu_std,
        mu_soc_h = mu_soc_h,
        mu_soc_v = mu_soc_v,
        mu_sens = mu_sens,
        mu_L = mu_L,
        strat_pool = strat_pool,
        mu_parochial = mu_parochial,
        abarrier = abarrier,
        steps = steps,
        envshift = envshift,
        λ_shift = u_shift,
        aleph_shift = aleph_shift,
        peg_aleph = peg_aleph,
        peg_lambda = peg_lambda,
        mixed = mixed,
        mixed_freq = mixed_freq,
        mixed_aleph = [mixed_aleph1, mixed_aleph2],
        mixed_aleph_shift = [mixed_aleph1_shift, mixed_aleph2_shift],
        mixed_λ = [mixed_u1, mixed_u2],
        mixed_λ_shift = [mixed_λ1_shift, mixed_λ2_shift],
        mixed_L = [mixed_L1, mixed_L2],
        parochial = parochial,
        periodic = periodic,
        demographic_filter = demographic_filter,
        selection = selection,
        b_coeff = b_coeff,
        seed = seed
        # Data fields have default values in the struct definition
    )

    model = StandardABM(
        Peep,
        nothing;
        properties = properties,
        model_step! = model_step!,
        rng = rng
    )

    agent_ids = collect(1:N)

    for a_id ∈ agent_ids
        group = mixed ? (rand(rng) < mixed_freq ? 1 : 2) : 1

        # Sample models (excluding the current agent)
        possible_models = setdiff(agent_ids, [a_id])
        models = sample(rng, possible_models, n; replace = false)

        # Initialize agent traits
        peep = Peep(
            id = a_id,
            α_young = 1.0,
            β_young = 1.0,
            α = 0.0,
            β = 0.0,
            s_child = 0.0,
            s_young = 0.0,
            s_vec = [],
            s = 0.0,
            s_mean = 0.0,
            s_median = 0.0,
            soc_h = selection && mu_soc_h > 0.0 ? rand(abmrng(model)) : soc_h,
            soc_v = selection && mu_soc_v > 0.0 ? rand(abmrng(model)) : soc_v,
            L = mixed ? model.mixed_L[group] : rand(abmrng(model), strat_pool),
            sens = selection && mu_sens > 0.0 ? rand(abmrng(model)) : sens,
            group = group,
            parochial = mu_parochial > 0 ? rand(abmrng(model), [true, false]) : parochial,
            log_payoff = 0.0,
            avg_payoff = 0.0,
            models = models,
            old_models = Int[],
            α_old = 0.0,
            β_old = 0.0,
            s_old = 0.0,
            soc_h_old = 0.0,
            soc_v_old = 0.0,
            L_old = 0,
            sens_old = 0.0
        )
        add_agent!(peep, model)
    end

    # Perform initial steps based on the specified number of steps
    if steps >= 1
        sample_environment!(model)
    end
    if steps >= 2
        pool!(model)
    end
    if steps >= 3
        play!(model)
    end

	## INITIAL DATA COLLECTION
    peeps = collect(allagents(model))

    # s statistics
    s_dist = [a.s_mean for a ∈ peeps]
    model.s_median, model.s_lerror, model.s_herror = median_error(s_dist)
    model.s_mean, model.s_ltail, model.s_htail = median_error(s_dist, f="mean", l=0.05, h=0.95)

    # s_young statistics
    s_dist_young = [a.s_young for a ∈ peeps]
    model.s_young_median, model.s_young_lerror, model.s_young_herror = median_error(s_dist_young)
    model.s_young_mean, _, _ = median_error(s_dist_young, f="mean")
    
    # s_child statistics
    s_dist_child = [a.s_child for a ∈ peeps]
    model.s_child_median, model.s_child_lerror, model.s_child_herror = median_error(s_dist_child)
    model.s_child_mean, _, _ = median_error(s_dist_child, f="mean")

    # s_end statistics
    s_dist_end = [a.s for a ∈ peeps]
    model.s_end_median, model.s_end_lerror, model.s_end_herror = median_error(s_dist_end)
    model.s_end_mean, _, _ = median_error(s_dist_end, f="mean")

    #concentration statistics
    conc_dist = [a.α + a.β for a ∈ peeps]
    model.concentration, model.concentration_lerror, model.concentration_herror = median_error(conc_dist)
    
    # soc_h statistics
    soc_h_values = [a.soc_h for a ∈ peeps]
    model.soc_h_median, model.soc_h_lerror, model.soc_h_herror = median_error(soc_h_values)

    # soc_v statistics
    soc_v_values = [a.soc_v for a ∈ peeps]
    model.soc_v_median, model.soc_v_lerror, model.soc_v_herror = median_error(soc_v_values)

    # sens statistics
    sens_values = [a.sens for a ∈ peeps]
    model.sens_median, model.sens_lerror, model.sens_herror = median_error(sens_values)

    # Existing code for mean values and frequencies
    model.Vbar = mean(exp.([a.log_payoff / model.T for a ∈ peeps]))
    
    model.freq_ub = count(a -> a.L == 1, peeps) / model.N
    model.freq_cb = count(a -> a.L == 2, peeps) / model.N
    model.freq_pb = count(a -> a.L == 3, peeps) / model.N

    if model.mixed
        
        g0 = filter(a -> a.group == 1, peeps)
        g1 = filter(a -> a.group == 2, peeps)

        # s statistics per group
        s_mean_g0 = [a.s_mean for a ∈ g0]
        model.s_median_g0, model.s_lerror_g0, model.s_herror_g0 = median_error(s_mean_g0)
        model.s_mean_g0, model.s_ltail_g0, model.s_htail_g0 = median_error(s_mean_g0, f="mean", l=0.05, h=0.95)

        s_young_g0 = [a.s_young for a ∈ g0]
        model.s_young_median_g0, model.s_young_lerror_g0, model.s_young_herror_g0 = median_error(s_young_g0)
        model.s_young_mean_g0, _, _ = median_error(s_young_g0, f="mean")

        s_child_g0 = [a.s_child for a ∈ g0]
        model.s_child_median_g0, model.s_child_lerror_g0, model.s_child_herror_g0 = median_error(s_child_g0)
        model.s_child_mean_g0, _, _ = median_error(s_child_g0, f="mean")

        s_end_g0 = [a.s for a ∈ g0]
        model.s_end_median_g0, model.s_end_lerror_g0, model.s_end_herror_g0 = median_error(s_end_g0)
        model.s_end_mean_g0, _, _ = median_error(s_end_g0, f="mean")

        conc_dist_g0 = [a.α + a.β for a ∈ g0]
        model.concentration_g0, model.concentration_lerror_g0, model.concentration_herror_g0 = median_error(conc_dist_g0)

        s_mean_g1 = [a.s_mean for a ∈ g1]
        model.s_median_g1, model.s_lerror_g1, model.s_herror_g1 = median_error(s_mean_g1)
        model.s_mean_g1, model.s_ltail_g1, model.s_htail_g1 = median_error(s_mean_g1, f="mean", l=0.05, h=0.95)

        s_young_g1 = [a.s_young for a ∈ g1]
        model.s_young_median_g1, model.s_young_lerror_g1, model.s_young_herror_g1 = median_error(s_young_g1)
        model.s_young_mean_g1, _, _ = median_error(s_young_g1, f="mean")

        s_child_g1 = [a.s_child for a ∈ g1]
        model.s_child_median_g1, model.s_child_lerror_g1, model.s_child_herror_g1 = median_error(s_child_g1)
        model.s_child_mean_g1, _, _ = median_error(s_child_g1, f="mean")

        s_end_g1 = [a.s for a ∈ g1]
        model.s_end_median_g1, model.s_end_lerror_g1, model.s_end_herror_g1 = median_error(s_end_g1)
        model.s_end_mean_g1, _, _ = median_error(s_end_g1, f="mean")

        conc_dist_g1 = [a.α + a.β for a ∈ g1]
        model.concentration_g1, model.concentration_lerror_g1, model.concentration_herror_g1 = median_error(conc_dist_g1)

        # soc_h statistics per group
        soc_h_values_g0 = [a.soc_h for a ∈ g0]
        model.soc_h_median_g0, model.soc_h_lerror_g0, model.soc_h_herror_g0 = median_error(soc_h_values_g0)

        soc_h_values_g1 = [a.soc_h for a ∈ g1]
        model.soc_h_median_g1, model.soc_h_lerror_g1, model.soc_h_herror_g1 = median_error(soc_h_values_g1)

        # soc_v statistics per group
        soc_v_values_g0 = [a.soc_v for a ∈ g0]
        model.soc_v_median_g0, model.soc_v_lerror_g0, model.soc_v_herror_g0 = median_error(soc_v_values_g0)

        soc_v_values_g1 = [a.soc_v for a ∈ g1]
        model.soc_v_median_g1, model.soc_v_lerror_g1, model.soc_v_herror_g1 = median_error(soc_v_values_g1)

        # sens statistics per group
        sens_values_g0 = [a.sens for a ∈ g0]
        model.sens_median_g0, model.sens_lerror_g0, model.sens_herror_g0 = median_error(sens_values_g0)

        sens_values_g1 = [a.sens for a ∈ g1]
        model.sens_median_g1, model.sens_lerror_g1, model.sens_herror_g1 = median_error(sens_values_g1)

        # strategy frequencies per group

        model.freq_ub_g0 = length(filter(x -> x.L == 1, g0)) / length(g0)
        model.freq_ub_g1 = length(filter(x -> x.L == 1, g1)) / length(g1)
        
        model.freq_cb_g0 = length(filter(x -> x.L == 2, g0)) / length(g0)
        model.freq_cb_g1 = length(filter(x -> x.L == 2, g1)) / length(g1)
        
        model.freq_pb_g0 = length(filter(x -> x.L == 3, g0)) / length(g0)
        model.freq_pb_g1 = length(filter(x -> x.L == 3, g1)) / length(g1)

        model.freq_parochial_g0 = length(filter(x -> x.parochial, g0)) / length(g0)
        model.freq_parochial_g1 = length(filter(x -> x.parochial, g1)) / length(g1)

    end

    pass_the_torch!(model)
    return model
end


function model_step!(model)

	if model.selection
		selection!(model)
	end

	if model.tick != 0 && model.tick % model.envshift == 0 && !model.mixed
		if model.periodic
            if !model.peg_lambda
                current_λ = model.λ
                model.λ = model.λ_shift
                model.λ_shift = current_λ
            end
   
            if !model.peg_aleph
			    current_aleph = model.aleph
			    model.aleph = model.aleph_shift
			    model.aleph_shift = current_aleph
            end
		else
		    model.λ = model.λ_shift
		end
	end

	model.tick += 1

	sample_environment!(model)
	pool!(model)
	model.tick > 0 && learn_from_olds!(model)
	play!(model) 

    
    ## DATA COLLECTION
    peeps = collect(allagents(model))

    # s statistics
    s_dist = [a.s_mean for a ∈ peeps]
    model.s_median, model.s_lerror, model.s_herror = median_error(s_dist)
    model.s_mean, model.s_ltail, model.s_htail = median_error(s_dist, f="mean", l=0.05, h=0.95)

    # s_young statistics
    s_dist_young = [a.s_young for a ∈ peeps]
    model.s_young_median, model.s_young_lerror, model.s_young_herror = median_error(s_dist_young)
    model.s_young_mean, _, _ = median_error(s_dist_young, f="mean")
    
    # s_child statistics
    s_dist_child = [a.s_child for a ∈ peeps]
    model.s_child_median, model.s_child_lerror, model.s_child_herror = median_error(s_dist_child)
    model.s_child_mean, _, _ = median_error(s_dist_child, f="mean")

    # s_mean statistics
    s_dist_end = [a.s for a ∈ peeps]
    model.s_end_median, model.s_end_lerror, model.s_end_herror = median_error(s_dist_end)
    model.s_end_mean, _, _ = median_error(s_dist_end, f="mean")

    # s change statistics
    s_vecs = [a.s_vec for a in peeps]
    transposed = [getindex.(s_vecs, i) for i in 1:length(s_vecs[1])]
	mean_trajectory = mean.(transposed)

    inc = [1.0]
    for i in 1:length(mean_trajectory)
        if i > 1
            push!(inc, inc[i-1]*(mean_trajectory[i]/mean_trajectory[i-1]))
        end
    end

    model.sbar = mean(mean_trajectory)
    model.mean_increment = mean(inc)

    # concentration statistics
    conc_dist = [a.α + a.β for a ∈ peeps]
    model.concentration, model.concentration_lerror, model.concentration_herror = median_error(conc_dist)
    
    # soc_h statistics
    soc_h_values = [a.soc_h for a ∈ peeps]
    model.soc_h_median, model.soc_h_lerror, model.soc_h_herror = median_error(soc_h_values)

    # soc_v statistics
    soc_v_values = [a.soc_v for a ∈ peeps]
    model.soc_v_median, model.soc_v_lerror, model.soc_v_herror = median_error(soc_v_values)

    # sens statistics
    sens_values = [a.sens for a ∈ peeps]
    model.sens_median, model.sens_lerror, model.sens_herror = median_error(sens_values)

    # Existing code for mean values and frequencies
    model.Vbar = mean(exp.([a.log_payoff / model.T for a ∈ peeps]))
    
    model.freq_ub = count(a -> a.L == 1, peeps) / model.N
    model.freq_cb = count(a -> a.L == 2, peeps) / model.N
    model.freq_pb = count(a -> a.L == 3, peeps) / model.N

    if model.mixed

        g0 = filter(a -> a.group == 1, peeps)
        g1 = filter(a -> a.group == 2, peeps)

        # s statistics g0
        s_mean_g0 = [a.s_mean for a ∈ g0]
        model.s_median_g0, model.s_lerror_g0, model.s_herror_g0 = median_error(s_mean_g0)
        model.s_mean_g0, model.s_ltail_g0, model.s_htail_g0 = median_error(s_mean_g0, f="mean", l=0.05, h=0.95)

        s_young_g0 = [a.s_young for a ∈ g0]
        model.s_young_median_g0, model.s_young_lerror_g0, model.s_young_herror_g0 = median_error(s_young_g0)
        model.s_young_mean_g0, _, _ = median_error(s_young_g0, f="mean")

        s_child_g0 = [a.s_child for a ∈ g0]
        model.s_child_median_g0, model.s_child_lerror_g0, model.s_child_herror_g0 = median_error(s_child_g0)
        model.s_child_mean_g0, _, _ = median_error(s_child_g0, f="mean")

        s_end_g0 = [a.s for a ∈ g0]
        model.s_end_median_g0, model.s_end_lerror_g0, model.s_end_herror_g0 = median_error(s_end_g0)
        model.s_end_mean_g0, _, _ = median_error(s_end_g0, f="mean")

        conc_dist_g0 = [a.α + a.β for a ∈ g0]
        model.concentration_g0, model.concentration_lerror_g0, model.concentration_herror_g0 = median_error(conc_dist_g0)

        # s statistics g1

        s_mean_g1 = [a.s_mean for a ∈ g1]
        model.s_median_g1, model.s_lerror_g1, model.s_herror_g1 = median_error(s_mean_g1)
        model.s_mean_g1, model.s_ltail_g1, model.s_htail_g1 = median_error(s_mean_g1, f="mean", l=0.05, h=0.95)

        s_young_g1 = [a.s_young for a ∈ g1]
        model.s_young_median_g1, model.s_young_lerror_g1, model.s_young_herror_g1 = median_error(s_young_g1)
        model.s_young_mean_g1, _, _ = median_error(s_young_g1, f="mean")

        s_child_g1 = [a.s_child for a ∈ g1]
        model.s_child_median_g1, model.s_child_lerror_g1, model.s_child_herror_g1 = median_error(s_child_g1)
        model.s_child_mean_g1, _, _ = median_error(s_child_g1, f="mean")

        s_end_g1 = [a.s for a ∈ g1]
        model.s_end_median_g1, model.s_end_lerror_g1, model.s_end_herror_g1 = median_error(s_end_g1)
        model.s_end_mean_g1, _, _ = median_error(s_end_g1, f="mean")

        conc_dist_g1 = [a.α + a.β for a ∈ g1]
        model.concentration_g1, model.concentration_lerror_g1, model.concentration_herror_g1 = median_error(conc_dist_g1)

        # soc_h statistics per group
        soc_h_values_g0 = [a.soc_h for a ∈ g0]
        model.soc_h_median_g0, model.soc_h_lerror_g0, model.soc_h_herror_g0 = median_error(soc_h_values_g0)

        soc_h_values_g1 = [a.soc_h for a ∈ g1]
        model.soc_h_median_g1, model.soc_h_lerror_g1, model.soc_h_herror_g1 = median_error(soc_h_values_g1)

        # soc_v statistics per group
        soc_v_values_g0 = [a.soc_v for a ∈ g0]
        model.soc_v_median_g0, model.soc_v_lerror_g0, model.soc_v_herror_g0 = median_error(soc_v_values_g0)

        soc_v_values_g1 = [a.soc_v for a ∈ g1]
        model.soc_v_median_g1, model.soc_v_lerror_g1, model.soc_v_herror_g1 = median_error(soc_v_values_g1)

        # sens statistics per group
        sens_values_g0 = [a.sens for a ∈ g0]
        model.sens_median_g0, model.sens_lerror_g0, model.sens_herror_g0 = median_error(sens_values_g0)

        sens_values_g1 = [a.sens for a ∈ g1]
        model.sens_median_g1, model.sens_lerror_g1, model.sens_herror_g1 = median_error(sens_values_g1)

        # strategy frequencies per group

        model.freq_ub_g0 = length(filter(x -> x.L == 1, g0)) / length(g0)
        model.freq_ub_g1 = length(filter(x -> x.L == 1, g1)) / length(g1)
        
        model.freq_cb_g0 = length(filter(x -> x.L == 2, g0)) / length(g0)
        model.freq_cb_g1 = length(filter(x -> x.L == 2, g1)) / length(g1)
        
        model.freq_pb_g0 = length(filter(x -> x.L == 3, g0)) / length(g0)
        model.freq_pb_g1 = length(filter(x -> x.L == 3, g1)) / length(g1)

        model.freq_parochial_g0 = length(filter(x -> x.parochial, g0)) / length(g0)
        model.freq_parochial_g1 = length(filter(x -> x.parochial, g1)) / length(g1)

    end

	pass_the_torch!(model)
	
end