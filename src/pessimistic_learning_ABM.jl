using StatsBase, Random, Distributions, Agents

########## UTILITY FUNCTIONS ##########

"Return a vector of agents for the given `ids`. This replaces filter calls in the original code."
function get_peeps_by_id(model, ids::Vector{Int})
    # For efficiency we create a new Vector{Peep} of the same length as `ids`.
    # Then fill it in. This avoids a big O(N) filter over all agents.
    peeps = Vector{Peep}(undef, length(ids))
    for (i,id) in enumerate(ids)
        peeps[i] = model[id]  # or Agents.agent_by_id(model, id)
    end
    return peeps
end

"Returns a new vector of length (model.N - 1) containing all agent IDs except `self_id`."
function fill_possible_ids!(buffer::Vector{Int}, self_id::Int, N::Int)
    @inbounds k = 1
    @inbounds for i in 1:N
        if i != self_id
            buffer[k] = i
            k += 1
        end
    end
    return buffer
end

"Simple Kelly stake rule."
function kelly_stake(u)
    return u > 0.5 ? 2u - 1 : 0.0
end

"Return median and quantile-based errors."
function median_error(X; f="median", l=0.25, h=0.75)
    med = f == "median" ? median(X) : mean(X)
    ϵ⁻ = quantile(X, l)
    ϵ⁺ = quantile(X, h)
    return (med, ϵ⁻, ϵ⁺)
end

"Trauma function for stake-pulling."
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

"Maps aleph to the environment's growth factor."
function aleph_transform(aleph; Vb=1)
    return Vb/(1 - aleph)
end


########## MODEL FUNCTIONS ##########

function sample_environment!(model, agents)
    rng = abmrng(model)
    for a ∈ agents
        # reset each agent's alpha/beta, plus initial s_vec_young
        a.α_young = 1.0
        a.β_young = 1.0
        empty!(a.s_vec_young)  # clear old contents if any
        push!(a.s_vec_young, 0.5)
        
        for _ in 1:model.t
            # If model.mixed, use group-based probability, else single model.u
            #p = model.mixed ? model.mixed_u[a.group] : a.u
            p = a.u
            if rand(rng) < p
                a.α_young += 1
            else
                a.β_young += 1
            end
            push!(a.s_vec_young,
                  clamp(2*(a.α_young / (a.α_young + a.β_young)) - 1, 0, 1)
            )
        end
    end
end

function pool!(model, agents)
    rng = abmrng(model)
    # Shuffle agents (as before)
    shuffled_agents = copy(agents)
    shuffle!(rng, shuffled_agents)
    
    for a ∈ shuffled_agents
        # Instead of using a shared buffer, allocate a fresh local buffer:
        local_buffer = Vector{Int}(undef, model.N - 1)
        fill_possible_ids!(local_buffer, a.id, model.N)
        
        # Sample a.models from the locally allocated buffer
        a.models = sample(rng, local_buffer, model.n, replace=false)
        
        # Get the actual agents corresponding to the sampled IDs
        models = get_peeps_by_id(model, a.models)
        
        # If mixed and parochial, filter out agents from a different group:
        if a.parochial
            models = filter(x -> x.group == a.group, models)
        end
        
        if !isempty(models)
            # Sum up alpha and beta from the chosen models:
            model_alphas = sum(b.α_young for b in models)
            model_betas  = sum(b.β_young for b in models)
            
            if model_alphas > 1 && model_betas > 1
                a.α_young += (a.soc_h) * model_alphas
                a.β_young += (a.soc_h) * model_betas
            end
        end
        
        push!(a.s_vec_young,
              clamp(2 * (a.α_young / (a.α_young + a.β_young)) - 1, 0, 1))
        
        # Finalize alpha, beta and compute the child's stake:
        a.α = a.α_young
        a.β = a.β_young
        mean_beta = a.α / (a.α + a.β)
        a.s_child = kelly_stake(mean_beta)
        a.s = a.s_child
    end
end

function learn_from_olds!(model, agents)
    rng = abmrng(model)
    
    for a ∈ agents
        # Allocate a fresh local buffer for sampling "old models"
        local_buffer = Vector{Int}(undef, model.N - 1)
        fill_possible_ids!(local_buffer, a.id, model.N)
        a.old_models = sample(rng, local_buffer, model.m, replace=false)
        
        # Retrieve the actual agent objects
        old_models = get_peeps_by_id(model, a.old_models)
        
        # Apply demographic and parochial filters as before
        if model.demographic_filter
            old_models = filter(x -> x.log_payoff > 0.0, old_models)
        end
        if a.parochial
            old_models = filter(x -> x.group == a.group, old_models)
        end
        
        if !isempty(old_models)
            old_alphas  = [b.α_old for b in old_models]
            old_betas   = [b.β_old for b in old_models]
            old_payoffs = [b.log_payoff for b in old_models]
            
            learned_α = 0.0
            learned_β = 0.0
            if a.L == 1
                learned_α = mean(old_alphas)
                learned_β = mean(old_betas)
            elseif a.L == 2
                learned_α = rand(rng, old_alphas)
                learned_β = rand(rng, old_betas)
            elseif a.L == 3
                highest_payoff_idx = findmax(old_payoffs)[2]
                learned_α = old_alphas[highest_payoff_idx]
                learned_β = old_betas[highest_payoff_idx]
            end
            
            a.α_young = (1 - a.soc_v) * a.α_young + a.soc_v * learned_α
            a.β_young = (1 - a.soc_v) * a.β_young + a.soc_v * learned_β
        end
        
        a.α = a.α_young
        a.β = a.β_young
        push!(a.s_vec_young,
              clamp(2 * (a.α_young / (a.α_young + a.β_young)) - 1, 0, 1))
        mean_beta = a.α_young / (a.α_young + a.β_young)
        a.s_young = kelly_stake(mean_beta)
        a.s = a.s_young
    end
end

function pull_stake!(model, agent, other)
    normalized_difference = abs(other.s - agent.s) / maximum([agent.s, 1 - agent.s])
    #agent.β += trauma(normalized_difference, agent.sens)*agent.β
    agent.β += agent.sens*agent.β
    mean_beta = agent.α / (agent.α + agent.β)
    agent.s = kelly_stake(mean_beta)
end

function play!(model, agents)
    rng = abmrng(model)

    # initialization
    for a ∈ agents
        a.log_payoff = model.mixed ? log(aleph_transform(model.mixed_aleph[a.group])) :
                                     log(aleph_transform(a.ℵ))
        empty!(a.s_vec)
        push!(a.s_vec, last(a.s_vec_young))
        empty!(a.payoff_vec)
        push!(a.payoff_vec, a.log_payoff)
    end

    for _ in 1:model.T
        for a ∈ agents
            if a.log_payoff != -Inf
                u = rand(rng, Beta(a.α, a.β))
                s = kelly_stake(u)

                # group-based or single environment
                #p = model.mixed ? model.mixed_u[a.group] : a.u
                p = a.u

                if rand(rng) < p
                    a.log_payoff += log(1 + s)
                    a.α += 1
                else
                    a.log_payoff += log(1 - s)
                    a.β += 1
                end
                push!(a.payoff_vec, a.log_payoff)

                mean_beta = a.α / (a.α + a.β)
                a.s = kelly_stake(mean_beta)

                # ruin check => pull stakes from agents that reference a
                if a.log_payoff < 0.0
                    a.log_payoff = -Inf
                    # Instead of filter(x-> a.id in x.models, agents), we do a small loop:
                    # (If N is large, consider a "reverse lookup" approach.)
                    for b in agents
                        if b.log_payoff > 0.0
                            # check if b has `a.id` in its models
                            # (turn into a set if you do this a lot)
                            if a.id in b.models
                                pull_stake!(model, b, a)
                            end
                        end
                    end
                end
            else
                push!(a.payoff_vec, a.log_payoff) # maintain the same shape
            end
            push!(a.s_vec, a.s)
            a.s_mean = mean(a.s_vec)
        end
    end

    # final payoff adjustment
    for a ∈ agents
        a.log_payoff -= model.mixed ? log(aleph_transform(model.mixed_aleph[a.group])) :
                                      log(aleph_transform(a.ℵ))
        a.avg_payoff = exp(a.log_payoff / model.T)
    end
end

function pass_the_torch!(model, agents)
    for a in agents
        a.α_old = a.α
        a.β_old = a.β
        a.s_old = a.s
        a.soc_h_old = a.soc_h
        a.soc_v_old = a.soc_v
        a.L_old = a.L
        a.sens_old = a.sens
        a.parochial_old = a.parochial
    end
end

function selection!(model, agents)
    rng = abmrng(model)

    if !model.mixed
        fitness = Vector{Float64}(undef, model.N)
        @inbounds for i in 1:model.N
            a = agents[i]
            fitness[i] = clamp(exp(a.log_payoff / model.T), eps(), Inf)^model.b_coeff
        end
        total_fitness = sum(fitness)
        fitness_weights = weights(fitness ./ total_fitness)

        soc_h_vec = [a.soc_h_old for a ∈ agents]
        soc_v_vec = [a.soc_v_old for a ∈ agents]
        sens_vec = [a.sens_old  for a ∈ agents]
        L_vec = [a.L_old     for a ∈ agents]
        par_vec = [a.parochial_old for a in agents]

        for i in 1:model.N
            peep = agents[i]

            if model.mu_soc_h > 0.0
                if rand(rng) > model.mu_soc_h
                    new_soc_h = sample(rng, soc_h_vec, fitness_weights)
                    peep.soc_h = new_soc_h
                else
                    peep.soc_h = rand(rng)
                end
                peep.soc_h = clamp(rand(rng, Normal(peep.soc_h, model.mu_std)), 0, 1)
            end

            if model.mu_soc_v > 0.0
                if rand(rng) > model.mu_soc_v
                    new_soc_v = sample(rng, soc_v_vec, fitness_weights)
                    peep.soc_v = new_soc_v
                else
                    peep.soc_v = rand(rng)
                end
                peep.soc_v = clamp(rand(rng, Normal(peep.soc_v, model.mu_std)), 0, 1)
            end

            if model.mu_sens > 0.0
                if rand(rng) > model.mu_sens
                    new_sens = sample(rng, sens_vec, fitness_weights)
                    peep.sens = new_sens
                else
                    peep.sens = rand(rng)
                end
                peep.sens = clamp(rand(rng, Normal(peep.sens, model.mu_std)), 0, 1)
            end

            if model.mu_L > 0.0
                if rand(rng) > model.mu_L
                    new_L = sample(rng, L_vec, fitness_weights)
                    peep.L = new_L
                else
                    peep.L = rand(rng, model.strat_pool)
                end
            end

            if model.mu_parochial > 0.0
                if rand(rng) > model.mu_parochial
                    new_par = sample(rng, par_vec, fitness_weights)
                    peep.parochial = new_par
                else
                    peep.parochial = rand(rng, [true,false])
                end
            end

            if model.μ > 0.0
                if rand(rng) < model.μ
                    peep.u = 0.5 + 0.5*rand(rng)
                    peep.group = 2
                else
                    peep.u = model.u
                end
            else
                peep.u = model.u
                peep.group = 1
            end

            peep.ℵ = model.aleph
        end

    else
        # handle each group separately
        g0 = [a for a in agents if a.group == 1]
        g1 = [a for a in agents if a.group == 2]

        if !isempty(g0)
            fitness_g0 = [clamp(exp(a.log_payoff / model.T), eps(), Inf)^model.b_coeff for a ∈ g0]
            fw_g0 = weights(fitness_g0 ./ sum(fitness_g0))
            soc_h_vec_g0 = [a.soc_h_old for a ∈ g0]
            soc_v_vec_g0 = [a.soc_v_old for a ∈ g0]
            sens_vec_g0  = [a.sens_old  for a ∈ g0]
            L_vec_g0     = [a.L_old     for a ∈ g0]
            par_vec_g0   = [a.parochial_old for a ∈ g0]

            for peep ∈ g0
                if model.mu_soc_h > 0.0
                    if rand(rng) > model.mu_soc_h
                        new_soc_h = sample(rng, soc_h_vec_g0, fw_g0)
                        peep.soc_h = new_soc_h
                    else
                        peep.soc_h = rand(rng)
                    end
                    peep.soc_h = clamp(rand(rng, Normal(peep.soc_h, model.mu_std)), 0, 1)
                end

                if model.mu_soc_v > 0.0
                    if rand(rng) > model.mu_soc_v
                        new_soc_v = sample(rng, soc_v_vec_g0, fw_g0)
                        peep.soc_v = new_soc_v
                    else
                        peep.soc_v = rand(rng)
                    end
                    peep.soc_v = clamp(rand(rng, Normal(peep.soc_v, model.mu_std)), 0, 1)
                end

                if model.mu_sens > 0.0
                    if rand(rng) > model.mu_sens
                        new_sens = sample(rng, sens_vec_g0, fw_g0)
                        peep.sens = new_sens
                    else
                        peep.sens = rand(rng)
                    end
                    peep.sens = clamp(rand(rng, Normal(peep.sens, model.mu_std)), 0, 1)
                end

                if model.mu_L > 0.0
                    if rand(rng) > model.mu_L
                        new_L = sample(rng, L_vec_g0, fw_g0)
                        peep.L = new_L
                    else
                        peep.L = rand(rng, model.strat_pool)
                    end
                end

                if model.mu_parochial > 0.0
                    if rand(rng) > model.mu_parochial
                        new_par = sample(rng, par_vec_g0, fw_g0)
                        peep.parochial = new_par
                    else
                        peep.parochial = rand(rng, [true,false])
                    end
                end

                if model.μ > 0.0
                    if rand(rng) < model.μ
                        peep.u = 0.5 + 0.5*rand(rng)
                    else
                        peep.u = model.u
                    end
                else
                    peep.u = model.u
                end
            end
        end

        if !isempty(g1)
            fitness_g1 = [clamp(exp(a.log_payoff / model.T), eps(), Inf)^model.b_coeff for a ∈ g1]
            fw_g1 = weights(fitness_g1 ./ sum(fitness_g1))
            soc_h_vec_g1 = [a.soc_h_old for a ∈ g1]
            soc_v_vec_g1 = [a.soc_v_old for a ∈ g1]
            sens_vec_g1  = [a.sens_old  for a ∈ g1]
            L_vec_g1     = [a.L_old     for a ∈ g1]
            par_vec_g1   = [a.parochial_old for a ∈ g1]

            for peep ∈ g1
                if model.mu_soc_h > 0.0
                    if rand(rng) > model.mu_soc_h
                        new_soc_h = sample(rng, soc_h_vec_g1, fw_g1)
                        peep.soc_h = new_soc_h
                    else
                        peep.soc_h = rand(rng)
                    end
                    peep.soc_h = clamp(rand(rng, Normal(peep.soc_h, model.mu_std)), 0, 1)
                end

                if model.mu_soc_v > 0.0
                    if rand(rng) > model.mu_soc_v
                        new_soc_v = sample(rng, soc_v_vec_g1, fw_g1)
                        peep.soc_v = new_soc_v
                    else
                        peep.soc_v = rand(rng)
                    end
                    peep.soc_v = clamp(rand(rng, Normal(peep.soc_v, model.mu_std)), 0, 1)
                end

                if model.mu_sens > 0.0
                    if rand(rng) > model.mu_sens
                        new_sens = sample(rng, sens_vec_g1, fw_g1)
                        peep.sens = new_sens
                    else
                        peep.sens = rand(rng)
                    end
                    peep.sens = clamp(rand(rng, Normal(peep.sens, model.mu_std)), 0, 1)
                end

                if model.mu_L > 0.0
                    if rand(rng) > model.mu_L
                        new_L = sample(rng, L_vec_g1, fw_g1)
                        peep.L = new_L
                    else
                        peep.L = rand(rng, model.strat_pool)
                    end
                end

                if model.mu_parochial > 0.0
                    if rand(rng) > model.mu_parochial
                        new_par = sample(rng, par_vec_g1, fw_g1)
                        peep.parochial = new_par
                    else
                        peep.parochial = rand(rng, [true,false])
                    end
                end

                if model.μ > 0.0
                    if rand(rng) < model.μ
                        peep.u = 0.5 + 0.5*rand(rng)
                    else
                        peep.u = model.u
                    end
                else
                    peep.u = model.u
                end
            end
        end
    end
end


"Main stepping function – call each sub-step, then gather data."
function model_step!(model)
    agents = allagents(model) |> collect  # single pass for this step
    
    if model.selection
        selection!(model, agents)
    end

    if model.tick != 0 && (model.tick % model.envshift == 0) && !model.mixed && model.λ == 0
        if model.periodic
            if !model.peg_lambda
                current_u = model.u
                model.u = model.u_shift
                model.u_shift = current_u
            end
            if !model.peg_aleph
                current_aleph = model.aleph
                model.aleph = model.aleph_shift
                model.aleph_shift = current_aleph
            end
        else
            model.u = model.u_shift
        end
    else
        model.u = rand(abmrng(model)) < model.λ ? 0.5 + 0.5*rand(abmrng(model)) : model.u
    end

    model.tick += 1

    sample_environment!(model, agents)
    pool!(model, agents)
    if model.tick > 0
        learn_from_olds!(model, agents)
    end
    play!(model, agents)

    # collect data
    if model.tick >= (model.total_ticks)*(0.25)
        collect_data!(model, agents)
    end

    pass_the_torch!(model, agents)
end


########## AGENT & PARAMETER STRUCTS ##########

@agent struct Peep(NoSpaceAgent)
    u::Float64
    ℵ::Float64
    α_young::Float64
    β_young::Float64
    α::Float64
    β::Float64
    s_child::Float64
    s_young::Float64
    s_vec::Vector{Float64}
    s_vec_young::Vector{Float64}
    s::Float64
    s_mean::Float64
    s_median::Float64
    soc_h::Float64
    soc_v::Float64
    L::Int64
    sens::Float64
    group::Int64
    parochial::Bool
    log_payoff::Float64
    avg_payoff::Float64
    payoff_vec::Vector{Float64}
    models::Vector{Int64}
    old_models::Vector{Int64}
    α_old::Float64
    β_old::Float64
    s_old::Float64
    soc_h_old::Float64
    soc_v_old::Float64
    L_old::Int64
    sens_old::Float64
    parochial_old::Bool
end


Base.@kwdef mutable struct Parameters
    # Model parameters
    N::Int64
    n::Int64
    m::Int64
    T::Int64
    t::Int64
    u::Float64
    λ::Float64
    μ::Float64
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
    steps::Int64
    envshift::Int64
    u_shift::Float64
    aleph_shift::Float64
    peg_aleph::Bool
    peg_lambda::Bool
    mixed::Bool
    mixed_freq::Float64
    mixed_aleph::Vector{Float64}
    mixed_aleph_shift::Vector{Float64}
    mixed_u::Vector{Float64}
    mixed_u_shift::Vector{Float64}
    mixed_L::Vector{Int64}
    parochial::Bool
    periodic::Bool
    demographic_filter::Bool
    selection::Bool
    b_coeff::Float64
    randomize::Bool
    seed::Int64
    tick::Int64 = 0
    total_ticks::Int64 = 2500
    online_counter::Int64 = 0
    # Preallocated buffers or references
    possible_ids_buffer::Vector{Int} = Int[]  # For sampling
    # Data fields
    s_mean::Float64 = 0.0
    s_median::Float64 = 0.0
    s_lerror::Float64 = 0.0
    s_herror::Float64 = 0.0
    s_ltail::Float64 = 0.0
    s_htail::Float64 = 0.0
    s_ub_mean::Float64 = 0.0
    s_ub_median::Float64 = 0.0
    s_ub_lerror::Float64 = 0.0
    s_ub_herror::Float64 = 0.0
    s_ub_ltail::Float64 = 0.0
    s_ub_htail::Float64 = 0.0
    s_pb_mean::Float64 = 0.0
    s_pb_median::Float64 = 0.0
    s_pb_lerror::Float64 = 0.0
    s_pb_herror::Float64 = 0.0
    s_pb_ltail::Float64 = 0.0
    s_pb_htail::Float64 = 0.0
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
    freq_parochial::Float64 = 0.0
    freq_ub_g0::Float64 = 0.0
    freq_pb_g0::Float64 = 0.0
    freq_cb_g0::Float64 = 0.0
    freq_parochial_g0::Float64 = 0.0
    freq_ub_g1::Float64 = 0.0
    freq_pb_g1::Float64 = 0.0
    freq_cb_g1::Float64 = 0.0
    freq_parochial_g1::Float64 = 0.0
    # new fields for soc_h
    soc_h_median::Float64 = 0.0
    soc_h_lerror::Float64 = 0.0
    soc_h_herror::Float64 = 0.0
    soc_h_median_g0::Float64 = 0.0
    soc_h_lerror_g0::Float64 = 0.0
    soc_h_herror_g0::Float64 = 0.0
    soc_h_median_g1::Float64 = 0.0
    soc_h_lerror_g1::Float64 = 0.0
    soc_h_herror_g1::Float64 = 0.0
    # new fields for soc_v
    soc_v_median::Float64 = 0.0
    soc_v_lerror::Float64 = 0.0
    soc_v_herror::Float64 = 0.0
    soc_v_median_g0::Float64 = 0.0
    soc_v_lerror_g0::Float64 = 0.0
    soc_v_herror_g0::Float64 = 0.0
    soc_v_median_g1::Float64 = 0.0
    soc_v_lerror_g1::Float64 = 0.0
    soc_v_herror_g1::Float64 = 0.0
    # new fields for sens
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


########## INIT + MODEL STEP ##########

function initialize_pessimistic_learning(; 
    N=1000,
    n=10,
    m=10,
    T=100,
    t=15,
    u=0.65,
    λ=0.0,
    μ=0.0,
    aleph=0.05,
    randomize=false,
    soc_h=0.0,
    soc_v=0.0,
    sens=0.0,
    mu_std=0.0,
    mu_soc_h=0.0,
    mu_soc_v=0.0,
    mu_sens=0.0,
    mu_L=0.0,
    strategies="UB&PB",
    mu_parochial=0.0,
    steps=3,
    envshift=5000,
    u_shift=0.65,
    aleph_shift=0.05,
    peg_aleph=true,
    peg_lambda=false,
    mixed=false,
    mixed_freq=0.5,
    mixed_aleph1=0.05,
    mixed_aleph2=0.95,
    mixed_aleph1_shift=0.95,
    mixed_aleph2_shift=0.95,
    mixed_u1=0.6,
    mixed_u2=0.75,
    mixed_u1_shift=0.65,
    mixed_u2_shift=0.65,
    mixed_L1=1,
    mixed_L2=1,
    parochial=false,
    periodic=true,
    demographic_filter=true,
    selection=true,
    b_coeff=1.0,
    total_ticks=2500,
    seed=123456789
)
    rng = Xoshiro(seed)

    # Strategy pool
    strat_pool = strategies == "UB"      ? [1] :
                 strategies == "CB"      ? [2] :
                 strategies == "PB"      ? [3] :
                 strategies == "UB&CB"   ? [1, 2] :
                 strategies == "UB&PB"   ? [1, 3] :
                 strategies == "CB&PB"   ? [2, 3] :
                 strategies == "ALL"     ? [1, 2, 3] :
                 error("Invalid learning strategy pool.")

    props = Parameters(
        N=N, n=n, m=m, T=T, t=t, u=u, λ=λ, μ=μ, aleph=aleph, soc_h=soc_h, soc_v=soc_v,
        sens=sens, mu_std=mu_std, mu_soc_h=mu_soc_h, mu_soc_v=mu_soc_v,
        mu_sens=mu_sens, mu_L=mu_L, strat_pool=strat_pool,
        mu_parochial=mu_parochial, steps=steps, envshift=envshift,
        u_shift=u_shift, aleph_shift=aleph_shift, peg_aleph=peg_aleph,
        peg_lambda=peg_lambda, mixed=mixed, mixed_freq=mixed_freq,
        mixed_aleph=[mixed_aleph1, mixed_aleph2],
        mixed_aleph_shift=[mixed_aleph1_shift, mixed_aleph2_shift],
        mixed_u=[mixed_u1, mixed_u2],
        mixed_u_shift=[mixed_u1_shift, mixed_u2_shift],
        mixed_L=[mixed_L1, mixed_L2],
        parochial=parochial, periodic=periodic, 
        demographic_filter=demographic_filter,
        selection=selection, b_coeff=b_coeff, randomize=randomize, 
        total_ticks=total_ticks, seed=seed
    )

    # Preallocate the buffer for sampling (N - 1 each time)
    props.possible_ids_buffer = Vector{Int}(undef, N-1)

    # Create the ABM
    model = StandardABM(
        Peep,
        nothing;   # no spatial structure
        properties=props,
        rng=rng,
        model_step! = model_step!
    )

    model.u = model.randomize ? 0.5 + 0.5*rand(abmrng(model)) : model.u

    # Initialize agents
    agent_ids = collect(1:N)
    for a_id in agent_ids
        group = mixed ? (rand(rng) < mixed_freq ? 1 : 2) : 1
        # Sample "models" just once here for initialization
        buffer = Vector{Int}(undef, N-1)
        fill_possible_ids!(buffer, a_id, N)
        init_models = sample(rng, buffer, n, replace=false)

        L_value = mixed ? props.mixed_L[group] : rand(rng, strat_pool)
        # Potential random values for soc_h / soc_v / sens if selection & mutation
        # But we keep the initial code logic: use the given soc_h, etc.
        peep = Peep(
            id = a_id,
            u = model.μ > 0.0 ? ( rand(abmrng(model)) < model.μ ? 0.5 + 0.5*rand(abmrng(model)) : model.u ) : model.u,
            ℵ = model.aleph,
            α_young = 1.0,
            β_young = 1.0,
            α = 0.0,
            β = 0.0,
            s_child = 0.0,
            s_young = 0.0,
            s_vec = Float64[],
            s_vec_young = Float64[],
            s = 0.0,
            s_mean = 0.0,
            s_median = 0.0,
            soc_h = selection && mu_soc_h > 0.0 ? rand(rng) : soc_h,
            soc_v = selection && mu_soc_v > 0.0 ? rand(rng) : soc_v,
            L = L_value,
            sens = selection && mu_sens > 0.0 ? rand(rng) : sens,
            group = group,
            parochial = mu_parochial > 0 ? rand(rng, [true,false]) : parochial,
            log_payoff = 0.0,
            avg_payoff = 0.0,
            payoff_vec = Float64[],
            models = init_models,
            old_models = Int[],
            α_old = 0.0,
            β_old = 0.0,
            s_old = 0.0,
            soc_h_old = 0.0,
            soc_v_old = 0.0,
            L_old = 0,
            sens_old = 0.0,
            parochial_old = false
        )
        add_agent!(peep, model)
    end

    # Now do the initial steps
    agents = allagents(model) |> collect  # single pass
    if steps >= 1
        sample_environment!(model, agents)
    end
    if steps >= 2
        pool!(model, agents)
    end
    if steps >= 3
        play!(model, agents)
    end

    pass_the_torch!(model, agents)
    return model
end

"""
    collect_data!(model, agents)

Incrementally update (in-place) all of the model’s data fields to hold running averages
(or running means of step-wise medians/quantiles) across all calls to `collect_data!`.
Requires that `model.online_counter::Int64` exists to track the number of updates so far.
"""
function collect_data!(model, agents)
    props = model  # If your code references `model.s_mean`, etc., then `model` holds them
    peeps = agents

    # Bump our counter for an incremental update this round
    props.online_counter += 1
    n = props.online_counter

    # Helper for one-line incremental updates (simple "running average"):
    incremental_update(old, new) = old + (new - old) / n

    # --- s_mean, s_median, etc. ---
    # First gather the "current" step's distribution of s_mean
    s_dist = [a.s_mean for a in peeps]

    # Compute the step-based median & mean
    step_median, step_lerror, step_herror = median_error(s_dist)
    step_mean, step_ltail, step_htail     = median_error(s_dist, f="mean", l=0.05, h=0.95)

    # Now update model.s_median in-place as an average-of-medians so far
    props.s_median      = incremental_update(props.s_median, step_median)
    props.s_lerror      = incremental_update(props.s_lerror, step_lerror)
    props.s_herror      = incremental_update(props.s_herror, step_herror)

    props.s_mean        = incremental_update(props.s_mean, step_mean)
    props.s_ltail       = incremental_update(props.s_ltail, step_ltail)
    props.s_htail       = incremental_update(props.s_htail, step_htail)

    # --- Unbiased group (L==1) ---
    ub_peeps = filter(x -> x.L == 1, peeps)
    s_ub = [a.s_mean for a in ub_peeps]

    if !isempty(s_ub)
        ub_median, ub_lerror, ub_herror = median_error(s_ub)
        ub_mean, ub_ltail, ub_htail     = median_error(s_ub, f="mean", l=0.05, h=0.95)

        props.s_ub_median = incremental_update(props.s_ub_median, ub_median)
        props.s_ub_lerror = incremental_update(props.s_ub_lerror, ub_lerror)
        props.s_ub_herror = incremental_update(props.s_ub_herror, ub_herror)

        props.s_ub_mean   = incremental_update(props.s_ub_mean,   ub_mean)
        props.s_ub_ltail  = incremental_update(props.s_ub_ltail,  ub_ltail)
        props.s_ub_htail  = incremental_update(props.s_ub_htail,  ub_htail)
    end

    # --- Payoff-bias group (L==3) ---
    pb_peeps = filter(x -> x.L == 3, peeps)
    s_pb = [a.s_mean for a in pb_peeps]

    if !isempty(s_pb)
        pb_median, pb_lerror, pb_herror = median_error(s_pb)
        pb_mean, pb_ltail, pb_htail     = median_error(s_pb, f="mean", l=0.05, h=0.95)

        props.s_pb_median = incremental_update(props.s_pb_median, pb_median)
        props.s_pb_lerror = incremental_update(props.s_pb_lerror, pb_lerror)
        props.s_pb_herror = incremental_update(props.s_pb_herror, pb_herror)

        props.s_pb_mean   = incremental_update(props.s_pb_mean,   pb_mean)
        props.s_pb_ltail  = incremental_update(props.s_pb_ltail,  pb_ltail)
        props.s_pb_htail  = incremental_update(props.s_pb_htail,  pb_htail)
    end

    # s change statistics
    s_vecs = [a.s_vec for a in peeps]
    transposed = [getindex.(s_vecs, i) for i in 1:length(s_vecs[1])]
	mean_trajectory = mean.(transposed)

    inc = [1.0]
    for i in 1:length(mean_trajectory)
        if i > 1
            push!( inc, (mean_trajectory[i]/first(mean_trajectory)) )
        end
    end

    props.sbar = mean(mean_trajectory)
    props.mean_increment = mean(inc)

    # --- s_young ---
    s_dist_young = [a.s_young for a in peeps]
    step_young_median, step_young_lerr, step_young_herr = median_error(s_dist_young)
    step_young_mean, _, _ = median_error(s_dist_young, f="mean")

    props.s_young_median = incremental_update(props.s_young_median, step_young_median)
    props.s_young_lerror = incremental_update(props.s_young_lerror, step_young_lerr)
    props.s_young_herror = incremental_update(props.s_young_herror, step_young_herr)
    props.s_young_mean   = incremental_update(props.s_young_mean,   step_young_mean)

    # --- s_child ---
    s_dist_child = [a.s_child for a in peeps]
    step_child_median, step_child_lerr, step_child_herr = median_error(s_dist_child)
    step_child_mean, _, _ = median_error(s_dist_child, f="mean")

    props.s_child_median = incremental_update(props.s_child_median, step_child_median)
    props.s_child_lerror = incremental_update(props.s_child_lerror, step_child_lerr)
    props.s_child_herror = incremental_update(props.s_child_herror, step_child_herr)
    props.s_child_mean   = incremental_update(props.s_child_mean,   step_child_mean)

    # --- s_end ---
    s_dist_end = [a.s for a in peeps]
    step_end_median, step_end_lerr, step_end_herr = median_error(s_dist_end)
    step_end_mean, _, _ = median_error(s_dist_end, f="mean")

    props.s_end_median = incremental_update(props.s_end_median, step_end_median)
    props.s_end_lerror = incremental_update(props.s_end_lerror, step_end_lerr)
    props.s_end_herror = incremental_update(props.s_end_herror, step_end_herr)
    props.s_end_mean   = incremental_update(props.s_end_mean,   step_end_mean)

    # --- concentration (α + β) ---
    conc_dist = [a.α + a.β for a in peeps]
    conc_median, conc_le, conc_he = median_error(conc_dist)
    props.concentration         = incremental_update(props.concentration,         conc_median)
    props.concentration_lerror  = incremental_update(props.concentration_lerror,  conc_le)
    props.concentration_herror  = incremental_update(props.concentration_herror,  conc_he)

    # --- soc_h, soc_v, sens ---
    soc_h_values = [a.soc_h for a in peeps]
    h_median, h_le, h_he = median_error(soc_h_values)
    props.soc_h_median   = incremental_update(props.soc_h_median, h_median)
    props.soc_h_lerror   = incremental_update(props.soc_h_lerror, h_le)
    props.soc_h_herror   = incremental_update(props.soc_h_herror, h_he)

    soc_v_values = [a.soc_v for a in peeps]
    v_median, v_le, v_he = median_error(soc_v_values)
    props.soc_v_median   = incremental_update(props.soc_v_median, v_median)
    props.soc_v_lerror   = incremental_update(props.soc_v_lerror, v_le)
    props.soc_v_herror   = incremental_update(props.soc_v_herror, v_he)

    sens_values = [a.sens for a in peeps]
    sens_median, sens_le, sens_he = median_error(sens_values)
    props.sens_median            = incremental_update(props.sens_median, sens_median)
    props.sens_lerror            = incremental_update(props.sens_lerror, sens_le)
    props.sens_herror            = incremental_update(props.sens_herror, sens_he)

    # --- geometric mean payoff (Vbar) ---
    step_Vbar = mean(exp.([a.log_payoff / props.T for a in peeps]))
    props.Vbar = step_Vbar#incremental_update(props.Vbar, step_Vbar)

    # --- frequencies (UB, CB, PB, PAROCHIAL) ---
    step_freq_ub = count(a -> a.L == 1, peeps) / props.N
    step_freq_cb = count(a -> a.L == 2, peeps) / props.N
    step_freq_pb = count(a -> a.L == 3, peeps) / props.N
    step_freq_parochial = count(a -> a.parochial, peeps) / props.N

    props.freq_ub = incremental_update(props.freq_ub, step_freq_ub)
    props.freq_cb = incremental_update(props.freq_cb, step_freq_cb)
    props.freq_pb = incremental_update(props.freq_pb, step_freq_pb)
    props.freq_parochial = incremental_update(props.freq_parochial, step_freq_parochial)

    # --- if model.mixed, do the group g0/g1 stats likewise
    if props.mixed
        g0 = filter(a -> a.group == 1, peeps)
        g1 = filter(a -> a.group == 2, peeps)

        if !isempty(g0)
            s_mean_g0 = [a.s_mean for a in g0]
            mg0, le_g0, he_g0 = median_error(s_mean_g0)
            mg0_mean, mg0_lt, mg0_ht = median_error(s_mean_g0, f="mean", l=0.05, h=0.95)

            props.s_median_g0 = incremental_update(props.s_median_g0, mg0)
            props.s_lerror_g0 = incremental_update(props.s_lerror_g0, le_g0)
            props.s_herror_g0 = incremental_update(props.s_herror_g0, he_g0)

            props.s_mean_g0   = incremental_update(props.s_mean_g0,   mg0_mean)
            props.s_ltail_g0  = incremental_update(props.s_ltail_g0,  mg0_lt)
            props.s_htail_g0  = incremental_update(props.s_htail_g0,  mg0_ht)

            # s_young_g0
            sy_g0 = [a.s_young for a in g0]
            sy_mg0, sy_le_g0, sy_he_g0 = median_error(sy_g0)
            sy_mg0_mean, _, _ = median_error(sy_g0, f="mean")

            props.s_young_median_g0 = incremental_update(props.s_young_median_g0, sy_mg0)
            props.s_young_lerror_g0 = incremental_update(props.s_young_lerror_g0, sy_le_g0)
            props.s_young_herror_g0 = incremental_update(props.s_young_herror_g0, sy_he_g0)
            props.s_young_mean_g0   = incremental_update(props.s_young_mean_g0,   sy_mg0_mean)

            # s_child_g0
            sc_g0 = [a.s_child for a in g0]
            sc_mg0, sc_le_g0, sc_he_g0 = median_error(sc_g0)
            sc_mg0_mean, _, _ = median_error(sc_g0, f="mean")

            props.s_child_median_g0 = incremental_update(props.s_child_median_g0, sc_mg0)
            props.s_child_lerror_g0 = incremental_update(props.s_child_lerror_g0, sc_le_g0)
            props.s_child_herror_g0 = incremental_update(props.s_child_herror_g0, sc_he_g0)
            props.s_child_mean_g0   = incremental_update(props.s_child_mean_g0,   sc_mg0_mean)

            # s_end_g0
            se_g0 = [a.s for a in g0]
            se_mg0, se_le_g0, se_he_g0 = median_error(se_g0)
            se_mg0_mean, _, _ = median_error(se_g0, f="mean")

            props.s_end_median_g0 = incremental_update(props.s_end_median_g0, se_mg0)
            props.s_end_lerror_g0 = incremental_update(props.s_end_lerror_g0, se_le_g0)
            props.s_end_herror_g0 = incremental_update(props.s_end_herror_g0, se_he_g0)
            props.s_end_mean_g0   = incremental_update(props.s_end_mean_g0,   se_mg0_mean)

            # concentration_g0
            conc_g0 = [a.α + a.β for a in g0]
            c_mg0, c_le_g0, c_he_g0 = median_error(conc_g0)
            props.concentration_g0         = incremental_update(props.concentration_g0, c_mg0)
            props.concentration_lerror_g0  = incremental_update(props.concentration_lerror_g0, c_le_g0)
            props.concentration_herror_g0  = incremental_update(props.concentration_herror_g0, c_he_g0)

            # soc_h_g0
            soc_h_g0 = [a.soc_h for a in g0]
            h_med_g0, h_le_g0, h_he_g0 = median_error(soc_h_g0)
            props.soc_h_median_g0 = incremental_update(props.soc_h_median_g0, h_med_g0)
            props.soc_h_lerror_g0 = incremental_update(props.soc_h_lerror_g0, h_le_g0)
            props.soc_h_herror_g0 = incremental_update(props.soc_h_herror_g0, h_he_g0)

            # soc_v_g0
            soc_v_g0 = [a.soc_v for a in g0]
            v_med_g0, v_le_g0, v_he_g0 = median_error(soc_v_g0)
            props.soc_v_median_g0 = incremental_update(props.soc_v_median_g0, v_med_g0)
            props.soc_v_lerror_g0 = incremental_update(props.soc_v_lerror_g0, v_le_g0)
            props.soc_v_herror_g0 = incremental_update(props.soc_v_herror_g0, v_he_g0)

            # sens_g0
            sens_g0 = [a.sens for a in g0]
            sens_med_g0, sens_le_g0, sens_he_g0 = median_error(sens_g0)
            props.sens_median_g0 = incremental_update(props.sens_median_g0, sens_med_g0)
            props.sens_lerror_g0 = incremental_update(props.sens_lerror_g0, sens_le_g0)
            props.sens_herror_g0 = incremental_update(props.sens_herror_g0, sens_he_g0)

            # freq_ub_g0, freq_cb_g0, freq_pb_g0
            f_ub_g0 = count(x -> x.L == 1, g0) / length(g0)
            f_cb_g0 = count(x -> x.L == 2, g0) / length(g0)
            f_pb_g0 = count(x -> x.L == 3, g0) / length(g0)

            props.freq_ub_g0 = incremental_update(props.freq_ub_g0, f_ub_g0)
            props.freq_cb_g0 = incremental_update(props.freq_cb_g0, f_cb_g0)
            props.freq_pb_g0 = incremental_update(props.freq_pb_g0, f_pb_g0)
            props.freq_parochial_g0 = incremental_update(props.freq_parochial_g0,
                count(x -> x.parochial, g0) / length(g0))
        end

        if !isempty(g1)
            s_mean_g1 = [a.s_mean for a in g1]
            mg1, le_g1, he_g1 = median_error(s_mean_g1)
            mg1_mean, mg1_lt, mg1_ht = median_error(s_mean_g1, f="mean", l=0.05, h=0.95)

            props.s_median_g1 = incremental_update(props.s_median_g1, mg1)
            props.s_lerror_g1 = incremental_update(props.s_lerror_g1, le_g1)
            props.s_herror_g1 = incremental_update(props.s_herror_g1, he_g1)

            props.s_mean_g1   = incremental_update(props.s_mean_g1,   mg1_mean)
            props.s_ltail_g1  = incremental_update(props.s_ltail_g1,  mg1_lt)
            props.s_htail_g1  = incremental_update(props.s_htail_g1,  mg1_ht)

            # s_young_g1
            sy_g1 = [a.s_young for a in g1]
            sy_mg1, sy_le_g1, sy_he_g1 = median_error(sy_g1)
            sy_mg1_mean, _, _ = median_error(sy_g1, f="mean")

            props.s_young_median_g1 = incremental_update(props.s_young_median_g1, sy_mg1)
            props.s_young_lerror_g1 = incremental_update(props.s_young_lerror_g1, sy_le_g1)
            props.s_young_herror_g1 = incremental_update(props.s_young_herror_g1, sy_he_g1)
            props.s_young_mean_g1   = incremental_update(props.s_young_mean_g1,   sy_mg1_mean)

            # s_child_g1
            sc_g1 = [a.s_child for a in g1]
            sc_mg1, sc_le_g1, sc_he_g1 = median_error(sc_g1)
            sc_mg1_mean, _, _ = median_error(sc_g1, f="mean")

            props.s_child_median_g1 = incremental_update(props.s_child_median_g1, sc_mg1)
            props.s_child_lerror_g1 = incremental_update(props.s_child_lerror_g1, sc_le_g1)
            props.s_child_herror_g1 = incremental_update(props.s_child_herror_g1, sc_he_g1)
            props.s_child_mean_g1   = incremental_update(props.s_child_mean_g1,   sc_mg1_mean)

            # s_end_g1
            se_g1 = [a.s for a in g1]
            se_mg1, se_le_g1, se_he_g1 = median_error(se_g1)
            se_mg1_mean, _, _ = median_error(se_g1, f="mean")

            props.s_end_median_g1 = incremental_update(props.s_end_median_g1, se_mg1)
            props.s_end_lerror_g1 = incremental_update(props.s_end_lerror_g1, se_le_g1)
            props.s_end_herror_g1 = incremental_update(props.s_end_herror_g1, se_he_g1)
            props.s_end_mean_g1   = incremental_update(props.s_end_mean_g1,   se_mg1_mean)

            # concentration_g1
            conc_g1 = [a.α + a.β for a in g1]
            c_mg1, c_le_g1, c_he_g1 = median_error(conc_g1)
            props.concentration_g1         = incremental_update(props.concentration_g1, c_mg1)
            props.concentration_lerror_g1  = incremental_update(props.concentration_lerror_g1, c_le_g1)
            props.concentration_herror_g1  = incremental_update(props.concentration_herror_g1, c_he_g1)

            # soc_h_g1
            soc_h_g1 = [a.soc_h for a in g1]
            h_med_g1, h_le_g1, h_he_g1 = median_error(soc_h_g1)
            props.soc_h_median_g1 = incremental_update(props.soc_h_median_g1, h_med_g1)
            props.soc_h_lerror_g1 = incremental_update(props.soc_h_lerror_g1, h_le_g1)
            props.soc_h_herror_g1 = incremental_update(props.soc_h_herror_g1, h_he_g1)

            # soc_v_g1
            soc_v_g1 = [a.soc_v for a in g1]
            v_med_g1, v_le_g1, v_he_g1 = median_error(soc_v_g1)
            props.soc_v_median_g1 = incremental_update(props.soc_v_median_g1, v_med_g1)
            props.soc_v_lerror_g1 = incremental_update(props.soc_v_lerror_g1, v_le_g1)
            props.soc_v_herror_g1 = incremental_update(props.soc_v_herror_g1, v_he_g1)

            # sens_g1
            sens_g1 = [a.sens for a in g1]
            sens_med_g1, sens_le_g1, sens_he_g1 = median_error(sens_g1)
            props.sens_median_g1 = incremental_update(props.sens_median_g1, sens_med_g1)
            props.sens_lerror_g1 = incremental_update(props.sens_lerror_g1, sens_le_g1)
            props.sens_herror_g1 = incremental_update(props.sens_herror_g1, sens_he_g1)

            # freq_ub_g1, freq_cb_g1, freq_pb_g1
            f_ub_g1 = count(x -> x.L == 1, g1) / length(g1)
            f_cb_g1 = count(x -> x.L == 2, g1) / length(g1)
            f_pb_g1 = count(x -> x.L == 3, g1) / length(g1)

            props.freq_ub_g1 = incremental_update(props.freq_ub_g1, f_ub_g1)
            props.freq_cb_g1 = incremental_update(props.freq_cb_g1, f_cb_g1)
            props.freq_pb_g1 = incremental_update(props.freq_pb_g1, f_pb_g1)
            props.freq_parochial_g1 = incremental_update(props.freq_parochial_g1, count(x -> x.parochial, g1) / length(g1))
        end
    end
end