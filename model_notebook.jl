### A Pluto.jl notebook ###
# v0.19.19

using Markdown
using InteractiveUtils

# ╔═╡ c23c8120-ade5-11ed-0860-f71646149c4f
begin
	using Pkg
	Pkg.activate("./VarianceAverseSocialLearning/")
	using Revise
end

# ╔═╡ a631fc0c-2b68-4e46-8f01-b2397627f9cc
using Distributions

# ╔═╡ c7b956a1-efba-4216-abb7-6ccac953f552
using StatsBase, StatsPlots

# ╔═╡ 123588bb-700b-4306-a078-48ca29b8c1d0
using Plots

# ╔═╡ 03251ac1-672d-43bc-94eb-87c938745d69
import VarianceAverseSocialLearning as VA

# ╔═╡ 4b1b35dc-d9c5-4d8e-a416-735bd6ec6a0e
function rdeu_power(array::Vector, δ::Number; w=[])
	
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

# ╔═╡ e32e2113-02e2-47e8-831c-f1f18ee38754
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

# ╔═╡ d51d529a-570a-4c6e-bdf2-ef6b80341aed
calculate_stake(est) = 2*est - 1 > 0 ? 2*est - 1 : 0

# ╔═╡ 8aec8f63-a83d-4a34-a16c-3046c69ed918
λ = 1.5; u = 0.85; n = 10

# ╔═╡ da6a3bfe-a34f-4f85-b7b6-e737629502ab
begin
	sim_rates = VA.simulate_success_rates(λ=λ, n=n, u=u)
	histogram(
		sim_rates,
		xlab="sampled success rates",
		legend=false
	)
end

# ╔═╡ f3e3d745-4bb2-42e2-887b-1f8664f8c1a0
estimates = calculate_estimates(n, 5, sim_rates)

# ╔═╡ 235a5b9d-942a-4103-9850-1ef833e376c5
estimates

# ╔═╡ 4ada7ef2-8de4-4aaa-8136-3683603ac088
rdeu_power(estimates, 4)

# ╔═╡ 3c82261a-ccb4-470c-8907-237b1363e5ec
est_range = maximum(estimates) - minimum(estimates)

# ╔═╡ c820575e-6206-4497-92db-d78ac52ccf6c
var(estimates)

# ╔═╡ b3f37648-25ca-4bfe-949b-1c46742c4fac
est = mean(estimates)

# ╔═╡ 571abc34-4f93-4bfc-beca-90e4cabc2d39
stake = 2*est - 1 > 0 ? 2*est - 1 : 0

# ╔═╡ 29935d43-ade9-4903-b1db-6e36afbbaf11
failure_prob = ( 1 / 2u )^λ

# ╔═╡ bc89c379-3b64-475e-b2d2-e954fe524dde
function simulate_throws(u, λ;
	log_capital=0,
	log_benefit=0,
	seasons=100, 
	rounds=10, 
	weight=1.0
)
	for i in 1:seasons
		rate = u / rand( Pareto(λ) )
		for j in 1:rounds
			if rand() < rate
				log_capital = log(1 + stake*weight) + log_capital + log_benefit
			else
				log_capital = log(1 - stake*weight) + log_capital
			end
		end
	end
	
	return log_capital
	
end

# ╔═╡ dc3eda2a-3cf4-4c56-a387-8482516fb1e1
w1 = 0.9; w2 = 0.7; w3 = 0.5; w4 = 0.2; w5 = 1 - est_range^2

# ╔═╡ 2ec407ff-0016-4e0e-b8ff-e078b53a98ba
no_weighting = [simulate_throws(u, λ) for i in 1:100]

# ╔═╡ 384421af-fcda-4313-9384-15546860e787
weighting_1 = [simulate_throws(u, λ, weight=w1) for i in 1:100]

# ╔═╡ 53a3e7a7-0355-44e8-89d5-fcc678689f92
weighting_2 = [simulate_throws(u, λ, weight=w2) for i in 1:100]

# ╔═╡ 620e9dec-b775-4ced-8650-323929ccfab7
weighting_3 = [simulate_throws(u, λ, weight=w3) for i in 1:100]

# ╔═╡ b5b55861-1386-4e3e-ae7c-7209f1312760
weighting_4 = [simulate_throws(u, λ, weight=w4) for i in 1:100]

# ╔═╡ a194412c-51f5-46b9-ad97-f48a54bcaebf
weighting_5 = [simulate_throws(u, λ, weight=w5) for i in 1:100]

# ╔═╡ 8065a879-0159-42a0-bd03-7408538dfc29
mean( no_weighting )

# ╔═╡ fa412177-17dd-47f6-9dba-64ed7e55eae0
mean( weighting_1 )

# ╔═╡ 04d55b99-4349-4a81-be91-cae77afa8045
mean( weighting_2 )

# ╔═╡ 7a183dca-741f-4ff3-b7af-a0489e3b8fb2
mean( weighting_3 )

# ╔═╡ 20e50b98-624c-460d-8643-2130656f5315
mean( weighting_4 )

# ╔═╡ 17622d19-7022-492b-9650-664dc7da3aab
mean( weighting_5 )

# ╔═╡ 3d7307a4-bd46-44cc-89d2-fbe9ea14c35b
delta = 1 + 0.001

# ╔═╡ 72034ece-78ce-409f-8609-2dc94a1d5d74
begin
	histogram( delta.^no_weighting, alpha=0.5, label="no weighting" )
	#histogram!( delta.^weighting_1, alpha=0.5, label="weighting by $w1" )
	#histogram!( delta.^weighting_2, alpha=0.5, label="weighting by $w2" )
	histogram!( delta.^weighting_3, alpha=0.5, label="weighting by $w3" )
	histogram!( delta.^weighting_4, alpha=0.5, label="weighting by $w4" )
end

# ╔═╡ e3d793d5-8872-400a-994b-f6c0b4ff7004
function simulate_throws_v2(u, λ, stake;
	log_capital=0,
	log_benefit=0,
	seasons=100, 
	rounds=10
)
	for i in 1:seasons
		rate = u / rand( Pareto(λ) )
		for j in 1:rounds
			if rand() < rate
				log_capital = log(1 + stake) + log_capital + log_benefit
			else
				log_capital = log(1 - stake) + log_capital
			end
		end
	end
	
	return log_capital
	
end

# ╔═╡ 60acd41f-2208-4438-84b0-3792f98c28ff
λ2 = 1.5; n2 = 20; tries = 15; δ = 0.001

# ╔═╡ 3233c69d-fcac-44ad-b2dd-6a6c0f8e4157
histogram( rand(Pareto(λ2), 1000), legend=false )

# ╔═╡ 72068b7d-648a-456e-8d78-2e94e475c475
sim_rates2 = VA.simulate_success_rates(λ=λ2, n=n2, u=u)

# ╔═╡ 600bf9be-7dfa-4f5c-b059-8bfb57dff209
estimates2 = calculate_estimates(n2, tries, sim_rates2)

# ╔═╡ 2f7ab3c6-2b5a-44e7-8c26-0efb7f95e0e4
estimated_stakes = calculate_stake.(estimates2)

# ╔═╡ 402fe706-80d1-4ab9-a96e-e5a4daee8a4b
deltas = 1:1:5

# ╔═╡ c5a1e197-eeec-46ac-a810-71e936238cb4
pweighted_stakes = [rdeu_power(estimated_stakes, d) for d in deltas]

# ╔═╡ 3816ce75-cd04-4f37-a090-e12e3cf9fd20
sims = [ 
	[simulate_throws_v2(u, λ2, s) for i in 1:1000] 
	for s in pweighted_stakes 
]

# ╔═╡ 479b5375-b49d-4cb8-8e40-55a91c7e28bb
begin
	histogram( (1 + δ).^sims[1], alpha=0.5, label="δ = $(deltas[1])" )
	#histogram!( (1 + δ).^sims[2], alpha=0.5, label="δ = $(deltas[2])" )
	histogram!( (1 + δ).^sims[3], alpha=0.5, label="δ = $(deltas[3])" )
	#histogram!( (1 + δ).^sims[4], alpha=0.5, label="δ =  $(deltas[4])" )
	histogram!( (1 + δ).^sims[5], alpha=0.5, label="δ =  $(deltas[5])" )
end

# ╔═╡ d50b1702-1b47-46d5-9b13-0316d600b934
survivals = [filter(x -> x > 0, sims[i]) |> length for i in deltas]

# ╔═╡ Cell order:
# ╠═c23c8120-ade5-11ed-0860-f71646149c4f
# ╠═03251ac1-672d-43bc-94eb-87c938745d69
# ╠═a631fc0c-2b68-4e46-8f01-b2397627f9cc
# ╠═c7b956a1-efba-4216-abb7-6ccac953f552
# ╠═123588bb-700b-4306-a078-48ca29b8c1d0
# ╠═4b1b35dc-d9c5-4d8e-a416-735bd6ec6a0e
# ╠═e32e2113-02e2-47e8-831c-f1f18ee38754
# ╠═d51d529a-570a-4c6e-bdf2-ef6b80341aed
# ╠═8aec8f63-a83d-4a34-a16c-3046c69ed918
# ╠═da6a3bfe-a34f-4f85-b7b6-e737629502ab
# ╠═f3e3d745-4bb2-42e2-887b-1f8664f8c1a0
# ╠═235a5b9d-942a-4103-9850-1ef833e376c5
# ╠═4ada7ef2-8de4-4aaa-8136-3683603ac088
# ╠═3c82261a-ccb4-470c-8907-237b1363e5ec
# ╠═c820575e-6206-4497-92db-d78ac52ccf6c
# ╠═b3f37648-25ca-4bfe-949b-1c46742c4fac
# ╠═571abc34-4f93-4bfc-beca-90e4cabc2d39
# ╠═29935d43-ade9-4903-b1db-6e36afbbaf11
# ╠═bc89c379-3b64-475e-b2d2-e954fe524dde
# ╠═dc3eda2a-3cf4-4c56-a387-8482516fb1e1
# ╠═2ec407ff-0016-4e0e-b8ff-e078b53a98ba
# ╠═384421af-fcda-4313-9384-15546860e787
# ╠═53a3e7a7-0355-44e8-89d5-fcc678689f92
# ╠═620e9dec-b775-4ced-8650-323929ccfab7
# ╠═b5b55861-1386-4e3e-ae7c-7209f1312760
# ╠═a194412c-51f5-46b9-ad97-f48a54bcaebf
# ╠═8065a879-0159-42a0-bd03-7408538dfc29
# ╠═fa412177-17dd-47f6-9dba-64ed7e55eae0
# ╠═04d55b99-4349-4a81-be91-cae77afa8045
# ╠═7a183dca-741f-4ff3-b7af-a0489e3b8fb2
# ╠═20e50b98-624c-460d-8643-2130656f5315
# ╠═17622d19-7022-492b-9650-664dc7da3aab
# ╠═3d7307a4-bd46-44cc-89d2-fbe9ea14c35b
# ╠═72034ece-78ce-409f-8609-2dc94a1d5d74
# ╠═e3d793d5-8872-400a-994b-f6c0b4ff7004
# ╠═3233c69d-fcac-44ad-b2dd-6a6c0f8e4157
# ╠═60acd41f-2208-4438-84b0-3792f98c28ff
# ╠═72068b7d-648a-456e-8d78-2e94e475c475
# ╠═600bf9be-7dfa-4f5c-b059-8bfb57dff209
# ╠═2f7ab3c6-2b5a-44e7-8c26-0efb7f95e0e4
# ╠═402fe706-80d1-4ab9-a96e-e5a4daee8a4b
# ╠═c5a1e197-eeec-46ac-a810-71e936238cb4
# ╠═3816ce75-cd04-4f37-a090-e12e3cf9fd20
# ╠═479b5375-b49d-4cb8-8e40-55a91c7e28bb
# ╠═d50b1702-1b47-46d5-9b13-0316d600b934
