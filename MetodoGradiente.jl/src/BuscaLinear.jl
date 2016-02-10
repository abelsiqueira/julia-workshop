export armijo

function armijo(ϕ, ϕd₀; α = 0.5, η = 0.5)
  t = 1.0
  ϕ₀ = ϕ(0.0)
  while ϕ(t) > ϕ₀ + α*t*ϕd₀
    t *= η
  end
  return t
end
