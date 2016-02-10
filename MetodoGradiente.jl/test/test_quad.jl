using Base.Test
using MetodoGradiente

function test_quad()
  σ, λ = 0.1, 10
  n = 100
  fails = 0
  maxk = 0
  for N = 1:100
    Λ = rand(n)*(λ-σ) + σ
    f(x) = 0.5*dot(x,Λ.*x)
    ∇f(x) = Λ.*x
    x₀ = ones(n)
    x, fx, ef, k = metodo_gradiente(f, ∇f, x₀)
    maxk = max(maxk, k)
    fails += ef
  end
  @test fails == 0
end

test_quad()
