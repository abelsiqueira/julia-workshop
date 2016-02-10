module MetodoGradiente

include("BuscaLinear.jl")

export metodo_gradiente

function metodo_gradiente(f, ∇f, x₀; ϵ = 1e-4, α = 0.5, η = 0.5, kmax = 1000)
    ef = 0
    x = copy(x₀)
    k = 0
    fx = f(x)
    ∇fx = ∇f(x)
    while norm(∇fx) > ϵ
        t = armijo(t->f(x - t*∇fx), -dot(∇fx,∇fx))
        x = x - t*∇fx
        fx = f(x)
        ∇fx = ∇f(x)
        k += 1
        if k > kmax
            ef = 1
            break
        end
    end
    return x, fx, ef, k
end

end
