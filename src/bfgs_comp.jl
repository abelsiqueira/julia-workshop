function metodo_bfgs(f, ∇f, x₀; ϵ = 1e-4, α = 0.5, η = 0.5, kmax = 1000)
  B = eye(length(x₀))
  ef = 0
  x = copy(x₀)
  k = 0
  while norm(∇f(x)) > ϵ
    d = -B\∇f(x)
    dt∇f = dot(d, ∇f(x))
    t = 1.0
    while f(x + t*d) > f(x) + α*t*dt∇f
      t *= η
    end
    s = t*d
    y = ∇f(x+s)-∇f(x)
    x = x + s

    k += 1
    if k > kmax
      ef = 1
      break
    end

    if dot(s,y) > 0.0
      B = B + y*y'/dot(s,y) - B*s*s'*B/dot(s,B*s)
    end
  end
  return x, f(x), ef, k
end

function metodo_bfgs2(f, ∇f, x₀; ϵ = 1e-4, α = 0.5, η = 0.5, kmax = 1000)
  B = eye(length(x₀))
  ef = 0
  x = copy(x₀)
  fx = f(x)
  ∇fx = ∇f(x)
  k = 0
  while norm(∇fx) > ϵ
    d = -B\∇fx
    dt∇f = dot(d, ∇fx)
    t = 1.0
    while f(x + t*d) > fx + α*t*dt∇f
      t *= η
    end
    s = t*d
    x = x + s
    y = ∇fx
    fx = f(x)
    ∇fx = ∇f(x)
    y = ∇fx - y

    k += 1
    if k > kmax
      ef = 1
      break
    end

    sty = dot(s,y)
    if sty > 0.0
      Bs = B*s
      stBs = dot(s,Bs)
      B = B + y*y'/sty - Bs*Bs'/stBs
    end
  end
  return x, fx, ef, k
end

function metodo_bfgs3(f, ∇f, x₀; ϵ = 1e-4, α = 0.5, η = 0.5, kmax = 1000)
  n = length(x₀)
  B = eye(n)
  ef = 0
  x = copy(x₀)
  fx = f(x)
  ∇fx = ∇f(x)
  k = 0
  while norm(∇fx) > ϵ
    d = -B\∇fx
    dt∇f = dot(d, ∇fx)
    t = 1.0
    while f(x + t*d) > fx + α*t*dt∇f
      t *= η
    end
    s = t*d
    x = x + s
    y = ∇fx
    fx = f(x)
    ∇fx = ∇f(x)
    y = ∇fx - y

    k += 1
    if k > kmax
      ef = 1
      break
    end

    sty = dot(s,y)
    if sty > 0.0
      Bs = B*s
      stBs = dot(s,Bs)
      for i = 1:n
        for j = 1:n
          B[i,j] += y[i]*y[j]/sty - Bs[i]*Bs[j]/stBs
        end
      end
    end
  end
  return x, fx, ef, k
end

using CUTEst

function runcutest(mtd)
  lista = ["ROSENBR", "BARD", "ALLINITU", "CUBE", "BEALE", "OSCIPATH"]
  times = zeros(length(lista))
  for (i,p) in enumerate(lista)
    nlp = CUTEstModel(p)
    f(x) = ufn(nlp, x)
    ∇f(x) = ugr(nlp, x)
    try
      x, fx, ef, k = mtd(f, ∇f, nlp.meta.x0)
      times[i] = time()
      @profile for j = 1:10
        x, fx, ef, k = mtd(f, ∇f, nlp.meta.x0)
      end
      times[i] = time() - times[i]
      println("t = $(times[i]), ef = $ef, k = $k, $(norm(∇f(x)))")
    finally
      cutest_finalize(nlp)
    end
  end
  return times
end

using ProfileView

for mtd in [metodo_bfgs, metodo_bfgs2, metodo_bfgs3]
  runcutest(mtd)
  Profile.clear()
  runcutest(mtd)
  ProfileView.svgwrite("$mtd.svg")
end
