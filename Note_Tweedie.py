import numpy as np
from scipy.stats import gamma,poisson

"""
    Poisson-Gammaの密度関数を実際にプロット.
    GLM(1<p<2)の密度関数と比較して理解を深める.
"""

# In[] パラメタをセット
x = np.linspace(0,30,100)
y = np.zeros([10,100]) #poisson-gammaのpdfの値を入れる
lambda_ = 3
alpha = 3
beta  = 0.5


# In[] 密度関数を計算.N行目はFreq=N+1の場合の密度
for n in range(1,6):
    temp = gamma.pdf(x, n*alpha,1/beta)
    temp *= poisson.pmf(n, lambda_)
    y[n-1,:] = temp
poisson.pmf(0, lambda_) + y.sum()


# In[] 描画 - conditional on N
plt.figure(figsize=(4,3))
for n in range(5):
    plt.plot(x,y[n])
plt.xlabel("severity")
plt.ylabel("prob density")
# plt.savefig("a.pdf",bbox_inches="tight")


# In[] 描画 - summation for N
plt.figure(figsize=(4,3))
plt.plot(x, y.sum(axis=0))
plt.xlabel("severity")
plt.ylabel("prob density")


# In[] 対応するGLMの密度も見てみる
def convertParamset(lambda_, alpha, beta):
    # (λ,α,β) -> (μ,φ,p)
    mu = lambda_ * alpha/beta
    p  = (alpha+2) / (alpha+1)
    phi = (lambda_**(1-p) * (alpha/beta)**(2-p)) / (2-p)
    return [mu, phi, p]

def poissonGammaGLM(x, mu,phi,p):
    logf = (1/phi) * (x * (mu**(1-p))/(1-p) - (mu**(2-p))/(2-p))
    f = np.exp(logf)
    return f


# In[] パラメタを変換し, 密度関数をプロット
mu, phi, p = convertParamset(lambda_, alpha, beta)
print(mu,phi,p)
y_glm = poissonGammaGLM(x, mu,phi,p)
plt.plot(x,y_glm)









# In[]
