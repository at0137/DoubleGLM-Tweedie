import numpy as np
import pandas as pd
from scipy.stats import gamma,poisson

from TweedieGLM import DoubleGLM

"""
    feature(x,z),β,γ,pを所与として, (mu,phi,p)を算出
    (mu,phi,p) => (lambda,alpha,beta)のPo-Gaに従う乱数yを生成.
    (y;x,z)の組から(β,γ,p)を推定し,正解と比較する.
"""

# In[] 関数定義:
def convertParamset1(lambda_, alpha, beta):
    # (λ,α,β) -> (μ,φ,p)
    mu = lambda_ * alpha/beta
    p  = (alpha+2) / (alpha+1)
    phi = (lambda_**(1-p) * (alpha/beta)**(2-p)) / (2-p)
    return [mu, phi, p]

def convertParamset2(mu, phi, p):
    # (μ,φ,p) -> (λ,α,β)
    lambda_ = mu**(2-p) / (phi * (2-p))
    beta = (1/phi) * mu**(1-p) / (p-1)
    alpha = (2 - p) / (p - 1)
    return [lambda_, alpha, beta]

def poissonGammaPDF(x, mu,phi,p):
    # パラメタ(μ,φ,p)を設定し, x上での密度関数を返す
    logf = (1/phi) * (x * (mu**(1-p))/(1-p) - (mu**(2-p))/(2-p))
    f = np.exp(logf)
    return f

def poissonGammaRVS(lambda_,alpha,beta):
    # パラメタ(λ,α,β)を設定しPoisson-Gamma乱数を生成
    Po  = poisson.rvs(mu=lambda_) #Poisson乱数
    res = np.ones_like(lambda_) * Po
    res[Po>0] = gamma.rvs(alpha*Po[Po>0], 1/beta[Po>0]) #Gamma(N*alpha, beta)
    return res



# In[] 正解データ作る.
# feature data
Nentry  = 10000
num_feature = 5
x = np.random.normal(size=[Nentry, num_feature])
z = np.random.normal(size=[Nentry, num_feature])
# coefs
beta_true  = np.random.uniform(-0.5,0.5, size=num_feature)
gamma_true = np.random.uniform(-0.5,0.5, size=num_feature)
# customer i's parameters
mu  = np.exp(np.matmul(x, beta_true))
phi = np.exp(np.matmul(z, gamma_true))
p   = 1.3
phi.min(), phi.max()
mu.min(),  mu.max()

# generate y from Poisson-Gamma(λ,α,β)
lambda_, alpha, beta = convertParamset2(mu, phi, p)
y = poissonGammaRVS(lambda_,alpha,beta)



# In[] パラメタ推定を実行
model = DoubleGLM(intercept=False, isFreq=False)
model.setData(X=x,Z=z,Y=y)
model.updataBeta()





#
