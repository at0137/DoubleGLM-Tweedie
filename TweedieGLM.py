import numpy as np
from numpy.random import normal
from scipy.special import gamma as gammaFunc

""" TEST IMPLMENTATION of Tweedie without Frequency
    * サンプル数をnとして n x n 行列の計算がどんくらいかかるのか心配. nデカかったらやばい
    * 0割が起きないか, 逆行列計算がきちんと達成されるか懸念
    * その他バグチェックをどう行うかが課題.
        - Rのglmと比較して大きな違いがなければOK?
        - 論文中のデータセットが入手できて、結果を再現できると一番いい.
"""

class DoubleGLM:
    EPSILON = 1e-6
    Converge = 1e-3

    def __init__(self, intercept=False, isFreq=False):

        self.isFreq = isFreq # whether or not Frequency is available
        self.intercept = intercept # whether or not intercept is considered

        self.X = np.array([]) # n x dimBeta feature matrix for mu
        self.Z = np.array([]) # n x dimGamma feature matrix for phi
        self.Y = np.array([]) # n x 1 target for mu(claim cost)
        self.d = np.array([]) # n x 1 target for phi(unit deviance)
        self.w = np.array([]) # n x 1 sample weight(policy years)
        # Note: X.T in Ref[1] correspond to X in this script

        self.beta = np.array([])  # regression coefs for mu
        self.gamma = np.array([]) # regression coefs for phi
        self.mu = np.array([])  # current prediction of mu
        self.phi = np.array([]) # current prediction of phi
        self.p = 1.5 # scale parameter of mean-var relation
        self.n = 0 # of samples
        self.dimBeta  = 0 # of features for mu
        self.dimGamma = 0 # of features for phi


    def setData(self, X, Z, Y, w=None):
        """ set Feature,Target,SampleWeight. Input shape must be (Nsample, Ncols>=1). """

        self.n = X.shape[0] # Nsample

        if self.intercept:
            # set Feature
            self.X = np.concatenate((np.ones([self.n, 1]), X), axis=1)
            self.Z = np.concatenate((np.ones([self.n, 1]), Z), axis=1)
            # set Constants
            self.dimBeta = X.shape[1] + 1
            self.dimGamma = Z.shape[1] + 1
        else:
            # set Feature
            self.X = X
            self.Z = Z
            # set Constants
            self.dimBeta = X.shape[1]
            self.dimGamma = Z.shape[1]

        # set target (n x 1)
        self.Y = Y.reshape(Y.shape[0], 1)
        self.d = np.ones_like(Y)
        self.mu = np.copy(self.Y)       # current prediction of mu
        self.phi = np.ones_like(self.Y) # current prediction of phi

        # initialize coefs (by normal dist.)
        self.beta = normal(size=[1, self.dimBeta])
        self.gamma = normal(size=[1, self.dimGamma])

        # set sample weights (n x 1)
        if w is None:
            self.w = np.ones_like(self.Y)
        else:
            self.w = w.reshape(w.shape[0], 1)

        return None


    # --- [main logic] ---
    def updataBeta(self):
        """ Fisher scoreing Iteration of beta. """

        priorBeta = np.copy(self.beta) # 返り値で更新幅を与えるので初期値を保持しておく
        W = self.__genW() # diag Matrix
        # update beta : Fisher Scoring Update
        result = np.matmul(np.matmul(self.X, W), self.X.T)
        result = np.matmul(np.inv(result), self.X)
        result = np.matmul(result, W)
        # claimFreq=0の人は, firstIterationでmu=0の0割が必ず発生する. 適切な対処法は+epsilonで良い?
        z = (self.Y - self.mu)/(self.mu + DoubleGLM.EPSILON) + np.log(self.mu + DoubleGLM.EPSION)
        self.beta = np.matmul(result, z)

        # update current mu
        self.mu = np.exp(np.matmul(self.X, self.beta))
        # update current deviance
        d1 = self.Y * (self.Y**(1-p) - self.mu**(1-p)) / (1-self.p)
        d2 = (self.Y**(2-p) - self.mu**(2-p)) / (2-self.p)
        self.d = 2*self.w * (d1 - d2)

        return np.abs(priorBeta - self.beta)


    def updateGamma(self):
        """ Fisher scoring update of gamma """

        priorGamma = np.copy(self.gamma)
        Wd, zd = self.__gen_Wd_zd() # diagMat and working vec
        # update gamma : Adjusted Fisher Scoring Iteration
        result = np.matmul( np.matmul(self.Z, Wd), self.Z.T)
        result = np.linalg.inv(result)
        result = np.matmul( np.matmul(result, self.Z), Wd)
        self.gamma = np.matmul(result, zd)

        # update current phi
        self.phi = np.exp(np.matmul(self.Z, self.gamma))

        return np.abs(priorGamma - self.gamma)


    def updateP(self, M=10):
        """ determine p through maximizing penalized logL(by line search on p).
            M represents assumed sup[claimFreq] """

        possible_P = np.arange(1.01, 2, 0.01)
        logL = np.zeros_like(possible_P)
        # calculate logLikelihood at each p
        for m, p in enumerate(possible_P):
            lambda_, tau, alpha = self.__convertParamset(p)
            # logL_temp is array consists of each customer's logLikelihood
            logL_temp = -(self.Y/tau) - lambda_ - np.log(self.Y)
            logL_temp += np.log(self.__generalizedBessel(self.y,M,lambda_,tau,alpha))
            logL_temp[self.Y==0] = -lambda_[self.Y==0] # when there's no claim, logL = -lambda
            logL[m] = logL_temp.sum() # summuation for sample i

        # update p
        self.p = possoble_P[np.argmax(logL)]

        return None


    def fit(self, M=10, N=100):
        """ Iteratively updateBeta and updateGamma until convergence, then determine p.
            M represents assumed sup[claimFreq].
            N represents max iteration times of Fisher updation.  """

        # determine Beta and Gamma
        diffBeta = diffGamma = 1e5
        for itr in range(1,N+1):
            print("Fisher Updation : {}".format(itr))
            converge_Beta = diffBeta < DoubleGLM.Converge
            converge_Gamma = diffGamma < DoubleGLM.Converge
            if not converge_Beta:
                diffBeta = self.updateBeta().max()
            if not converge_Gamma:
                diffGamma = self.updateGamma().max()
            if converge_Beta and converge_Gamma:
                print("Fisher Scoring Updation completed @ iteration {}".format(itr))
                break

        # determine p
        self.updateP(M)

        return None


    # --- [Utils] ---
    def __genW(self):
        # for updateBeta iteration, generate matrix "W"
        W_elements = self.mu**2 * self.w / (self.phi * self.phi * self.mu**self.p + DoubleGLM.EPSILON)
        W_elements = W_elements.reshape(W_elements.shape[0], )
        W = np.diag(W_elements)
        return W

    def __genHatMatrix(self):
        # for gen_Wd_zd, generate "hat matrix"
        W = self.__genW()
        hatMatrix = np.matmul(np.matmul(self.X, W), self.X.T)
        hatMatrix = np.matmul(self.X.T, np.linalg.inv(hatMatrix))
        hatMatrix = np.matmul(np.sqrt(W), hatMatrix)
        hatMatrix = np.matmul(hatMatrix, self.X)
        hatMatrix = np.matmul(hatMatrix, np.sqrt(W))
        return hatMatrix

    def __gen_Wd_zd(self):
        # for updateGamma iteration, generate "Wd" and "zd"
        hatMarix = self.__genHatMatrix()
        h = np.diag(hatMatrix)
        # generate Wd
        Wd_elements = self.phi**2 * (1-h) / (2*self.phi**2) #round項とVd項が打ち消し合ってる.そんな簡単な形になる?
        Wd = np.diag(Wd_elements)
        # generate zd
        zd = (1/self.phi) * (self.d/(1-h) - self.phi) + np.log(self.phi)
        zd = zd.reshape(self.n, 1)
        return (Wd, zd)

    def __convertParamset(self, p):
        # (mu,phi,p) -> (lambda,tau,alpha)
        lambda_ = self.mu**(2-p) / (self.phi * (2-p))
        tau = self.phi * (p - 1) * self.mu**(p-1)
        alpha = (2 - p) / (p - 1)
        return (lambda_, tau, alpha)

    def __generalizedBessel(self, y, M, lambda_, tau, alpha):
        # M represents sup[index j]. y,lambda,tau must be 1d-array.
        W = np.array([ ( lambda_**j * (y/tau)**(j*alpha) )/( gammaFunc(j+1)*gammaFunc(j*alpha) )
                        for j in range(1,M+1) ])
        W = W.sum(axis=0) # summuation for j=1:M and array shape transform into (nsample, 1)
        return W
