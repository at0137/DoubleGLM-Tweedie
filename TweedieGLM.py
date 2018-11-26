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
    EPSILON = 1e-7
    Converge = 1e-3

    def __init__(self, intercept=False):

        self.intercept = intercept # whether or not intercept is considered

        self.X = np.array([]) # n x dimBeta feature matrix for mu
        self.Z = np.array([]) # n x dimGamma feature matrix for phi
        self.Y = np.array([]) # n x 1 target for mu(claim cost)
        self.d = np.array([]) # n x 1 target for phi(unit deviance)
        self.w = np.array([]) # n x 1 sample weight(policy years)

        self.beta = np.array([])  # regression coefs for mu
        self.gamma = np.array([]) # regression coefs for phi
        self.mu = np.array([])  # current prediction of mu
        self.logPhi = np.array([]) # current prediction of logPhi
        self.p = 1.5 # scale parameter of mean-var relation
        self.n = 0 # of samples
        self.dimBeta  = 0 # of features for mu
        self.dimGamma = 0 # of features for logPhi


    def setData(self, X, Z, Y, w=None):
        """ set Feature,Target,SampleWeight. Input shape must be (Nsample, Ncols>=1). """

        np.random.seed(1)
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
        self.Y = Y.reshape(Y.shape[0], 1) + DoubleGLM.EPSILON #target( +e) => to avoid nan
        self.d = np.ones(self.Y.shape)
        self.mu  = np.copy(self.Y)       # current prediction of mu
        self.logPhi = np.zeros(self.Y.shape) # current prediction of loglogPhi

        # initialize coefs (by normal dist.)
        self.beta = normal(size=[self.dimBeta, 1])
        self.gamma = normal(size=[self.dimGamma, 1])

        # set sample weights (n x 1)
        if w is None:
            self.w = np.ones(self.Y.shape)
        else:
            self.w = w.reshape(w.shape[0], 1)

        return None


    # --- [main logic] ---
    def updateBeta(self):
        """ Fisher scoreing Iteration of beta. """

        priorBeta = np.copy(self.beta) # 返り値で更新幅を与えるので初期値を保持しておく
        W = self.__genW() # diag Matrix
        # update beta : Fisher Scoring Update
        result = np.matmul(np.matmul(self.X.T, W), self.X)
        result = np.matmul(np.linalg.inv(result), self.X.T)
        result = np.matmul(result, W)
        # claimFreq=0の人は, firstIterationでmu=0の0割が必ず発生する. 適切な対処法は+epsilonで良い?
        z = (self.Y - self.mu)/(self.mu + DoubleGLM.EPSILON) + np.log(self.mu + DoubleGLM.EPSILON)
        self.beta = np.matmul(result, z)

        # update current mu
        self.mu = np.exp(np.matmul(self.X, self.beta))
        # update current deviance
        d1 = self.Y * (self.Y**(1-self.p) - self.mu**(1-self.p)) / (1-self.p)
        d2 = (self.Y**(2-self.p) - self.mu**(2-self.p)) / (2-self.p)
        self.d = 2*self.w * (d1 - d2)

        return np.abs(priorBeta - self.beta)


    def updateGamma(self):
        """ Fisher scoring update of gamma """

        priorGamma = np.copy(self.gamma)
        Wd, zd = self.__gen_Wd_zd() # diagMat and working vec
        # update gamma : Adjusted Fisher Scoring Iteration
        result = np.matmul(np.matmul(self.Z.T, Wd), self.Z)
        result = np.linalg.inv(result)
        result = np.matmul( np.matmul(result, self.Z.T), Wd )
        self.gamma = np.matmul(result, zd)

        # update current logPhi
        self.logPhi = np.matmul(self.Z, self.gamma)

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
                print("beta updated...")

            if not converge_Gamma:
                diffGamma = self.updateGamma().max()
                print("gamma updated...")
            print("current update margin: beta=", diffBeta, " gamma=" ,diffGamma)

            if converge_Beta and converge_Gamma:
                print("Fisher Scoring Updation completed @ iteration {}".format(itr))
                break

        # determine p
        print("Then, adjust scale parameter p")
        self.updateP(M)

        return None


    # --- [Utils] ---
    def __genW(self):
        # generate matrix "W" for updateBeta iteration
        W_elements = self.mu**2 * self.w / (np.exp(2*self.logPhi) * self.mu**self.p + DoubleGLM.EPSILON)
        W_elements = W_elements.reshape(W_elements.shape[0], )
        W = np.diag(W_elements)
        return W

    def __genHatMatrix(self):
        # generate "hat matrix" for gen_Wd_zd
        W = self.__genW()
        hatMatrix = np.matmul(np.matmul(self.X.T, W), self.X)
        hatMatrix = np.matmul(self.X, np.linalg.inv(hatMatrix))
        hatMatrix = np.matmul(np.sqrt(W), hatMatrix)
        hatMatrix = np.matmul(hatMatrix, self.X.T)
        hatMatrix = np.matmul(hatMatrix, np.sqrt(W))
        return hatMatrix

    def __gen_Wd_zd(self):
        # generate "Wd" and "zd" for Gamma update iteration
        Wd = np.diag(0.5 * np.ones(self.n)) #round項とVd項が打ち消し合ってるんだが?
        zd = (self.d/np.exp(self.logPhi) - 1) + self.logPhi
        return [Wd, zd]

    def __gen_Wd_zd_star(self):
        # generate "Wd*" and "zd*" for updateGamma iteration
        hatMatrix = self.__genHatMatrix()
        h = np.diag(hatMatrix).reshape(self.n, 1)
        # generate Wd
        Wd_elements = (1-h)/2 # #round項とVd項が打ち消し合ってるんだが?
        Wd_elements = Wd_elements.reshape(self.n) #to conv diag, shape[1] must be blank
        Wd = np.diag(Wd_elements)
        # generate zd
        zd = (self.d/((1-h) * np.exp(self.logPhi) + DoubleGLM.EPSILON) - 1) + self.logPhi
        zd = zd.reshape(self.n, 1)
        return [Wd, zd]

    # @classmethod ... pending
    def __convertParamset(self, p):
        # (μ,φ,p) -> (λ,α,β)
        lambda_ = self.mu**(2-p) / (np.exp(self.logPhi) * (2-p))
        tau = np.exp(self.logPhi) * (p - 1) * self.mu**(p-1)
        alpha = (2 - p) / (p - 1)
        return (lambda_, tau, alpha)

    def __generalizedBessel(self, y, M, lambda_, tau, alpha):
        # M represents sup[index freq]. y,lambda,tau must be 1d-array.
        W = np.array([ ( lambda_**j * (y/tau)**(j*alpha) )/( gammaFunc(j+1)*gammaFunc(j*alpha) )
                        for j in range(1,M+1) ])
        W = W.sum(axis=0) # summuation for j=1:M and array shape transform into (nsample, 1)
        return W
