import numpy as np
import matplotlib.pyplot as plt
import copy

class SGLD:
    """A simple SGLD class"""

    def __init__(self, N, stepsize):
        self.N = N
        self.stepsize = stepsize

    def update(self, g):
        return self.stepsize * g + np.sqrt(2 * self.stepsize / self.N) * np.random.randn(len(g))

class pSGLD:
    """A simple naive SBPS class"""
    def __init__(self,D,N,stepsize,beta1,lam):
        self.stepsize=stepsize
        self.N=N;
        self.D=D;
        self.lam=lam
        self.beta1=beta1
        self.grad2=np.zeros(D)
    def reset_preconditioner(self):
        self.grad2=np.zeros(self.D)
    def update(self,g):
        self.grad2=g*g*(1-self.beta1)+self.beta1*self.grad2
        preconditioner=1. /(np.sqrt(self.grad2)+self.lam)
        return self.stepsize*preconditioner*g+np.sqrt(self.stepsize*2/self.N*preconditioner)*np.random.randn(len(g))

class mSGNHT:
    def __init__(self,D,N,stepsize):
        self.D = D
        self.N = N
        self.stepsize = stepsize
        self.particle = np.zeros(D)
        self.thermostats = np.ones(D)
    def update(self,g):
        self.particle = (1-self.stepsize*self.thermostats)*self.particle + self.stepsize*g + \
                        np.sqrt(2*self.stepsize/self.N)*np.random.randn(self.D)
        self.thermostats += self.stepsize*(self.particle*self.particle - 1/self.N)
        return self.stepsize*self.particle

class ZZ_SS:
    """A simple Zig-Zag with Subsampling class"""
    def __init__(self,D,N,M):
        self.D=D
        self.N=N
        self.M=M
        self.expParam=1./(N*M)
        self.theta=np.sign(np.random.randn(D))
        self.times=np.zeros(D)
        self.flipbit=np.random.randint(0,D)
        self.flipcounter=0
        self.move_time=0;
        self.all_times = []
        self.all_vs = []
        self.total_time = 0

    def draw_times(self):
        self.times=np.random.exponential(self.expParam,self.D)
        self.move_time=np.min(self.times)
        tmp=np.where(self.times == self.move_time)
        self.flipbit=tmp[0][0]

    def flip_trial(self,val):

        r=np.random.rand(1)
        a= - (val*self.theta[self.flipbit])/self.M
        #print(str(val) + ', ' + str((-val/self.M)) + ', ' + str (r))
        if r<a:
            #print('flip')
            self.theta[self.flipbit]=-self.theta[self.flipbit]
            self.flipcounter+=1

    def reset_counter(self):
        self.flipcounter=0

    def update(self,g):
        self.flip_trial(g[self.flipbit])
        self.draw_times()
        self.all_times.append(copy.copy(self.move_time))
        self.total_time += self.move_time
        self.all_vs.append(copy.copy(self.theta))
        return self.theta*self.move_time

class lipschitzSBPS:
    """A simple naive SBPS class"""
    def __init__(self,D,N,stepsize,verbose):
        self.verbose=verbose
        self.L=stepsize;
        self.N=N;
        self.D=D;
        self.v = np.random.randn(self.D)
        self.normalize()
        self.bounce_counter=0
        self.total_time = 0
        self.all_vs = []
        self.all_times = []
    def reset_counter(self):
        self.bounce_counter=0
    def inner(self):
        self.lam=np.dot(self.v,self.g)
    def reflect(self):
        self.v=self.v-2*self.lam/np.dot(self.g,self.g)*self.g
    def normalize(self):
        self.v=self.v/np.sqrt(np.dot(self.v,self.v))
    def should_I_bounce(self):
        acceptanceProb=self.lam/self.L
        if np.random.uniform()<acceptanceProb:
            self.reflect()
            self.bounce_counter+=1
            if self.verbose:
                print('bounce!')
    def update(self,g):
        self.g=g
        self.inner()
        self.tau=np.random.exponential(1/(self.L*self.N))
        self.should_I_bounce()
        self.all_times.append(copy.copy(self.tau))
        self.total_time += self.tau
        self.all_vs.append(self.v)
        return -self.tau*self.v

class SBPS:
    def __init__(self,D,N,M,k,ref_const=.05,tau=0,fix_neg=False,zero_mean_prior=False,prior_var = 1e5,opt=False,A=1,gamma=0.5,max_rt=50):
        self.v = np.zeros(D)
        self.N = N
        self.M = M
        self.D = D
        self.up_factor = k
        self.ref_const = ref_const
        self.tau = tau
        self.fix_neg = fix_neg
        self.max_rt = max_rt
        self.prior_var = prior_var
        self.zero_mean_prior = zero_mean_prior
        # optimization variant
        self.opt = opt
        self.A = A
        self.gamma = gamma

        self.Gs =[]
        self.Ts =[]
        self.Stds = []
        self.errs = []
        self.bounce_times = []
        self.all_times = []
        self.all_vs = []
        self.all_Gs = []
        self.all_Stds = []
        self.weights = []
        self.all_S = []
        self.slopes = []

        self.p_errs =0
        self.maxG =0
        self.this_time = 0
        self.total_time = 0
        self.num_bounce = 0
        self.count_ref = 0
        self.count_max = 0

        self.upper = 0
        self.S = 10
        self.neg_S = 0
        self.dt = .01
        self.prior_mean = 0
        self.prior_vars = [self.prior_var]
        self.Sigma = 0
        self.should_refresh = False
        self.num_iter = 1

    def clean(self):
        self.Gs =[-self.Gs[-1]]
        self.Stds =[self.Stds[-1]]
        self.Ts =[0]
        self.weights = [1/float(self.Stds[-1]**2)]
        self.this_time = 0
        if len(self.all_times) > 0 and self.tau != 0:
            self.tau = np.mean(self.bounce_times)
        self.update_slope()

    def reweight(self):
        if self.tau != 0:
            decays = [np.exp((self.this_time - i)/float(self.tau)) for i in self.Ts]
            self.Stds = [a*b for a,b in zip(self.Stds,decays)]

    def G(self,t):
        return self.S*t+self.intercept

    def pred_g(self, t):
        return self.G(t), self.G(t) + self.up_factor*np.sqrt(np.dot(np.dot([1,t],self.Sigma),[1,t]) + self.Stds[-1]**2)

    def sample_next(self):
        u = -np.log(np.random.rand())
        predgs = []
        uppers = []
        rt = 0
        pg = 0
        prev_pg, prev_upper = self.pred_g(self.Ts[-1])
        predgs.append(prev_pg)
        uppers.append(prev_upper)
        while u > 0:
            # break if trajectory is too long
            if rt > self.max_rt:
                self.count_max += 1
                ri = prev_upper
                break

            # predict next value
            rt += self.dt
            pg, upper = self.pred_g(self.Ts[-1] + rt)
            predgs.append(pg)
            uppers.append(upper)

            if upper > 0:
                # bound is positive - reduce u by the appropriate amount
                this_area = prev_upper*self.dt + self.dt*(upper-prev_upper)/2
                if this_area < u:
                    u -= this_area
                else:
                    # calculating exact time to return, moving less than dt
                    rt -= self.dt

                    a = (upper-prev_upper)/(2*self.dt)
                    b = prev_upper
                    c = -u
                    try:
                        lt = (-b + np.sqrt(b**2 -4*a*c))/(2*a)
                    except FloatingPointError:
                        print('FloatingPointError, a=', a)
                    rt += lt

                    ri = prev_upper + lt*(upper-prev_upper)/self.dt
                    u = 0
            prev_pg = pg
            prev_upper = upper
        return rt, ri

    def refresh_v(self):
        self.v = np.random.rand(self.v.size).reshape(self.v.shape)-.5
        self.v = self.v/np.linalg.norm(self.v)
        self.v = self.v
        self.count_ref += 1
        G = -self.N*np.dot(self.g,self.v)
        if self.opt: # we are optimizing, scale with temperature
            G = self.anneal(G)
        self.Gs =[G]
        self.Stds =[self.Stds[-1]]
        self.Ts =[0]
        self.weights = [1/float(self.Stds[-1]**2)]
        self.this_time = 0
        self.update_slope()
        self.should_refresh = False

    def bounce(self):
        self.v=self.v-2*np.dot(self.v,self.g)/np.dot(self.g,self.g)*self.g

        # storing various quantities
        self.bounce_times.append(self.Ts[-1])
        self.num_bounce += 1
        self.slopes.append(self.S)
        self.prior_vars.append(self.Sigma[1,1])

    def update_slope(self):
        vG = np.asarray(self.Gs)
        vT = np.asarray(self.Ts)
        vW = np.asarray(self.weights)

        # update covariance matrix
        Sigma0 = np.zeros((2,2))
        for i in range(len(self.Gs)):
            Sigma0 += self.weights[i]*np.outer([1,self.Ts[i]],[1,self.Ts[i]])
        Sigma0[1,1] += 1/float(self.prior_var)
        self.Sigma = np.linalg.inv(Sigma0)

        [I_new,S_new] = np.dot(self.Sigma,[np.sum(vG*vW),np.sum(vG*vT*vW)+self.prior_mean/float(self.prior_var)])
        self.S = S_new
        self.intercept  = I_new

        # if slope is negative and fix_neg is true fix the slope
        if self.S <= 0:
            self.neg_S += 1
            if self.fix_neg is True:
                if len(self.all_S) > 0:
                    self.S = max(0,np.mean(self.all_S))
                    self.intercept = np.mean(vG) - np.mean(vT)*self.S
                else:
                    self.S = 0
                    self.intercept = np.mean(vG)

    def accept_or_reject(self,G,upper):
        self.update_slope()
        if  np.random.rand()  < G/upper: # accepted proposal, bouncing
            if G/upper > 1:
                self.p_errs += 1
                self.errs.append(G/upper)
            self.bounce()
            self.clean()

    def init(self):
        # if this is the start of the run initializing the velocity and slope
        try:
            self.t
        except AttributeError:
            self.t = 0
            self.S = 1
            self.v = -self.g/float(np.linalg.norm(self.g))
            self.intercept = -self.N*np.dot(self.g,self.v)
            self.upper = 10

    def anneal(self,G):
        beta = self.A * self.num_iter ** self.gamma
        #beta = self.A * (100*self.num_bounce+1) ** self.gamma
        return G * beta

    def update(self,g,var):
        # g should be the mean over datapoints in the minibatch (not the sum)
        self.g=np.float64(g)
        self.var = np.float64(var)
        assert not np.isnan(self.var)
        self.init()
        G = -self.N*np.dot(self.g,self.v)
        std = self.N*np.sqrt((1-self.M/float(self.N))* self.var/self.M)
        if self.opt: # we are optimizing, scale with temperature
            G = self.anneal(G)
            std = self.anneal(std)

        #store
        self.all_Gs.append(G)
        self.all_Stds.append(std)
        self.this_time += self.t
        self.Gs.append(G)
        self.Ts.append(copy.copy(self.this_time))
        self.reweight()
        self.Stds.append(std)
        self.weights.append(1/float(std**2))

        ## Refresh based on independent process ##
        if self.should_refresh:
            self.refresh_v()
            self.clean()
        else:
            # no refreshment, do a standard accept/reject
            self.accept_or_reject(G,self.upper)

        bounce_t, self.upper = self.sample_next()
        ref_t = -np.log(np.random.rand())/self.ref_const
        self.t = min(bounce_t,ref_t)
        if ref_t < bounce_t:
            # we are refreshing instead of bouncing, make a note of this and update v accordingly
            self.should_refresh = True

        self.all_times.append(copy.copy(self.t))
        self.all_S.append(self.S)

        ## If zero_mean_prior is false, update the prior mean ##
        if not self.zero_mean_prior:
            self.prior_mean = np.mean(self.all_S)

        self.total_time += self.t
        self.all_vs.append(self.v)
        self.num_iter += 1
        return (self.v)*self.t

    def print_summary(self):
        num_iters = len(self.all_Gs)
        reject = (num_iters - self.num_bounce)/float(num_iters)
        num_pacc = (self.p_errs/float(self.num_bounce))
        print('Number of bounces: ' + str(self.num_bounce))
        print('percent p(acc)>1: ', num_pacc)
        print( 'percent rejections: ', reject)
        print('percent negative slope: ', self.neg_S/float(len(self.all_Gs)))
        print('total travel time: ', self.total_time )
        print('max trajectories: ', self.count_max)
        print('refreshments:', self.count_ref)
