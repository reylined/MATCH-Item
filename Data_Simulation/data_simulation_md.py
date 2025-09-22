import numpy as np
import pandas as pd
import scipy.integrate as integrate
import scipy.optimize as optimize


def simJM_item_MD(I, num_items, obstime, a_param=None, b_param=None, seed=None):
    
    if seed is not None:
        np.random.seed(seed)
    
    J = len(obstime)
    K = sum(num_items)
    
    ## Latent variables
    # theta_i^(1)(t) = beta1^(1)*time + beta2^(1)*time*X + ranef_i1 + ranef_i2*time
    # theta_i^(2)(t) = beta1^(2)*time + beta2^(2)*time*X + ranef_i3 + ranef_i4*time
    
    
    beta11 = 0.17
    beta12 = 0.06
    beta21 = 0.18
    beta22 = 0.05
    
    cov = np.diag([1,1,1,1]).astype(float)
    cov[1,0] = cov[0,1] = -0.151
    cov[2,0] = cov[0,2] = -0.262
    cov[2,1] = cov[1,2] =  0.024
    cov[3,0] = cov[0,3] =  0.074
    cov[3,1] = cov[1,3] =  0.396
    cov[3,2] = cov[2,3] = -0.161
    
    ranef = np.random.multivariate_normal(mean=[0,0,0,0], cov=cov, size=I)
    
    X = np.random.binomial(n=1, p=0.5, size=I)
    
    theta1 = np.add.outer(ranef[:,0],beta11*obstime) + beta12*np.outer(X,obstime) + np.outer(ranef[:,1],obstime)
    theta1 = theta1[:,:,np.newaxis,np.newaxis]
    theta2 = np.add.outer(ranef[:,2],beta21*obstime) + beta22*np.outer(X,obstime) + np.outer(ranef[:,3],obstime)
    theta2 = theta2[:,:,np.newaxis,np.newaxis]
    
    
    b_param = np.random.uniform(low=0, high=4, size=K)
    a_param = np.random.uniform(low=0, high=4, size=(K,3))
    a_param = np.sort(a_param, axis=1)
    
    ## Item level
    a1 = a_param[:num_items[0],:]
    b1 = b_param[:num_items[0]]
    a1 = a1[np.newaxis,np.newaxis,:,:]
    b1 = b1[np.newaxis,np.newaxis,:,np.newaxis]
    a2 = a_param[num_items[0]:,:]
    b2 = b_param[num_items[0]:]
    a2 = a2[np.newaxis,np.newaxis,:,:]
    b2 = b2[np.newaxis,np.newaxis,:,np.newaxis]
    
    p1 = 1/(1+np.exp(-a1+b1*theta1))
    p1 = np.nan_to_num(p1, nan=1)
    p1 = np.insert(p1, obj=0, values=0, axis=3)
    p1 = np.insert(p1, obj=p1.shape[3], values=1, axis=3)
    p2 = 1/(1+np.exp(-a2+b2*theta2))
    p2 = np.nan_to_num(p2, nan=1)
    p2 = np.insert(p2, obj=0, values=0, axis=3)
    p2 = np.insert(p2, obj=p2.shape[3], values=1, axis=3)
    
    u = np.random.uniform(low=0, high=1, size=(I,J,K))
    u = u[:,:,:,np.newaxis]
    
    items1 = np.argmax(p1-u[:,:,:num_items[0],:]>0, axis=3) - 1
    items2 = np.argmax(p2-u[:,:,num_items[0]:,:]>0, axis=3) - 1
    
    items1_df = pd.DataFrame(np.reshape(items1, newshape=(-1,num_items[0])))
    items2_df = pd.DataFrame(np.reshape(items2, newshape=(-1,num_items[1])))
    items_df = pd.concat([items1_df, items2_df], axis=1)
    items_df.columns = ["item"+str(c) for c in range(K)]
    items_df['subscore1'] = items_df.iloc[:,:num_items[0]].sum(axis=1)
    items_df['subscore2'] = items_df.iloc[:,num_items[0]:K].sum(axis=1)
    items_df['sumscore'] = items_df.iloc[:,:K].sum(axis=1)
    

    # Simulate Survival Times using Inverse Sampling Transform
    scale = np.exp(-7)
    U = np.random.uniform(size=I)
    alpha1 = 1.5
    alpha2 = -1
    
    def CHF(tau):
        def h(t):
            theta1_i = beta11*t + beta12*t*X[i] + ranef[i,0] + ranef[i,1]*t
            theta2_i = beta21*t + beta22*t*X[i] + ranef[i,2] + ranef[i,3]*t
            return scale * np.exp(alpha1*theta1_i + alpha2*theta2_i)
        return np.exp(-1 * integrate.quad(lambda xi: h(xi),0,tau)[0])
        
    Ti = np.empty(I)
    Ti[:] = np.NaN
    for i in range(0,I):
        try:
            Ti[i] = optimize.brentq(lambda xi: U[i]-CHF(xi), 0, 100)
        except ValueError:
            Ti[i] = 100
    
    # Get true survival probabilities
    true_prob = np.ones((I, len(obstime)))
    for i in range(0,I):
        for j in range(1,len(obstime)):
            tau = obstime[j]
            true_prob[i,j] = CHF(tau)
            
    C = np.random.uniform(low=obstime[1], high=obstime[-1]+25, size=I)
    C = np.minimum(C, obstime[-1])
    event = Ti<C
    true_time = np.minimum(Ti, C)
    
    # Round true_time up to nearest obstime
    time = [np.min([obs for obs in obstime if obs-t>=0]) for t in true_time]
    
    subj_obstime = np.tile(obstime, reps=I)
    true_prob = true_prob.flatten()
    ID = np.repeat(range(0,I), repeats=J)
    visit = np.tile(range(0,J), reps=I)
    data = pd.DataFrame({"id":ID, "visit":visit, "obstime":subj_obstime,
                        "time":np.repeat(time,repeats=J),
                        "event":np.repeat(event,repeats=J),
                        "true":true_prob})
    data = data.join(items_df)
    
    return data