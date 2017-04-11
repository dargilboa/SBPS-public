import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import time
from skimage.measure import block_reduce
from samplers import SBPS
import copy

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert inputs.shape[0] == targets.shape[0]
    if shuffle:
        indices = np.arange(inputs.shape[0])
        np.random.shuffle(indices)
    for start_idx in range(0, inputs.shape[0] - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]

def logit(x):
    return 1./(1.+np.exp(x))

def NLLexplicit(prob,y):
    return (np.dot(y,np.log(prob))+np.dot(1-y,np.log(1-prob)))/len(y)

def ACF(data,burnin):
    corrs = []
    for i in range(len(data)):
        #ACF
        sig = data[i][burnin:,0] - np.mean(data[i][burnin:,0])
        corr = np.correlate(sig,sig,'full')
        corr = corr/np.max(corr)
        corr = corr[corr.shape[0]/2:]
        corrs.append(corr)
    return corrs


# function to crop and save figures, requires epstopdf and pdfcrop
def savefig(name,dest=None):
    fname = name + '.eps'
    plt.savefig(fname, format='eps', dpi=1000, bbox_inches='tight')
    os.system('epstopdf ' + fname)
    os.system('pdfcrop ' + name + '.pdf')
    if dest is not None:
        os.system('mv ' + name + '-crop.pdf ' + dest + '/' + name + '.pdf')
        os.system('rm ' + name + '.pdf')
        print ('saved ' + dest + '/' + name + '.pdf')
    else:
        os.system('mv ' + name + '-crop.pdf '+ name + '.pdf -f')
        print ('saved ' + name + '.pdf')
    os.system('rm ' + fname) #remove eps

# takes a list of tf tensors and returns a single vector with all tensors flattened
def flatten_tensor_list(tensor_list):
    shapes = [w.get_shape() for w in tensor_list]
    sizes = [np.prod(ws.as_list()) for ws in shapes]
    flat_list = tf.reshape(tensor_list[0],[sizes[0]])
    for i in range(len(tensor_list)-1):
        flat_list = tf.concat(0,[flat_list,tf.reshape(tensor_list[i+1],[sizes[i+1]])])
    return flat_list

# downsample dataset
def downsample_data(data,b_size):
    return block_reduce(data.reshape(-1,28,28),block_size=(1,b_size, b_size)).reshape(-1,int((28/b_size)**2))

class clock:
    def __init__(self):
        import time
        self.t = 0
    def tick(self):
        self.t = time.time()
    def tock(self):
        num_secs = np.float32(time.time() - self.t)
        print('Time elapsed - ', num_secs, ' secs (', np.float32(num_secs/60), ' mins)')

def run_sampler(sampler,get_weights,set_weights,test_error,total_iter,n_epochs,NLL_factor,\
                data,labels,batch_size,grad_calc,train_step,grad_var_calc=None,use_preconditioner=False,W_init=None):
    """
    Runs sampler, returns test

    Parameters:
    sampler - instance of sampler class that contains an update module that given a raw gradient returns a modified
    gradient that is then used to advance the position of a walker

    """
    # initialize tf for optimization
    #weights = flatten_tensor_list(all_weights)
    D = get_weights().shape[0]
    #D = int(np.sum([np.prod(s) for s in w_shapes]))
    lam=1e-4
    beta1=.99
    min_var = 1e-15
    small_var_encountered = 0
    grad2=np.zeros(D)
    preconditioner = 1

    # initialize variables to store history
    test_err = []
    samples = []
    # assign initial location of sampler if supplied
    if W_init is not None:
        set_weights(W_init)

    # take initial sample and test error
    samples.append(get_weights())
    test_err.append(test_error())
    print('Evaluating Test error / NLL - ', test_err[-1])

    # main optimization loop
    cl = clock()
    cl.tick()
    for n in range(n_epochs):
        for minibatch in iterate_minibatches(data, labels, batch_size, shuffle=True):
            # grab new minibatch
            x_batch, y_batch = minibatch

            # calculate gradient
            if sampler.__class__ == SBPS:
                velocity_input=sampler.v
                # get gradient
                gradient=grad_calc(x_batch, y_batch)

                # build preconditioner
                if use_preconditioner:
                    grad2=gradient*gradient*(1-beta1)+beta1*grad2
                    preconditioner=1. /(np.sqrt(grad2)+lam)
                    preconditioner = preconditioner / float(np.mean(preconditioner))

                gradient_times_velocity_variance=grad_var_calc(x_batch, y_batch, preconditioner*velocity_input)

                # make sure variance isn't too small (if minibatch size is too small)
                if gradient_times_velocity_variance < min_var:
                    gradient_times_velocity_variance = min_var
                    small_var_encountered += 1

                # apply some operation to the gradient
                new_gradient=sampler.update(preconditioner*gradient,gradient_times_velocity_variance)

            else:
                gradient=grad_calc(x_batch, y_batch)
                new_gradient=sampler.update(gradient)

            # apply gradient and track quantities
            train_step(preconditioner*new_gradient)
            if sampler.__class__ == SBPS:
                sampler.all_vs[-1] = preconditioner*sampler.all_vs[-1]

            # store samples
            samples.append(get_weights())

            if np.mod(len(samples),NLL_factor) == 0:
                test_err.append(test_error())
                print('Evaluating Test error / NLL - ', test_err[-1])
    cl.tock()
    if small_var_encountered > 0:
        print('Warning - minibatch variance smaller than ',min_var, ' calculated ', small_var_encountered, \
              ' times. Consider increasing minibatch size.')
    return test_err,samples

def generate_SBPS_samples(sampler,get_weights,set_weights,test_error,train_step,NLL_factor,W_init=None):
    # initialize tf for optimization
    print( 'Generating discrete samples from continuous SBPS trajectory')
    # assign initial location of sampler if supplied
    if W_init is not None:
        set_weights(W_init)
    #    assign_weights(W_init,all_weights,w_shapes)
    #weights = flatten_tensor_list(all_weights)

    # set up loop, take initial sample
    total_times = len(sampler.all_times)
    time_step = sampler.total_time/total_times
    j = 0
    test_err = []
    samples = []
    total_time_moved = 0
    all_curr_ts = []
    i = 0
    time_until_next_sample = copy.copy(time_step)
    samples.append(get_weights())
    i += 1
    test_err.append(test_error())
    print('Evaluating Test error - ', test_err[-1])

    while i < total_times:
        curr_time = float(sampler.all_times[j])
        while curr_time > time_until_next_sample:
            # take another sample along current linear trajectory
            curr_time -= time_until_next_sample
            g = time_until_next_sample*sampler.all_vs[j]
            train_step(g)
            if i < total_times:
                samples.append(get_weights())
            if np.mod(i,NLL_factor) == 0:
                test_err.append(test_error())
                print('Evaluating Test error - ', test_err[-1])
            i += 1
            time_until_next_sample = copy.copy(time_step)

        # moving remainder of current linear trajectory
        time_until_next_sample -= curr_time
        g = curr_time*sampler.all_vs[j]
        train_step(g)
        j += 1

    # take final sample
    samples.append(get_weights())

    return test_err,np.asarray(samples)
