import numpy
import theano
import theano.tensor as T
from theano.tensor.nnet import conv
from cis.deep.utils.theano import debug_print
from WPDefined import repeat_whole_matrix, repeat_whole_tensor


def create_HiddenLayer_para(rng, n_in, n_out):

    W_values = numpy.asarray(rng.uniform(
            low=-numpy.sqrt(6. / (n_in + n_out)),
            high=numpy.sqrt(6. / (n_in + n_out)),
            size=(n_in, n_out)), dtype=theano.config.floatX)  # @UndefinedVariable
    W = theano.shared(value=W_values, name='W', borrow=True)

    b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)  # @UndefinedVariable
    b = theano.shared(value=b_values, name='b', borrow=True)
    return W,b

def create_Bi_GRU_para(rng, word_dim, hidden_dim):
        # Initialize the network parameters
        U = numpy.random.uniform(-numpy.sqrt(1./hidden_dim), numpy.sqrt(1./hidden_dim), (3, hidden_dim, word_dim))
        W = numpy.random.uniform(-numpy.sqrt(1./hidden_dim), numpy.sqrt(1./hidden_dim), (3, hidden_dim, hidden_dim))
        b = numpy.zeros((3, hidden_dim))
        # Theano: Created shared variables
        U = debug_print(theano.shared(name='U', value=U.astype(theano.config.floatX), borrow=True), 'U')
        W = debug_print(theano.shared(name='W', value=W.astype(theano.config.floatX), borrow=True), 'W')
        b = debug_print(theano.shared(name='b', value=b.astype(theano.config.floatX), borrow=True), 'b')

        Ub = numpy.random.uniform(-numpy.sqrt(1./hidden_dim), numpy.sqrt(1./hidden_dim), (3, hidden_dim, word_dim))
        Wb = numpy.random.uniform(-numpy.sqrt(1./hidden_dim), numpy.sqrt(1./hidden_dim), (3, hidden_dim, hidden_dim))
        bb = numpy.zeros((3, hidden_dim))
        # Theano: Created shared variables
        Ub = debug_print(theano.shared(name='Ub', value=Ub.astype(theano.config.floatX), borrow=True), 'Ub')
        Wb = debug_print(theano.shared(name='Wb', value=Wb.astype(theano.config.floatX), borrow=True), 'Wb')
        bb = debug_print(theano.shared(name='bb', value=bb.astype(theano.config.floatX), borrow=True), 'bb')
        return U, W, b, Ub, Wb, bb
    
def create_GRU_para(rng, word_dim, hidden_dim):
        # Initialize the network parameters
        U = numpy.random.uniform(-numpy.sqrt(1./hidden_dim), numpy.sqrt(1./hidden_dim), (3, hidden_dim, word_dim))
        W = numpy.random.uniform(-numpy.sqrt(1./hidden_dim), numpy.sqrt(1./hidden_dim), (3, hidden_dim, hidden_dim))
        b = numpy.zeros((3, hidden_dim))
        # Theano: Created shared variables
        U = debug_print(theano.shared(name='U', value=U.astype(theano.config.floatX), borrow=True), 'U')
        W = debug_print(theano.shared(name='W', value=W.astype(theano.config.floatX), borrow=True), 'W')
        b = debug_print(theano.shared(name='b', value=b.astype(theano.config.floatX), borrow=True), 'b')
        return U, W, b

def create_ensemble_para(rng, fan_in, fan_out):

        # initialize weights with random weights
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        W = theano.shared(numpy.asarray(
            rng.uniform(low=-W_bound, high=W_bound, size=(fan_out,fan_in)),
            dtype=theano.config.floatX),
                               borrow=True)

        return W
    
def create_highw_para(rng, fan_in, fan_out):

        # initialize weights with random weights
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        W = theano.shared(numpy.asarray(
            rng.uniform(low=-W_bound, high=W_bound, size=(fan_out,fan_in)),
            dtype=theano.config.floatX),
                               borrow=True)

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = numpy.zeros((fan_out,), dtype=theano.config.floatX)
        b = theano.shared(value=b_values, borrow=True)
        return W, b

def create_logistic_para(rng,fan_in,fan_out):
        
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        W = theano.shared(numpy.asarray(
            rng.uniform(low = - W_bound, high = W_bound, size = (fan_out,fan_in)),
            dtype = theano.config.floatX),
                                borrow = True)
        b_values = numpy.zeros((fan_out,),dtype = theano.config.floatX)
        b = theano.shared(value = b_values,borrow = True)
        
        return W,b
        
def create_conv_para(rng, filter_shape):
        fan_in = numpy.prod(filter_shape[1:])
        fan_out = filter_shape[0] * numpy.prod(filter_shape[2:])
        # initialize weights with random weights
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        W = theano.shared(numpy.asarray(
            rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
            dtype=theano.config.floatX),
                               borrow=True)

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        b = theano.shared(value=b_values, borrow=True)
        return W, b

def create_rnn_para(rng, dim):
        # initialize weights with random weights
        W_bound = numpy.sqrt(6. / (2*dim + dim))
#         Whh = theano.shared(numpy.asarray(
#             rng.uniform(low=-W_bound, high=W_bound, size=(dim, dim)),
#             dtype=theano.config.floatX),
#                                borrow=True)
#         Wxh = theano.shared(numpy.asarray(
#             rng.uniform(low=-W_bound, high=W_bound, size=(dim, dim)),
#             dtype=theano.config.floatX),
#                                borrow=True)
        W = theano.shared(numpy.asarray(
            rng.uniform(low=-W_bound, high=W_bound, size=(2*dim, dim)),
            dtype=theano.config.floatX),
                               borrow=True)        
        # the bias is a 1D tensor -- one bias per output feature map
        b_values = numpy.zeros((dim,), dtype=theano.config.floatX)
        b = theano.shared(value=b_values, borrow=True)
        return W, b

class Conv_with_input_para(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape, W, b):
        assert image_shape[1] == filter_shape[1]
        self.input = input
        self.W = W
        self.b = b

        # convolve input feature maps with filters
        conv_out = conv.conv2d(input=input, filters=self.W,
                filter_shape=filter_shape, image_shape=image_shape, border_mode='valid')    #here, we should pad enough zero padding for input 
        
        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1,n_filters,1,1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        conv_with_bias = T.tanh(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        narrow_conv_out=conv_with_bias.reshape((image_shape[0], 1, filter_shape[0], image_shape[3]-filter_shape[3]+1)) #(batch, 1, kernerl, ishape[1]-filter_size1[1]+1)
        
        #pad filter_size-1 zero embeddings at both sides
        left_padding = T.zeros((image_shape[0], 1, filter_shape[0], filter_shape[3]-1), dtype=theano.config.floatX)
        right_padding = T.zeros((image_shape[0], 1, filter_shape[0], filter_shape[3]-1), dtype=theano.config.floatX)
        self.output = T.concatenate([left_padding, narrow_conv_out, right_padding], axis=3) 
        

        # store parameters of this layer
        self.params = [self.W, self.b]

class RNN_with_input_para(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, rnn_Whh, rnn_Wxh, rnn_b, dim):
        self.input = input.transpose(1,0) #iterate over first dim
        self.Whh = rnn_Whh
        self.Wxh=rnn_Wxh
        self.b = rnn_b
        self.h0 = theano.shared(name='h0',
                                value=numpy.zeros(dim,
                                dtype=theano.config.floatX))
        def recurrence(x_t, h_tm1):
            w_t = T.nnet.sigmoid(T.dot(x_t, self.Wxh)
                                 + T.dot(h_tm1, self.Whh) + self.b)
            h_t=h_tm1*w_t+x_t*(1-w_t)
#             s_t = T.nnet.softmax(T.dot(h_t, self.w) + self.b)
            return h_t
        
        h, _ = theano.scan(fn=recurrence,
                                sequences=self.input,
                                outputs_info=self.h0,#[self.h0, None],
                                n_steps=self.input.shape[0])        
        self.output=h.reshape((self.input.shape[0], self.input.shape[1])).transpose(1,0)
        

        # store parameters of this layer
        self.params = [self.Whh, self.Wxh, self.b]
class Bi_GRU_Matrix_Input(object):
    def __init__(self, X, word_dim, hidden_dim, U, W, b, U_b, W_b, b_b, bptt_truncate):
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        
        def forward_prop_step(x_t, s_t1_prev):            
            # GRU Layer 1
            z_t1 =debug_print( T.nnet.sigmoid(U[0].dot(x_t) + W[0].dot(s_t1_prev) + b[0]), 'z_t1')
            r_t1 = debug_print(T.nnet.sigmoid(U[1].dot(x_t) + W[1].dot(s_t1_prev) + b[1]), 'r_t1')
            c_t1 = debug_print(T.tanh(U[2].dot(x_t) + W[2].dot(s_t1_prev * r_t1) + b[2]), 'c_t1')
            s_t1 = debug_print((T.ones_like(z_t1) - z_t1) * c_t1 + z_t1 * s_t1_prev, 's_t1')
            return s_t1
        
        s, updates = theano.scan(
            forward_prop_step,
            sequences=X.transpose(1,0),
            truncate_gradient=self.bptt_truncate,
            outputs_info=dict(initial=T.zeros(self.hidden_dim)))
        
#         self.output_matrix=debug_print(s.transpose(), 'GRU_Matrix_Input.output_matrix')
#         self.output_vector_mean=T.mean(self.output_matrix, axis=1)
#         self.output_vector_max=T.max(self.output_matrix, axis=1)
#         self.output_vector_last=self.output_matrix[:,-1]
        #backward
        X_b=X[:,::-1]
        def backward_prop_step(x_t_b, s_t1_prev_b):            
            # GRU Layer 1
            z_t1_b =debug_print( T.nnet.sigmoid(U_b[0].dot(x_t_b) + W_b[0].dot(s_t1_prev_b) + b_b[0]), 'z_t1_b')
            r_t1_b = debug_print(T.nnet.sigmoid(U_b[1].dot(x_t_b) + W_b[1].dot(s_t1_prev_b) + b_b[1]), 'r_t1_b')
            c_t1_b = debug_print(T.tanh(U_b[2].dot(x_t_b) + W_b[2].dot(s_t1_prev_b * r_t1_b) + b_b[2]), 'c_t1_b')
            s_t1_b = debug_print((T.ones_like(z_t1_b) - z_t1_b) * c_t1_b + z_t1_b * s_t1_prev_b, 's_t1_b')
            return s_t1_b
        
        s_b, updates_b = theano.scan(
            backward_prop_step,
            sequences=X_b.transpose(1,0),
            truncate_gradient=self.bptt_truncate,
            outputs_info=dict(initial=T.zeros(self.hidden_dim)))
        #dim: hidden_dim*2        
        self.output_matrix=debug_print(T.concatenate([s.transpose(), s_b.transpose()[:,::-1]], axis=0), 'Bi_GRU_Matrix_Input.output_matrix')
        self.output_vector_mean=T.mean(self.output_matrix, axis=1)
        self.output_vector_max=T.max(self.output_matrix, axis=1)
        #dim: hidden_dim*4
        self.output_vector_last=T.concatenate([self.output_matrix[:,-1], self.output_matrix[:,0]], axis=0)

class Bi_GRU_Tensor3_Input(object):
    def __init__(self, T, lefts, rights, hidden_dim, U, W, b, Ub,Wb,bb):
        T=debug_print(T,'T')
        lefts=debug_print(lefts, 'lefts')
        rights=debug_print(rights, 'rights')
        def recurrence(matrix, left, right):
            sub_matrix=debug_print(matrix[:,left:-right], 'sub_matrix')
            GRU_layer=Bi_GRU_Matrix_Input(sub_matrix, sub_matrix.shape[0], hidden_dim,U,W,b, Ub,Wb,bb, -1)
            return GRU_layer.output_vector_mean
        new_M, updates = theano.scan(recurrence,
                                     sequences=[T, lefts, rights],
                                     outputs_info=None)
        self.output=debug_print(new_M.transpose(), 'Bi_GRU_Tensor3_Input.output')
        
class GRU_Matrix_Input(object):
    def __init__(self, X, word_dim, hidden_dim, U, W, b, bptt_truncate):
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        
        def forward_prop_step(x_t, s_t1_prev):            
            # GRU Layer 1
            z_t1 =debug_print( T.nnet.sigmoid(U[0].dot(x_t) + W[0].dot(s_t1_prev) + b[0]), 'z_t1')
            r_t1 = debug_print(T.nnet.sigmoid(U[1].dot(x_t) + W[1].dot(s_t1_prev) + b[1]), 'r_t1')
            c_t1 = debug_print(T.tanh(U[2].dot(x_t) + W[2].dot(s_t1_prev * r_t1) + b[2]), 'c_t1')
            s_t1 = debug_print((T.ones_like(z_t1) - z_t1) * c_t1 + z_t1 * s_t1_prev, 's_t1')
            return s_t1
        
        s, updates = theano.scan(
            forward_prop_step,
            sequences=X.transpose(1,0),
            truncate_gradient=self.bptt_truncate,
            outputs_info=dict(initial=T.zeros(self.hidden_dim)))
        
        self.output_matrix=debug_print(s.transpose(), 'GRU_Matrix_Input.output_matrix')
        self.output_vector_mean=T.mean(self.output_matrix, axis=1)
        self.output_vector_max=T.max(self.output_matrix, axis=1)
        self.output_vector_last=self.output_matrix[:,-1]

class GRU_Tensor3_Input(object):
    def __init__(self, T, lefts, rights, hidden_dim, U, W, b):
        T=debug_print(T,'T')
        lefts=debug_print(lefts, 'lefts')
        rights=debug_print(rights, 'rights')
        def recurrence(matrix, left, right):
            sub_matrix=debug_print(matrix[:,left:-right], 'sub_matrix')
            GRU_layer=GRU_Matrix_Input(sub_matrix, sub_matrix.shape[0], hidden_dim,U,W,b, -1)
            return GRU_layer.output_vector_mean
        new_M, updates = theano.scan(recurrence,
                                     sequences=[T, lefts, rights],
                                     outputs_info=None)
        self.output=debug_print(new_M.transpose(), 'GRU_Tensor3_Input.output')

class biRNN_with_input_para(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, rnn_W, rnn_b, rnn_W_r, rnn_b_r, dim):
        self.input = debug_print(input.transpose(1,0), 'self.input') #iterate over first dim
        self.rnn_W=rnn_W
        self.b = rnn_b

        self.Wr = rnn_W_r
        self.b_r = rnn_b_r
        self.h0 = theano.shared(name='h0',
                                value=numpy.zeros(dim,
                                dtype=theano.config.floatX))
        self.h0_r = theano.shared(name='h0',
                                value=numpy.zeros(dim,
                                dtype=theano.config.floatX))
        def recurrence(x_t, h_tm1):
            concate=T.concatenate([x_t,h_tm1], axis=0)
#             w_t = T.nnet.sigmoid(T.dot(x_t, self.Wxh)
#                                  + T.dot(h_tm1, self.Whh) + self.b)
            w_t = T.nnet.sigmoid(T.dot(concate, self.rnn_W) + self.b)
            h_t=h_tm1*w_t+x_t*(1-w_t)
#             s_t = T.nnet.softmax(T.dot(h_t, self.w) + self.b)
            return h_t
         
        h, _ = theano.scan(fn=recurrence,
                                sequences=self.input,
                                outputs_info=self.h0,#[self.h0, None],
                                n_steps=self.input.shape[0])        
        self.output_one=debug_print(h.reshape((self.input.shape[0], self.input.shape[1])).transpose(1,0), 'self.output_one')
        #reverse direction
        self.input_two=debug_print(input[:,::-1].transpose(1,0), 'self.input_two')
        def recurrence_r(x_t_r, h_tm1_r):
            concate=T.concatenate([x_t_r,h_tm1_r], axis=0)
#             w_t = T.nnet.sigmoid(T.dot(x_t, self.Wxh)
#                                  + T.dot(h_tm1, self.Whh) + self.b)
            w_t = T.nnet.sigmoid(T.dot(concate, self.Wr) + self.b_r)
#             h_t=h_tm1*w_t+x_t*(1-w_t)
# #             s_t = T.nnet.softmax(T.dot(h_t, self.w) + self.b)
# 
# 
#             w_t = T.nnet.sigmoid(T.dot(x_t_r, self.Wxh_r)
#                                  + T.dot(h_tm1_r, self.Whh_r) + self.b_r)
            h_t=h_tm1_r*w_t+x_t_r*(1-w_t)
#             s_t = T.nnet.softmax(T.dot(h_t, self.w) + self.b)
            return h_t        
        h_r, _ = theano.scan(fn=recurrence_r,
                                sequences=self.input_two,
                                outputs_info=self.h0_r,#[self.h0, None],
                                n_steps=self.input_two.shape[0])        
        self.output_two=debug_print(h_r.reshape((self.input_two.shape[0], self.input_two.shape[1])).transpose(1,0)[:,::-1], 'self.output_two')
        self.output=debug_print(self.output_one+self.output_two, 'self.output')
#         # store parameters of this layer
#         self.params = [self.Whh, self.Wxh, self.b]
class Conv_with_input_para_one_col_featuremap(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape, W, b):
        assert image_shape[1] == filter_shape[1]
        self.input = input
        self.W = W
        self.b = b
        
        input=debug_print(input, 'input_Conv_with_input_para_one_col_featuremap')
        # convolve input feature maps with filters
        conv_out = conv.conv2d(input=input, filters=self.W,
                filter_shape=filter_shape, image_shape=image_shape, border_mode='full')    #here, we should pad enough zero padding for input 
        
        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1,n_filters,1,1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        conv_out=debug_print(conv_out, 'conv_out')
        conv_with_bias = debug_print(T.tanh(conv_out + self.b.dimshuffle('x', 0, 'x', 'x')), 'conv_with_bias')
        posi=conv_with_bias.shape[2]/2
        conv_with_bias=conv_with_bias[:,:,posi:(posi+1),:]
        wide_conv_out=debug_print(conv_with_bias.reshape((image_shape[0], 1, filter_shape[0], image_shape[3]+filter_shape[3]-1)), 'wide_conv_out') #(batch, 1, kernerl, ishape[1]+filter_size1[1]-1)
        

        self.output_tensor = debug_print(wide_conv_out, 'self.output_tensor')
        self.output_matrix=debug_print(wide_conv_out.reshape((filter_shape[0], image_shape[3]+filter_shape[3]-1)), 'self.output_matrix')
        self.output_sent_rep_Dlevel=debug_print(T.max(self.output_matrix, axis=1), 'self.output_sent_rep_Dlevel')
        

        # store parameters of this layer
        self.params = [self.W, self.b]
        
        
class Conv(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height,filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows,#cols)
        """

        assert image_shape[1] == filter_shape[1]
        self.input = input

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = numpy.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = filter_shape[0] * numpy.prod(filter_shape[2:])
        # initialize weights with random weights
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(numpy.asarray(
            rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
            dtype=theano.config.floatX),
                               borrow=True)

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        # convolve input feature maps with filters
        conv_out = conv.conv2d(input=input, filters=self.W,
                filter_shape=filter_shape, image_shape=image_shape, border_mode='valid')    #here, we should pad enough zero padding for input 
        
        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1,n_filters,1,1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        conv_with_bias = T.tanh(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        narrow_conv_out=conv_with_bias.reshape((image_shape[0], 1, filter_shape[0], image_shape[3]-filter_shape[3]+1)) #(batch, 1, kernerl, ishape[1]-filter_size1[1]+1)
        
        #pad filter_size-1 zero embeddings at both sides
        left_padding = 1e-20+T.zeros((image_shape[0], 1, filter_shape[0], filter_shape[3]-1), dtype=theano.config.floatX)
        right_padding = 1e-20+T.zeros((image_shape[0], 1, filter_shape[0], filter_shape[3]-1), dtype=theano.config.floatX)
        self.output = T.concatenate([left_padding, narrow_conv_out, right_padding], axis=3) 
        

        # store parameters of this layer
        self.params = [self.W, self.b]

class Average_Pooling_for_Top(object):
    """The input is output of Conv: a tensor.  The output here should also be tensor"""

    def __init__(self, rng, input_l, input_r, kern, left_l, right_l, left_r, right_r, length_l, length_r, dim, topk): # length_l, length_r: valid lengths after conv
#     layer3_DQ=Average_Pooling_for_Top(rng, input_l=layer2_DQ.output, input_r=layer2_Q.output_sent_rep_Dlevel, kern=nkerns[1],
#                      left_l=left_D, right_l=right_D, left_r=0, right_r=0, 
#                       length_l=len_D+filter_sents[1]-1, length_r=1,
#                        dim=maxDocLength+filter_sents[1]-1, topk=3)


        fan_in = kern #kern numbers
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = kern
        # initialize weights with random weights
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(numpy.asarray(
            rng.uniform(low=-W_bound, high=W_bound, size=(kern, kern)),
            dtype=theano.config.floatX),
                               borrow=True) #a weight matrix kern*kern
        
        input_r_matrix=debug_print(input_r,'input_r_matrix')

        input_l_matrix=debug_print(input_l.reshape((input_l.shape[2], input_l.shape[3])), 'origin_input_l_matrix')
        input_l_matrix=debug_print(input_l_matrix[:, left_l:(input_l_matrix.shape[1]-right_l)],'input_l_matrix')

            
            
        simi_matrix=compute_simi_feature_matrix_with_column(input_l_matrix, input_r_matrix, length_l, 1, dim) #(input.shape[0]/2, input.shape[1], input.shape[3], input.shape[3])
        simi_question=debug_print(simi_matrix.reshape((1, length_l)),'simi_question')
        
        neighborsArgSorted = T.argsort(simi_question, axis=1)
        kNeighborsArg = neighborsArgSorted[:,-topk:]#only average the top 3 vectors
        kNeighborsArgSorted = T.sort(kNeighborsArg, axis=1) # make y indices in acending lie
        jj = kNeighborsArgSorted.flatten()
        sub_matrix=input_l_matrix.transpose(1,0)[jj].reshape((topk, input_l_matrix.shape[0]))
        sub_weights=simi_question.transpose(1,0)[jj].reshape((topk, 1))
        
        sub_weights =sub_weights/T.sum(sub_weights) #L-1 normalize attentions
        #weights_answer=simi_answer/T.sum(simi_answer)    
        #concate=T.concatenate([weights_question, weights_answer], axis=1)
        #reshaped_concate=concate.reshape((input.shape[0], 1, 1, length_last_dim))
        
        sub_weights=T.repeat(sub_weights, kern, axis=1)
        #weights_answer_matrix=T.repeat(weights_answer, kern, axis=0)
        
        #with attention
#         output_D_doc_level_rep=debug_print(T.sum(sub_matrix*sub_weights, axis=0), 'output_D_doc_level_rep') # is a column now    
        output_D_doc_level_rep=debug_print(T.max(sub_matrix, axis=0), 'output_D_doc_level_rep') # is a column now 
        self.output_D_doc_level_rep=output_D_doc_level_rep    
        
        

        self.params = [self.W]



class Average_Pooling(object):
    """The input is output of Conv: a tensor.  The output here should also be tensor"""

    def __init__(self, rng, input_D, input_r, kern, left_D, right_D,left_D_s, right_D_s, left_r, right_r, length_D_s, length_r, dim, doc_len, topk): # length_l, length_r: valid lengths after conv
#     layer1_DQ=Average_Pooling(rng, input_l=layer0_D_output, input_r=layer0_Q_output, kern=nkerns[0],
#                                       left_D=left_D, right_D=right_D,
#                      left_l=left_D_s, right_l=right_D_s, left_r=left_Q, right_r=right_Q, 
#                       length_l=len_D_s+filter_words[1]-1, length_r=len_Q+filter_words[1]-1,
#                        dim=maxSentLength+filter_words[1]-1, doc_len=maxDocLength, topk=3)


        fan_in = kern #kern numbers
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = kern
        # initialize weights with random weights
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(numpy.asarray(
            rng.uniform(low=-W_bound, high=W_bound, size=(kern, kern)),
            dtype=theano.config.floatX),
                               borrow=True) #a weight matrix kern*kern
        input_r_matrix=debug_print(input_r.reshape((input_r.shape[2], input_r.shape[3])),'origin_input_r_matrix')
        input_r_matrix=debug_print(input_r_matrix[:, left_r:(input_r_matrix.shape[1]-right_r)],'input_r_matrix')
        valid_D_s=[]
        for i in range(left_D, doc_len-right_D): # only consider valid sentences in doc
            input_l=input_D[i,:,:,:] # order-3 tensor
            left_l=left_D_s[i]
            right_l=right_D_s[i]
            length_l=length_D_s[i]
            
            
            input_l_matrix=debug_print(input_l.reshape((input_D.shape[2], input_D.shape[3])), 'origin_input_l_matrix')
            input_l_matrix=debug_print(input_l_matrix[:, left_l:(input_l_matrix.shape[1]-right_l)],'input_l_matrix')

            
            
            simi_tensor=compute_simi_feature_batch1_new(input_l_matrix, input_r_matrix, length_l, length_r, self.W, dim) #(input.shape[0]/2, input.shape[1], input.shape[3], input.shape[3])
            simi_question=debug_print(T.max(simi_tensor, axis=1).reshape((1, length_l)),'simi_question')
            
            neighborsArgSorted = T.argsort(simi_question, axis=1)
            kNeighborsArg = neighborsArgSorted[:,-topk:]#only average the top 3 vectors
            kNeighborsArgSorted = T.sort(kNeighborsArg, axis=1) # make y indices in acending lie
            jj = kNeighborsArgSorted.flatten()
            sub_matrix=input_l_matrix.transpose(1,0)[jj].reshape((topk, input_l_matrix.shape[0]))
            sub_weights=simi_question.transpose(1,0)[jj].reshape((topk, 1))
            
            sub_weights =sub_weights/T.sum(sub_weights) #L-1 normalize attentions
            #weights_answer=simi_answer/T.sum(simi_answer)    
            #concate=T.concatenate([weights_question, weights_answer], axis=1)
            #reshaped_concate=concate.reshape((input.shape[0], 1, 1, length_last_dim))
            
            sub_weights=T.repeat(sub_weights, kern, axis=1)
            #weights_answer_matrix=T.repeat(weights_answer, kern, axis=0)
            
            #with attention
            dot_l=debug_print(T.sum(sub_matrix*sub_weights, axis=0).transpose(1,0), 'dot_l') # is a column now
            valid_D_s.append(dot_l)
            #dot_r=debug_print(T.sum(input_r_matrix*weights_answer_matrix, axis=1),'dot_r')      
            '''
            #without attention
            dot_l=debug_print(T.sum(input_l_matrix, axis=1), 'dot_l') # first add 1e-20 for each element to make non-zero input for weight gradient
            dot_r=debug_print(T.sum(input_r_matrix, axis=1),'dot_r')      
            '''
            '''
            #with attention, then max pooling
            dot_l=debug_print(T.max(input_l_matrix*weights_question_matrix, axis=1), 'dot_l') # first add 1e-20 for each element to make non-zero input for weight gradient
            dot_r=debug_print(T.max(input_r_matrix*weights_answer_matrix, axis=1),'dot_r')          
            '''
            #norm_l=debug_print(T.sqrt((dot_l**2).sum()),'norm_l')
            #norm_r=debug_print(T.sqrt((dot_r**2).sum()), 'norm_r')
            
            #self.output_vector_l=debug_print((dot_l/norm_l).reshape((1, kern)),'output_vector_l')
            #self.output_vector_r=debug_print((dot_r/norm_r).reshape((1, kern)), 'output_vector_r')      
        valid_matrix=T.concatenate(valid_D_s, axis=1)
        left_padding = T.zeros((input_l_matrix.shape[0], left_D), dtype=theano.config.floatX)
        right_padding = T.zeros((input_l_matrix.shape[0], right_D), dtype=theano.config.floatX)
        matrix_padded = T.concatenate([left_padding, valid_matrix, right_padding], axis=1)         
        self.output_D=matrix_padded
        self.output_D_valid_part=valid_matrix
        self.output_QA_sent_level_rep=T.mean(input_r_matrix, axis=1)
        
        #now, average pooling by comparing self.output_QA and self.output_D_valid_part
        simi_matrix=compute_simi_feature_matrix_with_column(self.output_D_valid_part, self.output_QA, doc_len-left_D-right_D, 1, doc_len) #(input.shape[0]/2, input.shape[1], input.shape[3], input.shape[3])
        simi_question=debug_print(simi_matrix.reshape((1, doc_len-left_D-right_D)),'simi_question')
        
        neighborsArgSorted = T.argsort(simi_question, axis=1)
        kNeighborsArg = neighborsArgSorted[:,-topk:]#only average the top 3 vectors
        kNeighborsArgSorted = T.sort(kNeighborsArg, axis=1) # make y indices in acending lie
        jj = kNeighborsArgSorted.flatten()
        sub_matrix=self.output_D_valid_part.transpose(1,0)[jj].reshape((topk, self.output_D_valid_part.shape[0]))
        sub_weights=simi_question.transpose(1,0)[jj].reshape((topk, 1))
        
        sub_weights =sub_weights/T.sum(sub_weights) #L-1 normalize attentions
        #weights_answer=simi_answer/T.sum(simi_answer)    
        #concate=T.concatenate([weights_question, weights_answer], axis=1)
        #reshaped_concate=concate.reshape((input.shape[0], 1, 1, length_last_dim))
        
        sub_weights=T.repeat(sub_weights, kern, axis=1)
        #weights_answer_matrix=T.repeat(weights_answer, kern, axis=0)
        
        #with attention
        output_D_sent_level_rep=debug_print(T.sum(sub_matrix*sub_weights, axis=0).transpose(1,0), 'output_D_sent_level_rep') # is a column now    
        self.output_D_sent_level_rep=output_D_sent_level_rep    
        
        

        self.params = [self.W]

class Average_Pooling_Scan(object):
    """The input is output of Conv: a tensor.  The output here should also be tensor"""

    def __init__(self, rng, input_D, input_r, kern, left_D, right_D,left_D_s, right_D_s, left_r, right_r, length_D_s, length_r, dim, doc_len, topk): # length_l, length_r: valid lengths after conv
#     layer1_DQ=Average_Pooling(rng, input_l=layer0_D_output, input_r=layer0_Q_output, kern=nkerns[0],
#                                       left_D=left_D, right_D=right_D,
#                      left_l=left_D_s, right_l=right_D_s, left_r=left_Q, right_r=right_Q, 
#                       length_l=len_D_s+filter_words[1]-1, length_r=len_Q+filter_words[1]-1,
#                        dim=maxSentLength+filter_words[1]-1, doc_len=maxDocLength, topk=3)


        fan_in = kern #kern numbers
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = kern
        # initialize weights with random weights
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(numpy.asarray(
            rng.uniform(low=-W_bound, high=W_bound, size=(kern, kern)),
            dtype=theano.config.floatX),
                               borrow=True) #a weight matrix kern*kern
        
#         input_tensor_l=T.dtensor4("input_tensor_l")
#         input_tensor_r=T.dtensor4("input_tensor_r")
#         kern_scan=T.lscalar("kern_scan")
#         length_D_s_scan=T.lvector("length_D_s_scan")
#         left_D_s_scan=T.lvector("left_D_s_scan")
#         right_D_s_scan=T.lvector("right_D_s_scan")
#         length_r_scan=T.lscalar("length_r_scan")
#         left_r_scan=T.lscalar("left_r_scan")
#         right_r_scan=T.lscalar("right_r_scan")
#         dim_scan=T.lscalar("dim_scan")
#         topk_scan=T.lscalar("topk_scan")
        

        
        def sub_operation(input_l, length_l, left_l, right_l, input_r, kernn , length_r, left_r, right_r, dim, topk):
            input_l_matrix=debug_print(input_l.reshape((input_l.shape[1], input_l.shape[2])), 'origin_input_l_matrix')#input_l should be order3 tensor now
            input_l_matrix=debug_print(input_l_matrix[:, left_l:(input_l_matrix.shape[1]-right_l)],'input_l_matrix')
#             input_r_matrix=debug_print(input_r.reshape((input_r.shape[2], input_r.shape[3])),'origin_input_r_matrix')#input_r should be order4 tensor still
#             input_r_matrix=debug_print(input_r_matrix[:, left_r:(input_r_matrix.shape[1]-right_r)],'input_r_matrix')
#              
#              
#             simi_tensor=compute_simi_feature_batch1_new(input_l_matrix, input_r_matrix, length_l, length_r, self.W, dim) #(input.shape[0]/2, input.shape[1], input.shape[3], input.shape[3])
#             simi_question=debug_print(T.max(simi_tensor, axis=1).reshape((1, length_l)),'simi_question')
#              
#             neighborsArgSorted = T.argsort(simi_question, axis=1)
#             kNeighborsArg = neighborsArgSorted[:,-topk:]#only average the top 3 vectors
#             kNeighborsArgSorted = T.sort(kNeighborsArg, axis=1) # make y indices in acending lie
#             jj = kNeighborsArgSorted.flatten()
#             sub_matrix=input_l_matrix.transpose(1,0)[jj].reshape((topk, input_l_matrix.shape[0]))
#             sub_weights=simi_question.transpose(1,0)[jj].reshape((topk, 1))
#               
#             sub_weights =sub_weights/T.sum(sub_weights) #L-1 normalize attentions
#             sub_weights=T.repeat(sub_weights, kernn, axis=1)
#             dot_l=debug_print(T.sum(sub_matrix*sub_weights, axis=0), 'dot_l') # is a column now     
#             dot_l=T.max(sub_matrix, axis=0)   
            dot_l=debug_print(T.max(input_l_matrix, axis=1), 'dot_l') # max pooling
            return dot_l

     
        
#         results, updates = theano.scan(fn=sub_operation,
#                                        outputs_info=None,
#                                        sequences=[input_tensor_l, length_D_s_scan, left_D_s_scan, right_D_s_scan],
#                                        non_sequences=[input_tensor_r, kern_scan, length_r_scan, left_r_scan, right_r_scan, dim_scan, topk_scan])

        results, updates = theano.scan(fn=sub_operation,
                                       outputs_info=None,
                                       sequences=[input_D[left_D:doc_len-right_D], length_D_s[left_D: doc_len-right_D], left_D_s[left_D: doc_len-right_D], right_D_s[left_D: doc_len-right_D]],
                                       non_sequences=[input_r, kern, length_r, left_r, right_r, dim, topk])
        
#         scan_function = theano.function(inputs=[input_tensor_l, input_tensor_r, kern_scan, length_D_s_scan, left_D_s_scan, right_D_s_scan, length_r_scan, left_r_scan, right_r_scan, dim_scan, topk_scan], 
#                                         outputs=results,
#                                         updates=updates)
# 
# 
#       
#         sents=scan_function(input_D[left_D:doc_len-right_D], input_r, kern, 
#                             length_D_s[left_D: doc_len-right_D], left_D_s[left_D: doc_len-right_D], right_D_s[left_D: doc_len-right_D], 
#                             length_r, 
#                             left_r, 
#                             right_r, 
#                             dim, 
#                             topk)
        sents=results
        input_r_matrix=debug_print(input_r.reshape((input_r.shape[2], input_r.shape[3])),'origin_input_r_matrix')
        input_r_matrix=debug_print(input_r_matrix[:, left_r:(input_r_matrix.shape[1]-right_r)],'input_r_matrix')


        valid_matrix=debug_print(sents.transpose(1,0), 'valid_matrix')
        left_padding = T.zeros((input_D.shape[2], left_D), dtype=theano.config.floatX)
        right_padding = T.zeros((input_D.shape[2], right_D), dtype=theano.config.floatX)
        matrix_padded = T.concatenate([left_padding, valid_matrix, right_padding], axis=1)         
        self.output_D=matrix_padded   #it shows the second conv for doc has input of all sentences
        self.output_D_valid_part=valid_matrix
        self.output_QA_sent_level_rep=T.max(input_r_matrix, axis=1)
        
        #now, average pooling by comparing self.output_QA and self.output_D_valid_part, choose one key sentence
        topk=1
        simi_matrix=debug_print(compute_simi_feature_matrix_with_column(self.output_D_valid_part, self.output_QA_sent_level_rep, doc_len-left_D-right_D, 1, doc_len), 'simi_matrix_matrix_with_column') #(input.shape[0]/2, input.shape[1], input.shape[3], input.shape[3])
        simi_question=debug_print(simi_matrix.reshape((1, doc_len-left_D-right_D)),'simi_question')
        
        neighborsArgSorted = T.argsort(simi_question, axis=1)
        kNeighborsArg = neighborsArgSorted[:,-topk:]#only average the top 3 vectors
        kNeighborsArgSorted = T.sort(kNeighborsArg, axis=1) # make y indices in acending lie
        jj = kNeighborsArgSorted.flatten()
        sub_matrix=self.output_D_valid_part.transpose(1,0)[jj].reshape((topk, self.output_D_valid_part.shape[0]))
        sub_weights=simi_question.transpose(1,0)[jj].reshape((topk, 1))
        
        sub_weights =sub_weights/T.sum(sub_weights) #L-1 normalize attentions
        #weights_answer=simi_answer/T.sum(simi_answer)    
        #concate=T.concatenate([weights_question, weights_answer], axis=1)
        #reshaped_concate=concate.reshape((input.shape[0], 1, 1, length_last_dim))
        
        sub_weights=T.repeat(sub_weights, kern, axis=1)
        #weights_answer_matrix=T.repeat(weights_answer, kern, axis=0)
        
        #with attention
#         output_D_sent_level_rep=debug_print(T.sum(sub_matrix*sub_weights, axis=0), 'output_D_sent_level_rep') # is a column now    
        output_D_sent_level_rep=debug_print(T.max(sub_matrix, axis=0), 'output_D_sent_level_rep') # is a column now    
        self.output_D_sent_level_rep=output_D_sent_level_rep    
        
        

        self.params = [self.W]

class GRU_Average_Pooling_Scan(object):
    """The input is output of Conv: a tensor.  The output here should also be tensor"""

    def __init__(self, rng, input_D, input_r, kern, left_D, right_D, dim, doc_len, topk): # length_l, length_r: valid lengths after conv
#     layer1_DQ=Average_Pooling(rng, input_l=layer0_D_output, input_r=layer0_Q_output, kern=nkerns[0],
#                                       left_D=left_D, right_D=right_D,
#                      left_l=left_D_s, right_l=right_D_s, left_r=left_Q, right_r=right_Q, 
#                       length_l=len_D_s+filter_words[1]-1, length_r=len_Q+filter_words[1]-1,
#                        dim=maxSentLength+filter_words[1]-1, doc_len=maxDocLength, topk=3)


#         fan_in = kern #kern numbers
#         # each unit in the lower layer receives a gradient from:
#         # "num output feature maps * filter height * filter width" /
#         #   pooling size
#         fan_out = kern
#         # initialize weights with random weights
#         W_bound = numpy.sqrt(6. / (fan_in + fan_out))
#         self.W = theano.shared(numpy.asarray(
#             rng.uniform(low=-W_bound, high=W_bound, size=(kern, kern)),
#             dtype=theano.config.floatX),
#                                borrow=True) #a weight matrix kern*kern
        
#         input_tensor_l=T.dtensor4("input_tensor_l")
#         input_tensor_r=T.dtensor4("input_tensor_r")
#         kern_scan=T.lscalar("kern_scan")
#         length_D_s_scan=T.lvector("length_D_s_scan")
#         left_D_s_scan=T.lvector("left_D_s_scan")
#         right_D_s_scan=T.lvector("right_D_s_scan")
#         length_r_scan=T.lscalar("length_r_scan")
#         left_r_scan=T.lscalar("left_r_scan")
#         right_r_scan=T.lscalar("right_r_scan")
#         dim_scan=T.lscalar("dim_scan")
#         topk_scan=T.lscalar("topk_scan")
        

        
#         def sub_operation(input_l, length_l, left_l, right_l, input_r, kernn , length_r, left_r, right_r, dim, topk):
#             input_l_matrix=debug_print(input_l.reshape((input_l.shape[1], input_l.shape[2])), 'origin_input_l_matrix')#input_l should be order3 tensor now
#             input_l_matrix=debug_print(input_l_matrix[:, left_l:(input_l_matrix.shape[1]-right_l)],'input_l_matrix')
# #             input_r_matrix=debug_print(input_r.reshape((input_r.shape[2], input_r.shape[3])),'origin_input_r_matrix')#input_r should be order4 tensor still
# #             input_r_matrix=debug_print(input_r_matrix[:, left_r:(input_r_matrix.shape[1]-right_r)],'input_r_matrix')
# #              
# #              
# #             simi_tensor=compute_simi_feature_batch1_new(input_l_matrix, input_r_matrix, length_l, length_r, self.W, dim) #(input.shape[0]/2, input.shape[1], input.shape[3], input.shape[3])
# #             simi_question=debug_print(T.max(simi_tensor, axis=1).reshape((1, length_l)),'simi_question')
# #              
# #             neighborsArgSorted = T.argsort(simi_question, axis=1)
# #             kNeighborsArg = neighborsArgSorted[:,-topk:]#only average the top 3 vectors
# #             kNeighborsArgSorted = T.sort(kNeighborsArg, axis=1) # make y indices in acending lie
# #             jj = kNeighborsArgSorted.flatten()
# #             sub_matrix=input_l_matrix.transpose(1,0)[jj].reshape((topk, input_l_matrix.shape[0]))
# #             sub_weights=simi_question.transpose(1,0)[jj].reshape((topk, 1))
# #               
# #             sub_weights =sub_weights/T.sum(sub_weights) #L-1 normalize attentions
# #             sub_weights=T.repeat(sub_weights, kernn, axis=1)
# #             dot_l=debug_print(T.sum(sub_matrix*sub_weights, axis=0), 'dot_l') # is a column now     
# #             dot_l=T.max(sub_matrix, axis=0)   
#             dot_l=debug_print(T.max(input_l_matrix, axis=1), 'dot_l') # max pooling
#             return dot_l
# 
#      
#         
# #         results, updates = theano.scan(fn=sub_operation,
# #                                        outputs_info=None,
# #                                        sequences=[input_tensor_l, length_D_s_scan, left_D_s_scan, right_D_s_scan],
# #                                        non_sequences=[input_tensor_r, kern_scan, length_r_scan, left_r_scan, right_r_scan, dim_scan, topk_scan])
# 
#         results, updates = theano.scan(fn=sub_operation,
#                                        outputs_info=None,
#                                        sequences=[input_D[left_D:doc_len-right_D], length_D_s[left_D: doc_len-right_D], left_D_s[left_D: doc_len-right_D], right_D_s[left_D: doc_len-right_D]],
#                                        non_sequences=[input_r, kern, length_r, left_r, right_r, dim, topk])
        
#         scan_function = theano.function(inputs=[input_tensor_l, input_tensor_r, kern_scan, length_D_s_scan, left_D_s_scan, right_D_s_scan, length_r_scan, left_r_scan, right_r_scan, dim_scan, topk_scan], 
#                                         outputs=results,
#                                         updates=updates)
# 
# 
#       
#         sents=scan_function(input_D[left_D:doc_len-right_D], input_r, kern, 
#                             length_D_s[left_D: doc_len-right_D], left_D_s[left_D: doc_len-right_D], right_D_s[left_D: doc_len-right_D], 
#                             length_r, 
#                             left_r, 
#                             right_r, 
#                             dim, 
#                             topk)
#         sents=results
#         input_r_matrix=debug_print(input_r.reshape((input_r.shape[2], input_r.shape[3])),'origin_input_r_matrix')
#         input_r_matrix=debug_print(input_r_matrix[:, left_r:(input_r_matrix.shape[1]-right_r)],'input_r_matrix')


        valid_matrix=debug_print(input_D, 'valid_matrix')
        left_padding = T.zeros((input_D.shape[0], left_D), dtype=theano.config.floatX)
        right_padding = T.zeros((input_D.shape[0], right_D), dtype=theano.config.floatX)
        matrix_padded = T.concatenate([left_padding, valid_matrix, right_padding], axis=1)         
        self.output_D=matrix_padded   #it shows the second conv for doc has input of all sentences
        self.output_D_valid_part=valid_matrix
        self.output_QA_sent_level_rep=input_r
        
        #now, average pooling by comparing self.output_QA and self.output_D_valid_part, choose one key sentence
        topk=1
        simi_matrix=debug_print(compute_simi_feature_matrix_with_column(self.output_D_valid_part, self.output_QA_sent_level_rep, doc_len-left_D-right_D, 1, doc_len), 'simi_matrix_matrix_with_column') #(input.shape[0]/2, input.shape[1], input.shape[3], input.shape[3])
        simi_question=debug_print(simi_matrix.reshape((1, doc_len-left_D-right_D)),'simi_question')
        
        neighborsArgSorted = T.argsort(simi_question, axis=1)
        kNeighborsArg = neighborsArgSorted[:,-topk:]#only average the top 3 vectors
        kNeighborsArgSorted = T.sort(kNeighborsArg, axis=1) # make y indices in acending lie
        jj = kNeighborsArgSorted.flatten()
        sub_matrix=self.output_D_valid_part.transpose(1,0)[jj].reshape((topk, self.output_D_valid_part.shape[0]))
        sub_weights=simi_question.transpose(1,0)[jj].reshape((topk, 1))
        
        sub_weights =sub_weights/T.sum(sub_weights) #L-1 normalize attentions
        #weights_answer=simi_answer/T.sum(simi_answer)    
        #concate=T.concatenate([weights_question, weights_answer], axis=1)
        #reshaped_concate=concate.reshape((input.shape[0], 1, 1, length_last_dim))
        
        sub_weights=T.repeat(sub_weights, kern, axis=1)
        #weights_answer_matrix=T.repeat(weights_answer, kern, axis=0)
        
        #with attention
#         output_D_sent_level_rep=debug_print(T.sum(sub_matrix*sub_weights, axis=0), 'output_D_sent_level_rep') # is a column now    
        output_D_sent_level_rep=debug_print(T.max(sub_matrix, axis=0), 'output_D_sent_level_rep') # is a column now    
        self.output_D_sent_level_rep=output_D_sent_level_rep    
        
        

#         self.params = [self.W]

def drop(input, p, rng): 
    """
    :type input: numpy.array
    :param input: layer or weight matrix on which dropout resp. dropconnect is applied
    
    :type p: float or double between 0. and 1. 
    :param p: p probability of NOT dropping out a unit or connection, therefore (1.-p) is the drop rate.
    
    """            
    srng = T.shared_randomstreams.RandomStreams(rng.randint(999999))
    mask = srng.binomial(n=1, p=p, size=input.shape, dtype=theano.config.floatX)
    return input * mask
class Average_Pooling_RNN(object):
    """The input is output of Conv: a tensor.  The output here should also be tensor"""

    def __init__(self, rng, input_D, input_r, kern, left_D, right_D,doc_len, topk): # length_l, length_r: valid lengths after conv
#     layer1_DQ=Average_Pooling(rng, input_l=layer0_D_output, input_r=layer0_Q_output, kern=nkerns[0],
#                                       left_D=left_D, right_D=right_D,
#                      left_l=left_D_s, right_l=right_D_s, left_r=left_Q, right_r=right_Q, 
#                       length_l=len_D_s+filter_words[1]-1, length_r=len_Q+filter_words[1]-1,
#                        dim=maxSentLength+filter_words[1]-1, doc_len=maxDocLength, topk=3)


        


        self.output_D_valid_part=input_D
        self.output_QA_sent_level_rep=input_r
        
        #now, average pooling by comparing self.output_QA and self.output_D_valid_part, choose one key sentence
        topk=1
        simi_matrix=debug_print(compute_simi_feature_matrix_with_column(self.output_D_valid_part, self.output_QA_sent_level_rep, doc_len-left_D-right_D, 1, doc_len), 'simi_matrix_matrix_with_column') #(input.shape[0]/2, input.shape[1], input.shape[3], input.shape[3])
        simi_question=debug_print(simi_matrix.reshape((1, doc_len-left_D-right_D)),'simi_question')
         
        neighborsArgSorted = T.argsort(simi_question, axis=1)
        kNeighborsArg = neighborsArgSorted[:,-topk:]#only average the top 3 vectors
        kNeighborsArgSorted = T.sort(kNeighborsArg, axis=1) # make y indices in acending lie
        jj = kNeighborsArgSorted.flatten()
        sub_matrix=self.output_D_valid_part.transpose(1,0)[jj].reshape((topk, self.output_D_valid_part.shape[0]))
        sub_weights=simi_question.transpose(1,0)[jj].reshape((topk, 1))
         
        sub_weights =sub_weights/T.sum(sub_weights) #L-1 normalize attentions
        #weights_answer=simi_answer/T.sum(simi_answer)    
        #concate=T.concatenate([weights_question, weights_answer], axis=1)
        #reshaped_concate=concate.reshape((input.shape[0], 1, 1, length_last_dim))
         
        sub_weights=T.repeat(sub_weights, kern, axis=1)
        #weights_answer_matrix=T.repeat(weights_answer, kern, axis=0)
         
        #with attention
#         output_D_sent_level_rep=debug_print(T.sum(sub_matrix*sub_weights, axis=0), 'output_D_sent_level_rep') # is a column now    
        output_D_rep=debug_print(T.max(sub_matrix, axis=0), 'output_D_rep') # is a column now    
        self.output_D_sent_level_rep=output_D_rep    


def compute_simi_feature_batch1_new(input_l_matrix, input_r_matrix, length_l, length_r, para_matrix, dim):
    #matrix_r_after_translate=debug_print(T.dot(para_matrix, input_r_matrix), 'matrix_r_after_translate')
    matrix_r_after_translate=input_r_matrix
    
    input_l_tensor=input_l_matrix.dimshuffle('x',0,1)
    input_l_tensor=T.repeat(input_l_tensor, dim, axis=0)[:length_r,:,:]
    input_l_tensor=input_l_tensor.dimshuffle(2,1,0).dimshuffle(0,2,1)
    repeated_1=input_l_tensor.reshape((length_l*length_r, input_l_matrix.shape[0])).dimshuffle(1,0)
    
    input_r_tensor=matrix_r_after_translate.dimshuffle('x',0,1)
    input_r_tensor=T.repeat(input_r_tensor, dim, axis=0)[:length_l,:,:]
    input_r_tensor=input_r_tensor.dimshuffle(0,2,1)
    repeated_2=input_r_tensor.reshape((length_l*length_r, matrix_r_after_translate.shape[0])).dimshuffle(1,0)
    

    
    #cosine attention   
    length_1=debug_print(1e-10+T.sqrt(T.sum(T.sqr(repeated_1), axis=0)),'length_1')
    length_2=debug_print(1e-10+T.sqrt(T.sum(T.sqr(repeated_2), axis=0)), 'length_2')

    multi=debug_print(repeated_1*repeated_2, 'multi')
    sum_multi=debug_print(T.sum(multi, axis=0),'sum_multi')
    
    list_of_simi= debug_print(sum_multi/(length_1*length_2),'list_of_simi')   #to get rid of zero length
    simi_matrix=debug_print(list_of_simi.reshape((length_l, length_r)), 'simi_matrix')
    
    
#     #euclid, effective for wikiQA
#     gap=debug_print(repeated_1-repeated_2, 'gap')
#     eucli=debug_print(T.sqrt(1e-10+T.sum(T.sqr(gap), axis=0)),'eucli')
#     simi_matrix=debug_print((1.0/(1.0+eucli)).reshape((length_l, length_r)), 'simi_matrix')
    
    
    return simi_matrix#[:length_l, :length_r]

def compute_simi_feature_matrix_with_column(input_l_matrix, column, length_l, length_r, dim):
    column=column.reshape((column.shape[0],1))
    repeated_2=T.repeat(column, dim, axis=1)[:,:length_l]
    

    
    #cosine attention   
    length_1=debug_print(1e-10+T.sqrt(T.sum(T.sqr(input_l_matrix), axis=0)),'length_1')
    length_2=debug_print(1e-10+T.sqrt(T.sum(T.sqr(repeated_2), axis=0)), 'length_2')

    multi=debug_print(input_l_matrix*repeated_2, 'multi')
    sum_multi=debug_print(T.sum(multi, axis=0),'sum_multi')
    
    list_of_simi= debug_print(sum_multi/(length_1*length_2),'list_of_simi')   #to get rid of zero length
    simi_matrix=debug_print(list_of_simi.reshape((length_l, length_r)), 'simi_matrix')
    
    
#     #euclid, effective for wikiQA
#     gap=debug_print(input_l_matrix-repeated_2, 'gap')
#     eucli=debug_print(T.sqrt(1e-10+T.sum(T.sqr(gap), axis=0)),'eucli')
#     simi_matrix=debug_print((1.0/(1.0+eucli)).reshape((length_l, length_r)), 'simi_matrix')
    
    
    return simi_matrix#[:length_l, :length_r]

def compute_simi_feature(tensor, dim, para_matrix):
    odd_tensor=debug_print(tensor[0:tensor.shape[0]:2,:,:,:],'odd_tensor')
    even_tensor=debug_print(tensor[1:tensor.shape[0]:2,:,:,:], 'even_tensor')
    even_tensor_after_translate=debug_print(T.dot(para_matrix, 1e-20+even_tensor.reshape((tensor.shape[2], dim*tensor.shape[0]/2))), 'even_tensor_after_translate')
    fake_even_tensor=debug_print(even_tensor_after_translate.reshape((tensor.shape[0]/2, tensor.shape[1], tensor.shape[2], tensor.shape[3])),'fake_even_tensor')

    repeated_1=debug_print(T.repeat(odd_tensor, dim, axis=3),'repeated_1')
    repeated_2=debug_print(repeat_whole_matrix(fake_even_tensor, dim, False),'repeated_2')
    #repeated_2=T.repeat(even_tensor, even_tensor.shape[3], axis=2).reshape((tensor.shape[0]/2, tensor.shape[1], tensor.shape[2], tensor.shape[3]**2))    
    length_1=debug_print(1e-10+T.sqrt(T.sum(T.sqr(repeated_1), axis=2)),'length_1')
    length_2=debug_print(1e-10+T.sqrt(T.sum(T.sqr(repeated_2), axis=2)), 'length_2')

    multi=debug_print(repeated_1*repeated_2, 'multi')
    sum_multi=debug_print(T.sum(multi, axis=2),'sum_multi')
    
    list_of_simi= debug_print(sum_multi/(length_1*length_2),'list_of_simi')   #to get rid of zero length
    
    return list_of_simi.reshape((tensor.shape[0]/2, tensor.shape[1], tensor.shape[3], tensor.shape[3]))

def compute_acc(label_list, scores_list):
    #label_list contains 0/1, 500 as a minibatch, score_list contains score between -1 and 1, 500 as a minibatch
    if len(label_list)%500!=0 or len(scores_list)%500!=0:
        print 'len(label_list)%500: ', len(label_list)%500, ' len(scores_list)%500: ', len(scores_list)%500
        exit(0)
    if len(label_list)!=len(scores_list):
        print 'len(label_list)!=len(scores_list)', len(label_list), ' and ',len(scores_list)
        exit(0)
    correct_count=0
    total_examples=len(label_list)/500
    start_posi=range(total_examples)*500
    for i in start_posi:
        set_1=set()
        
        for scan in range(i, i+500):
            if label_list[scan]==1:
                set_1.add(scan)
        set_0=set(range(i, i+500))-set_1
        flag=True
        for zero_posi in set_0:
            for scan in set_1:
                if scores_list[zero_posi]> scores_list[scan]:
                    flag=False
        if flag==True:
            correct_count+=1
    
    return correct_count*1.0/total_examples
#def unify_eachone(tensor, left1, right1, left2, right2, dim, Np):
def top_k_pooling(matrix, sentlength_1, sentlength_2, Np):

    #tensor: (1, feature maps, 66, 66)
    #sentlength_1=dim-left1-right1
    #sentlength_2=dim-left2-right2
    #core=tensor[:,:, left1:(dim-right1),left2:(dim-right2) ]
    '''
    repeat_row=Np/sentlength_1
    extra_row=Np%sentlength_1
    repeat_col=Np/sentlength_2
    extra_col=Np%sentlength_2    
    '''
    #repeat core
    matrix_1=repeat_whole_tensor(matrix, 5, True) 
    matrix_2=repeat_whole_tensor(matrix_1, 5, False)

    list_values=matrix_2.flatten()
    neighborsArgSorted = T.argsort(list_values)
    kNeighborsArg = neighborsArgSorted[-(Np**2):]    
    top_k_values=list_values[kNeighborsArg]
    

    all_max_value=top_k_values.reshape((1, Np**2))
    
    return all_max_value  
def unify_eachone(matrix, sentlength_1, sentlength_2, Np):

    #tensor: (1, feature maps, 66, 66)
    #sentlength_1=dim-left1-right1
    #sentlength_2=dim-left2-right2
    #core=tensor[:,:, left1:(dim-right1),left2:(dim-right2) ]

    repeat_row=Np/sentlength_1
    extra_row=Np%sentlength_1
    repeat_col=Np/sentlength_2
    extra_col=Np%sentlength_2    

    #repeat core
    matrix_1=repeat_whole_tensor(matrix, 5, True) 
    matrix_2=repeat_whole_tensor(matrix_1, 5, False)
    
    new_rows=T.maximum(sentlength_1, sentlength_1*repeat_row+extra_row)
    new_cols=T.maximum(sentlength_2, sentlength_2*repeat_col+extra_col)
    
    #core=debug_print(core_2[:,:, :new_rows, : new_cols],'core')
    new_matrix=debug_print(matrix_2[:new_rows,:new_cols], 'new_matrix')
    #determine x, y start positions
    size_row=new_rows/Np
    remain_row=new_rows%Np
    size_col=new_cols/Np
    remain_col=new_cols%Np
    
    xx=debug_print(T.concatenate([T.arange(Np-remain_row+1)*size_row, (Np-remain_row)*size_row+(T.arange(remain_row)+1)*(size_row+1)]),'xx')
    yy=debug_print(T.concatenate([T.arange(Np-remain_col+1)*size_col, (Np-remain_col)*size_col+(T.arange(remain_col)+1)*(size_col+1)]),'yy')
    
    list_of_maxs=[]
    for i in xrange(Np):
        for j in xrange(Np):
            region=debug_print(new_matrix[xx[i]:xx[i+1], yy[j]:yy[j+1]],'region')
            #maxvalue1=debug_print(T.max(region, axis=2), 'maxvalue1')
            maxvalue=debug_print(T.max(region).reshape((1,1)), 'maxvalue')
            list_of_maxs.append(maxvalue)
    

    all_max_value=T.concatenate(list_of_maxs, axis=1).reshape((1, Np**2))
    
    return all_max_value            


class Create_Attention_Input_Cnn(object):
    """The input is output of Conv: a tensor.  The output here should also be tensor"""

    def __init__(self, rng, tensor_l, tensor_r, dim,kern, l_left_pad, l_right_pad, r_left_pad, r_right_pad): # length_l, length_r: valid lengths after conv
        #first reshape into matrix
        matrix_l=tensor_l.reshape((tensor_l.shape[2], tensor_l.shape[3]))
        matrix_r=tensor_r.reshape((tensor_r.shape[2], tensor_r.shape[3]))
        #start
        repeated_1=debug_print(T.repeat(matrix_l, dim, axis=1),'repeated_1') # add 10 because max_sent_length is only input for conv, conv will make size bigger
        repeated_2=debug_print(repeat_whole_tensor(matrix_r, dim, False),'repeated_2')
        '''
        #cosine attention   
        length_1=debug_print(1e-10+T.sqrt(T.sum(T.sqr(repeated_1), axis=0)),'length_1')
        length_2=debug_print(1e-10+T.sqrt(T.sum(T.sqr(repeated_2), axis=0)), 'length_2')
    
        multi=debug_print(repeated_1*repeated_2, 'multi')
        sum_multi=debug_print(T.sum(multi, axis=0),'sum_multi')
        
        list_of_simi= debug_print(sum_multi/(length_1*length_2),'list_of_simi')   #to get rid of zero length
        simi_matrix=debug_print(list_of_simi.reshape((length_l, length_r)), 'simi_matrix')
        
        '''
        #euclid, effective for wikiQA
        gap=debug_print(repeated_1-repeated_2, 'gap')
        eucli=debug_print(T.sqrt(1e-10+T.sum(T.sqr(gap), axis=0)),'eucli')
        simi_matrix=debug_print((1.0/(1.0+eucli)).reshape((dim, dim)), 'simi_matrix')
        W_bound = numpy.sqrt(6. / (dim + kern))
        self.W = theano.shared(numpy.asarray(rng.uniform(low=-W_bound, high=W_bound, size=(kern, dim)),dtype=theano.config.floatX),borrow=True) #a weight matrix kern*kern
        matrix_l_attention=debug_print(T.dot(self.W, simi_matrix.T), 'matrix_l_attention')
        matrix_r_attention=debug_print(T.dot(self.W, simi_matrix), 'matrix_r_attention')
        #reset zero at both side
        left_zeros_l=T.set_subtensor(matrix_l_attention[:,:l_left_pad], T.zeros((matrix_l_attention.shape[0], l_left_pad), dtype=theano.config.floatX))
        right_zeros_l=T.set_subtensor(left_zeros_l[:,-l_right_pad:], T.zeros((matrix_l_attention.shape[0], l_right_pad), dtype=theano.config.floatX))
        left_zeros_r=T.set_subtensor(matrix_r_attention[:,:r_left_pad], T.zeros((matrix_r_attention.shape[0], r_left_pad), dtype=theano.config.floatX))
        right_zeros_r=T.set_subtensor(left_zeros_r[:,-r_right_pad:], T.zeros((matrix_r_attention.shape[0], r_right_pad), dtype=theano.config.floatX))       
        #combine with original input matrix
        self.new_tensor_l=T.concatenate([matrix_l,right_zeros_l], axis=0).reshape((tensor_l.shape[0], 2*tensor_l.shape[1], tensor_l.shape[2], tensor_l.shape[3])) 
        self.new_tensor_r=T.concatenate([matrix_r,right_zeros_r], axis=0).reshape((tensor_r.shape[0], 2*tensor_r.shape[1], tensor_r.shape[2], tensor_r.shape[3])) 
        
        self.params=[self.W]

    
    
    