import cPickle
import gzip
import os
import sys
sys.setrecursionlimit(6000)
import time

import numpy
import theano
import theano.tensor as T
import theano.sandbox.neighbours as TSN
import time

from logistic_sgd import LogisticRegression
from WPDefined import ConvFoldPoolLayer, dropout_from_layer, shared_dataset, repeat_whole_matrix
from cis.deep.utils.theano import debug_print
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv
from loadData import load_MCTest_corpus_DPN, load_word2vec_to_init
from word2embeddings.nn.util import zero_value, random_value_normal
from common_functions import Conv_with_input_para, Conv_with_input_para_one_col_featuremap, Average_Pooling_for_Top,create_logistic_para, create_conv_para, Average_Pooling, create_highw_para, Average_Pooling_Scan
from random import shuffle
'''
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
'''
from scipy import linalg, mat, dot

#from preprocess_wikiQA import compute_map_mrr

#need to change
'''
1) add linguistics features
3) dropout
5) add sent-level, doc-level and overall-level similarity together to ranking loss
6) shuffle training data
7) attention used cosine
8) reduce kern, emb size for overfitting



Doesnt work:
3) margin=0.5
4) euclidean distance
4) glove initialization
2) unknown words have different random vectors

'''



def evaluate_lenet5(file_name,vocab_file,train_file,dev_file,learning_rate=0.001, n_epochs=2000, nkerns=[90,90], batch_size=1, window_width=2,
                    maxSentLength=64, maxDocLength=60, emb_size=50, hidden_size=200,
                    L2_weight=0.0065, update_freq=1, norm_threshold=5.0, max_s_length=128, max_d_length=128, margin=0.3):
    maxSentLength=max_s_length+2*(window_width-1)
    maxDocLength=max_d_length+2*(window_width-1)
    model_options = locals().copy()
    f = open(file_name,'w')
    f.write("model options " + str(model_options)+ '\n')
    rng = numpy.random.RandomState(23455)
    train_data, _train_Label,train_size, test_data, _test_Label,test_size, vocab_size=load_MCTest_corpus_DPN(vocab_file, train_file, dev_file, max_s_length,maxSentLength, maxDocLength)#vocab_size contain train, dev and test
    f.write('train_size : ' + str(train_size))

    [train_data_D, train_data_A1, train_Label, 
                 train_Length_D,train_Length_D_s, train_Length_A1, 
                train_leftPad_D,train_leftPad_D_s, train_leftPad_A1, 
                train_rightPad_D,train_rightPad_D_s, train_rightPad_A1]=train_data
    [test_data_D, test_data_A1,  test_Label, 
                 test_Length_D,test_Length_D_s, test_Length_A1, 
                test_leftPad_D,test_leftPad_D_s, test_leftPad_A1, 
                test_rightPad_D,test_rightPad_D_s, test_rightPad_A1]=test_data                


    n_train_batches=train_size/batch_size
    n_test_batches=test_size/batch_size
    
    train_batch_start=list(numpy.arange(n_train_batches)*batch_size)
    test_batch_start=list(numpy.arange(n_test_batches)*batch_size)

    


    rand_values=random_value_normal((vocab_size+1, emb_size), theano.config.floatX, numpy.random.RandomState(1234))
    rand_values[0]=numpy.array(numpy.zeros(emb_size),dtype=theano.config.floatX)
    rand_values=load_word2vec_to_init(rand_values, 'vocab_glove_50d.txt')
    embeddings=theano.shared(value=rand_values, borrow=True)      
    
    error_sum=0
    
    # allocate symbolic variables for the data
    index = T.lscalar()
    index_D = T.lmatrix()   # now, x is the index matrix, must be integer
    index_A1= T.lvector()
    y = T.lscalar() 
    
    len_D=T.lscalar()
    len_D_s=T.lvector()
    len_A1=T.lscalar()

    left_D=T.lscalar()
    left_D_s=T.lvector()
    left_A1=T.lscalar()

    right_D=T.lscalar()
    right_D_s=T.lvector()
    right_A1=T.lscalar()
        


    ishape = (emb_size, maxSentLength)  # sentence shape
    dshape = (nkerns[0], maxDocLength) # doc shape
    filter_words=(emb_size,window_width)
    filter_sents=(nkerns[0], window_width)
    
    ######################
    # BUILD ACTUAL MODEL #
    ######################
    f.write('... building the model\n')

    layer0_D_input = embeddings[index_D.flatten()].reshape((maxDocLength,maxSentLength, emb_size)).transpose(0, 2, 1).dimshuffle(0, 'x', 1, 2)
    layer0_A1_input = embeddings[index_A1.flatten()].reshape((batch_size,maxSentLength, emb_size)).transpose(0, 2, 1).dimshuffle(0, 'x', 1, 2)
    
        
    conv_W, conv_b=create_conv_para(rng, filter_shape=(nkerns[0], 1, filter_words[0], filter_words[1]))
    layer0_para=[conv_W, conv_b] 
    conv2_W, conv2_b=create_conv_para(rng, filter_shape=(nkerns[1], 1, nkerns[0], filter_sents[1]))
    layer2_para=[conv2_W, conv2_b]
    high_W, high_b=create_highw_para(rng, nkerns[0], nkerns[1]) # this part decides nkern[0] and nkern[1] must be in the same dimension
    highW_para=[high_W, high_b]
    params = layer2_para+layer0_para+highW_para#+[embeddings]

    layer0_D = Conv_with_input_para(rng, input=layer0_D_input,
            image_shape=(maxDocLength, 1, ishape[0], ishape[1]),
            filter_shape=(nkerns[0], 1, filter_words[0], filter_words[1]), W=conv_W, b=conv_b)
    layer0_A1 = Conv_with_input_para(rng, input=layer0_A1_input,
            image_shape=(batch_size, 1, ishape[0], ishape[1]),
            filter_shape=(nkerns[0], 1, filter_words[0], filter_words[1]), W=conv_W, b=conv_b)
    
    layer0_D_output=debug_print(layer0_D.output, 'layer0_D.output')
    layer0_A1_output=debug_print(layer0_A1.output, 'layer0_A1.output')
       

    layer1_DA1=Average_Pooling_Scan(rng, input_D=layer0_D_output, input_r=layer0_A1_output, kern=nkerns[0],
                                      left_D=left_D, right_D=right_D,
                     left_D_s=left_D_s, right_D_s=right_D_s, left_r=left_A1, right_r=right_A1, 
                      length_D_s=len_D_s+filter_words[1]-1, length_r=len_A1+filter_words[1]-1,
                       dim=maxSentLength+filter_words[1]-1, doc_len=maxDocLength, topk=3)
    
    
    layer2_DA1 = Conv_with_input_para(rng, input=layer1_DA1.output_D.reshape((batch_size, 1, nkerns[0], dshape[1])),
            image_shape=(batch_size, 1, nkerns[0], dshape[1]),
            filter_shape=(nkerns[1], 1, nkerns[0], filter_sents[1]), W=conv2_W, b=conv2_b)
    layer2_A1 = Conv_with_input_para_one_col_featuremap(rng, input=layer1_DA1.output_QA_sent_level_rep.reshape((batch_size, 1, nkerns[0], 1)),
            image_shape=(batch_size, 1, nkerns[0], 1),
            filter_shape=(nkerns[1], 1, nkerns[0], filter_sents[1]), W=conv2_W, b=conv2_b)
    layer2_A1_output_sent_rep_Dlevel=debug_print(layer2_A1.output_sent_rep_Dlevel, 'layer2_A1.output_sent_rep_Dlevel')
    
    
    layer3_DA1=Average_Pooling_for_Top(rng, input_l=layer2_DA1.output, input_r=layer2_A1_output_sent_rep_Dlevel, kern=nkerns[1],
                     left_l=left_D, right_l=right_D, left_r=0, right_r=0, 
                      length_l=len_D+filter_sents[1]-1, length_r=1,
                       dim=maxDocLength+filter_sents[1]-1, topk=3)
    
    #high-way
    
    transform_gate_DA1=debug_print(T.nnet.sigmoid(T.dot(high_W, layer1_DA1.output_D_sent_level_rep) + high_b), 'transform_gate_DA1')
    transform_gate_A1=debug_print(T.nnet.sigmoid(T.dot(high_W, layer1_DA1.output_QA_sent_level_rep) + high_b), 'transform_gate_A1')
    
        
    overall_D_A1=(1.0-transform_gate_DA1)*layer1_DA1.output_D_sent_level_rep+transform_gate_DA1*layer3_DA1.output_D_doc_level_rep
    overall_A1=(1.0-transform_gate_A1)*layer1_DA1.output_QA_sent_level_rep+transform_gate_A1*layer2_A1.output_sent_rep_Dlevel
    
    simi_sent_level1=debug_print(cosine(layer1_DA1.output_D_sent_level_rep, layer1_DA1.output_QA_sent_level_rep), 'simi_sent_level1')
  
  
    simi_doc_level1=debug_print(cosine(layer3_DA1.output_D_doc_level_rep, layer2_A1.output_sent_rep_Dlevel), 'simi_doc_level1')

    
    simi_overall_level1=debug_print(cosine(overall_D_A1, overall_A1), 'simi_overall_level1')

 
    simi_1=(simi_overall_level1+simi_sent_level1+simi_doc_level1)/3.0
    logistic_w, logistic_b  = create_logistic_para(rng, 1,  2)
    logistic_para = [logistic_w,logistic_b]
    params += logistic_para
    simi_1 = T.dot(logistic_w,simi_1) + logistic_b.dimshuffle(0,'x')
    simi_1 = simi_1.dimshuffle(1,0)
    
    simi_1 = T.nnet.softmax(simi_1)
    predict = T.argmax(simi_1,axis = 1)
    tmp = T.log(simi_1)
    cost = T.maximum(0.0,margin + tmp[0][1-y] - tmp[0][y])
    L2_reg = (high_W**2).sum() + (conv2_W**2).sum() + (conv_W**2).sum() + (logistic_w **2 ).sum()
    cost = cost + L2_weight*L2_reg
 


    
    


    
    test_model = theano.function([index], [cost,simi_1,predict],
          givens={
            index_D: test_data_D[index], #a matrix
            index_A1: test_data_A1[index],
            y:test_Label[index],
            len_D: test_Length_D[index],
            len_D_s: test_Length_D_s[index],
            len_A1: test_Length_A1[index],

            left_D: test_leftPad_D[index],
            left_D_s: test_leftPad_D_s[index],
            left_A1: test_leftPad_A1[index],
        
            right_D: test_rightPad_D[index],
            right_D_s: test_rightPad_D_s[index],
            right_A1: test_rightPad_A1[index],
            
            }, on_unused_input='ignore')

    
    
    accumulator=[]
    for para_i in params:
        eps_p=numpy.zeros_like(para_i.get_value(borrow=True),dtype=theano.config.floatX)
        accumulator.append(theano.shared(eps_p, borrow=True))
      
    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)

    updates = []
    for param_i, grad_i, acc_i in zip(params, grads, accumulator):
        grad_i=debug_print(grad_i,'grad_i')
        acc = acc_i + T.sqr(grad_i)
        updates.append((param_i, param_i - learning_rate * grad_i / T.sqrt(acc)))   #AdaGrad
        updates.append((acc_i, acc))    
 
  
    train_model = theano.function([index], [cost,simi_1,predict], updates=updates,
          givens={
            index_D: train_data_D[index],
            index_A1: train_data_A1[index],
            y:train_Label[index],
            len_D: train_Length_D[index],
            len_D_s: train_Length_D_s[index],
            len_A1: train_Length_A1[index],

            left_D: train_leftPad_D[index],
            left_D_s: train_leftPad_D_s[index],
            left_A1: train_leftPad_A1[index],
        
            right_D: train_rightPad_D[index],
            right_D_s: train_rightPad_D_s[index],
            right_A1: train_rightPad_A1[index],
            }, on_unused_input='ignore')




    ###############
    # TRAIN MODEL #
    ###############
    f.write('... training\n')
    # early-stopping parameters
    patience = 500000000000000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_params = None
    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = time.clock()
    mid_time = start_time

    epoch = 0
    done_looping = False
    
    max_acc=0.0
    best_epoch=0

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        #for minibatch_index in xrange(n_train_batches): # each batch
        minibatch_index=0
        #shuffle(train_batch_start)#shuffle training data


        simi_train = []
        predict_train = []
        for batch_start in train_batch_start: 
            # iter means how many batches have been runed, taking into loop
            iter = (epoch - 1) * n_train_batches + minibatch_index +1 
            minibatch_index=minibatch_index+1
            
            cost_average,simi,predict = train_model(batch_start)
            simi_train.append(simi)
            predict_train.append(predict)
            if iter%1000 == 0:
                f.write('@iter :'  + str(iter) + '\n')
            if iter % n_train_batches == 0:
                corr_train = compute_corr_train(predict_train,_train_Label)
                res =  'training @ iter = '+str(iter)+' average cost: '+str(cost_average)+'corr rate: ' + str(corr_train * 100.0 / train_size) + '\n'
                f.write(res)
            
            if iter % validation_frequency == 0 or iter % 20000 == 0:
                posi_test_sent=[]
                nega_test_sent=[]
                posi_test_doc=[]
                nega_test_doc=[]
                posi_test_overall=[]
                nega_test_overall=[]
                
                simi_test = []
                predict_test = []
                for i in test_batch_start:
                    cost,simi,predict = test_model(i)
                    #print simi
                    #f.write('test_predict : ' + str(predict) + ' test_simi : ' + str(simi) + '\n' )
                    simi_test.append(simi)
                    predict_test.append(predict)
                corr_test = compute_corr(simi_test,predict_test,f)
                test_acc = corr_test*1.0 /(test_size/4.0)
                res = '\t\t\tepoch ' + str(epoch) + ', minibatch ' + str(minibatch_index) + ' / ' + str(n_train_batches) + ' test acc of best model ' + str(test_acc * 100.0) + '\n'
                f.write(res)
                 

  
                find_better=False
                if test_acc > max_acc:
                    max_acc=test_acc
                    best_epoch=epoch    
                    find_better=True              
                res =  '\t\t\tmax: ' +    str(max_acc) + ' (at ' + str(best_epoch) + ')\n'
                f.write(res)
                if find_better==True:
                    store_model_to_file(params, best_epoch, max_acc)
                    print 'Finished storing best params'  

            if patience <= iter:
                done_looping = True
                break
        
        
        print 'Epoch ', epoch, 'uses ', (time.clock()-mid_time)/60.0, 'min'
        mid_time = time.clock()
        #writefile.close()
   
        #print 'Batch_size: ', update_freq
    end_time = time.clock()
    print('Optimization complete.')
    print('Best validation score of %f %% obtained at iteration %i,'\
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))


def store_model_to_file(best_params, best_epoch, best_acc):
    save_file = open('Best_Para_at'+str(best_epoch)+'_'+str(best_acc), 'wb')  # this will overwrite current contents
    for para in best_params:           
        cPickle.dump(para.get_value(borrow=True), save_file, -1)  # the -1 is for HIGHEST_PROTOCOL
    save_file.close()

def cosine(vec1, vec2):
    vec1=debug_print(vec1, 'vec1')
    vec2=debug_print(vec2, 'vec2')
    norm_uni_l=T.sqrt((vec1**2).sum())
    norm_uni_r=T.sqrt((vec2**2).sum())
    
    dot=T.dot(vec1,vec2.T)
    
    simi=debug_print(dot/(norm_uni_l*norm_uni_r), 'uni-cosine')
    return simi#.reshape((1,1))    
def Linear(sum_uni_l, sum_uni_r):
    return (T.dot(sum_uni_l,sum_uni_r.T)).reshape((1,1))    
def Poly(sum_uni_l, sum_uni_r):
    dot=T.dot(sum_uni_l,sum_uni_r.T)
    poly=(0.5*dot+1)**3
    return poly.reshape((1,1))    
def Sigmoid(sum_uni_l, sum_uni_r):
    dot=T.dot(sum_uni_l,sum_uni_r.T)
    return T.tanh(1.0*dot+1).reshape((1,1))    
def RBF(sum_uni_l, sum_uni_r):
    eucli=T.sum((sum_uni_l-sum_uni_r)**2)
    return T.exp(-0.5*eucli).reshape((1,1))    
def GESD (sum_uni_l, sum_uni_r):
    eucli=1/(1+T.sum((sum_uni_l-sum_uni_r)**2))
    kernel=1/(1+T.exp(-(T.dot(sum_uni_l,sum_uni_r.T)+1)))
    return (eucli*kernel).reshape((1,1))   
def EUCLID(sum_uni_l, sum_uni_r):
    return T.sqrt(T.sqr(sum_uni_l-sum_uni_r).sum()+1e-20)#.reshape((1,1))
def load_model(params):
    save_file = open('/mounts/data/proj/wenpeng/Dataset/MCTest/Best_Para_at54_0.574583333333')
    
    for para in params:
        para.set_value(cPickle.load(save_file), borrow=True)
    save_file.close() 
def load_model_for_conv2(params,filename):
    save_file = open(filename)
    
    for para in params:
        para.set_value(cPickle.load(save_file), borrow=True)
    save_file.close()    

def compute_corr_train(predict,label):
    corr = 0
    for i in range(len(predict)):
        if predict[i] == label[i] :
            corr += 1
    return corr
    
def compute_corr(simi, predict,f):
    if len(simi)%4!=0 or len(predict)%4!=0:
        print 'len(simi)%4!=0 or len(predict)%4!=0'
        print len(simi), len(predict)
        exit(0)
    size=len(simi)
    batch=4
    n_batches=size/batch
    
    batch_start=list(numpy.arange(n_batches)*batch)
    corr=0
    for start in batch_start:
        sub_simi=simi[start:start+batch]
        sub_predict=predict[start:start+batch]
        succ=True
        max_simi = -11111
        max_index = -1
        for i in range(batch):
            if max_simi < sub_simi[i][0][1]:
                max_simi = sub_simi[i][0][1]
                max_index = i
        if max_index == 0:
            corr += 1
    return corr
        

if __name__ == '__main__':
    file_name = "train_log"
    vocab_file = sys.argv[1]
    train_file = sys.argv[2]
    dev_file = sys.argv[3]
    evaluate_lenet5(file_name,vocab_file,train_file,dev_file)
