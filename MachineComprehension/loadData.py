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

from logistic_sgd import LogisticRegression
from WPDefined import ConvFoldPoolLayer, dropout_from_layer, shared_dataset, repeat_whole_matrix
from cis.deep.utils.theano import debug_print
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

from operator import itemgetter
def load_huizi_corpus(vocabFile, trainFile, testFile, max_truncate,maxlength): #maxSentLength=45
    #first load word vocab
    read_vocab=open(vocabFile, 'r')
    vocab={}
    word_ind=1
    for line in read_vocab:
        tokens=line.strip().split()
        vocab[tokens[1]]=word_ind #word2id
        word_ind+=1
    read_vocab.close()
    #load train file
    def load_train_file(file, word2id):   
        read_file=open(file, 'r')
        data=[]
        Y=[]
        Lengths=[]
        #true_lengths=[]
        leftPad=[]
        rightPad=[]
        line_control=0
        for line in read_file:
            tokens=line.strip().split('\t')  # question, answer, label
            Y.append(int(tokens[0])) 
            #question
            for i in [1,2]:
                sent=[]
                words=tokens[i].strip().lower().split()  
                #true_lengths.append(len(words))
                length=0
                for word in words:
                    id=word2id.get(word)
                    if id is not None:
                        sent.append(id)
                        length+=1
                        if length==max_truncate: #we consider max 43 words
                            break
                if length==0:
                    #print 'shit sentence: ', tokens[i]
                    #exit(0)
                    break
                Lengths.append(length)
                left=(maxlength-length)/2
                right=maxlength-left-length
                leftPad.append(left)
                rightPad.append(right)
 
                sent=[0]*left+sent+[0]*right
                data.append(sent)
            line_control+=1
            #if line_control==50:
            #    break
        read_file.close()
        '''
        #normalized length
        arr=numpy.array(Lengths)
        max=numpy.max(arr)
        min=numpy.min(arr)
        normalized_lengths=(arr-min)*1.0/(max-min)
        '''
        #return numpy.array(data),numpy.array(Y), numpy.array(Lengths), numpy.array(leftPad),numpy.array(rightPad)
        return numpy.array(data),numpy.array(Y), numpy.array(Lengths), numpy.array(leftPad),numpy.array(rightPad)

    def load_test_file(file, word2id):
        read_file=open(file, 'r')
        data=[]
        Y=[]
        Lengths=[]
        #true_lengths=[]
        leftPad=[]
        rightPad=[]
        line_control=0
        for line in read_file:
            tokens=line.strip().split('\t')
            Y.append(int(tokens[0])) # make the label starts from 0 to 4
            #Y.append(int(tokens[0]))
            for i in [1,2]:
                sent=[]
                words=tokens[i].strip().lower().split()  
                #true_lengths.append(len(words))
                length=0
                for word in words:
                    id=word2id.get(word)
                    if id is not None:
                        sent.append(id)
                        length+=1
                        if length==max_truncate: #we consider max 43 words
                            break
                if length==0:
                    #print 'shit sentence: ', tokens[i]
                    #exit(0)
                    break
                Lengths.append(length)
                left=(maxlength-length)/2
                right=maxlength-left-length
                leftPad.append(left)
                rightPad.append(right) 
                sent=[0]*left+sent+[0]*right
                data.append(sent)
            #line_control+=1
            #if line_control==1000:
            #    break
        read_file.close()
        '''
        #normalized lengths
        arr=numpy.array(Lengths)
        max=numpy.max(arr)
        min=numpy.min(arr)
        normalized_lengths=(arr-min)*1.0/(max-min)
        '''
        #return numpy.array(data),numpy.array(Y), numpy.array(Lengths), numpy.array(leftPad),numpy.array(rightPad) 
        return numpy.array(data),numpy.array(Y), numpy.array(Lengths), numpy.array(leftPad),numpy.array(rightPad) 

    indices_train, trainY, trainLengths, trainLeftPad, trainRightPad=load_train_file(trainFile, vocab)
    print 'train file loaded over, total pairs: ', len(trainLengths)/2
    indices_test, testY, testLengths, testLeftPad, testRightPad=load_test_file(testFile, vocab)
    print 'test file loaded over, total pairs: ', len(testLengths)/2
    
    #now, we need normaliza sentence length in the whole dataset (training and test)
    concate_matrix=numpy.concatenate((trainLengths, testLengths), axis=0)
    max=numpy.max(concate_matrix)
    min=numpy.min(concate_matrix)    
    normalized_trainLengths=(trainLengths-min)*1.0/(max-min)
    normalized_testLengths=(testLengths-min)*1.0/(max-min)

    
    def shared_dataset(data_y, borrow=True):
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),  # @UndefinedVariable
                                 borrow=borrow)
        return T.cast(shared_y, 'int64')  # for ARC-II on gpu
        #return shared_y


    #indices_train=shared_dataset(indices_train)
    #indices_test=shared_dataset(indices_test)
    train_set_Lengths=shared_dataset(trainLengths)
    test_set_Lengths=shared_dataset(testLengths)
    
    normalized_train_length=theano.shared(numpy.asarray(normalized_trainLengths, dtype=theano.config.floatX),  borrow=True)                           
    normalized_test_length = theano.shared(numpy.asarray(normalized_testLengths, dtype=theano.config.floatX),  borrow=True)       
    
    train_left_pad=shared_dataset(trainLeftPad)
    train_right_pad=shared_dataset(trainRightPad)
    test_left_pad=shared_dataset(testLeftPad)
    test_right_pad=shared_dataset(testRightPad)
                                
    train_set_y=shared_dataset(trainY)                             
    test_set_y = shared_dataset(testY)
    

    rval = [(indices_train,train_set_y, train_set_Lengths, normalized_train_length, train_left_pad, train_right_pad), (indices_test, test_set_y, test_set_Lengths, normalized_test_length, test_left_pad, test_right_pad)]
    return rval, word_ind-1
def load_ibm_corpus(vocabFile, trainFile, devFile, maxlength):
    #first load word vocab
    read_vocab=open(vocabFile, 'r')
    vocab={}
    word_ind=1
    for line in read_vocab:
        tokens=line.strip().split()
        vocab[tokens[1]]=word_ind #word2id
        word_ind+=1
    read_vocab.close()
    sentlength_limit=1040
    #load train file
    def load_train_file(file, word2id):   
        read_file=open(file, 'r')
        data=[]
        Lengths=[]
        leftPad=[]
        rightPad=[]
        line_control=0
        for line in read_file:
            tokens=line.strip().split('\t')  # label, question, answer
            #question
            for i in range(1,3):
                sent=[]
                words=tokens[i].strip().split()  
                length=len(words)
                if length>sentlength_limit:
                    words=words[:sentlength_limit]
                    length=sentlength_limit
                Lengths.append(length)
                left=(maxlength-length)/2
                right=maxlength-left-length
                leftPad.append(left)
                rightPad.append(right)
                if left<0 or right<0:
                    print 'Too long sentence:\n'+tokens[i]
                    exit(0)   
                sent+=[0]*left
                for word in words:
                    sent.append(word2id.get(word))
                sent+=[0]*right
                data.append(sent)
                del sent
                del words
            line_control+=1
            if line_control%100==0:
                print line_control
        read_file.close()
        return numpy.array(data),numpy.array(Lengths), numpy.array(leftPad),numpy.array(rightPad)

    def load_dev_file(file, word2id):
        read_file=open(file, 'r')
        data=[]
        Y=[]
        Lengths=[]
        leftPad=[]
        rightPad=[]
        line_control=0
        for line in read_file:
            tokens=line.strip().split('\t')
            Y.append(int(tokens[0])) # make the label starts from 0 to 4
            for i in range(1,3):
                sent=[]
                words=tokens[i].strip().split() 
                length=len(words)
                if length>sentlength_limit:
                    words=words[:sentlength_limit]
                    length=sentlength_limit
                Lengths.append(length)
                left=(maxlength-length)/2
                right=maxlength-left-length
                leftPad.append(left)
                rightPad.append(right)
                if left<0 or right<0:
                    print 'Too long sentence:\n'+line
                    exit(0)  
                sent+=[0]*left
                for word in words:
                    sent.append(word2id.get(word))
                sent+=[0]*right
                data.append(sent)
            line_control+=1
            #if line_control==1000:
            #    break
        read_file.close()
        return numpy.array(data),Y, numpy.array(Lengths), numpy.array(leftPad),numpy.array(rightPad) 

    indices_train, trainLengths, trainLeftPad, trainRightPad=load_train_file(trainFile, vocab)
    print 'train file loaded over, total pairs: ', len(trainLengths)/2
    indices_dev, devY, devLengths, devLeftPad, devRightPad=load_dev_file(devFile, vocab)
    print 'dev file loaded over, total pairs: ', len(devLengths)/2
   

    
    def shared_dataset(data_y, borrow=True):
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),  # @UndefinedVariable
                                 borrow=borrow)
        return T.cast(shared_y, 'int32')
        #return shared_y


    train_set_Lengths=shared_dataset(trainLengths)                             
    valid_set_Lengths = shared_dataset(devLengths)
    
    train_left_pad=shared_dataset(trainLeftPad)
    train_right_pad=shared_dataset(trainRightPad)
    dev_left_pad=shared_dataset(devLeftPad)
    dev_right_pad=shared_dataset(devRightPad)
                                
    #valid_set_y = shared_dataset(devY)
    

    rval = [(indices_train,train_set_Lengths, train_left_pad, train_right_pad), (indices_dev, devY, valid_set_Lengths, dev_left_pad, dev_right_pad)]
    return rval, word_ind-1

def load_word2vec_to_init(rand_values, file):

    readFile=open(file, 'r')
    line_count=1
    for line in readFile:
        tokens=line.strip().split()
        rand_values[line_count]=numpy.array(map(float, tokens[1:]))
        line_count+=1                                            
    readFile.close()
    print 'initialization over...'
    return rand_values
    
def load_msr_corpus(vocabFile, trainFile, testFile, maxlength): #maxSentLength=60
    #first load word vocab
    read_vocab=open(vocabFile, 'r')
    vocab={}
    word_ind=1
    for line in read_vocab:
        tokens=line.strip().split()
        vocab[tokens[1]]=word_ind #word2id
        word_ind+=1
    read_vocab.close()
    #load train file
    def load_train_file(file, word2id):   
        read_file=open(file, 'r')
        data=[]
        Y=[]
        Lengths=[]
        leftPad=[]
        rightPad=[]
        line_control=0
        for line in read_file:
            tokens=line.strip().split('\t')  # label, sent1, sent2
            Y.append(int(tokens[0])) #repeat
            Y.append(int(tokens[0])) 
            #question
            for i in [1,2,2,1]: #shuffle the example
                sent=[]
                words=tokens[i].strip().lower().split()  
                length=0
                for word in words:
                    id=word2id.get(word)
                    if id is not None:
                        sent.append(id)
                        length+=1

                Lengths.append(length)
                left=(maxlength-length)/2
                right=maxlength-left-length
                leftPad.append(left)
                rightPad.append(right)
                if left<0 or right<0:
                    print 'Too long sentence:\n'+tokens[i]
                    exit(0)   
                sent=[0]*left+sent+[0]*right
                data.append(sent)
            #line_control+=1
        read_file.close()
        '''
        #normalized length
        arr=numpy.array(Lengths)
        max=numpy.max(arr)
        min=numpy.min(arr)
        normalized_lengths=(arr-min)*1.0/(max-min)
        '''
        #return numpy.array(data),numpy.array(Y), numpy.array(Lengths), numpy.array(leftPad),numpy.array(rightPad)
        return numpy.array(data),numpy.array(Y), numpy.array(Lengths), numpy.array(leftPad),numpy.array(rightPad)

    def load_test_file(file, word2id):
        read_file=open(file, 'r')
        data=[]
        Y=[]
        Lengths=[]
        leftPad=[]
        rightPad=[]
        line_control=0
        for line in read_file:
            tokens=line.strip().split('\t')
            Y.append(int(tokens[0])) # make the label starts from 0 to 4
            #Y.append(int(tokens[0]))
            for i in [1,2]:
                sent=[]
                words=tokens[i].strip().lower().split()  
                length=0
                for word in words:
                    id=word2id.get(word)
                    if id is not None:
                        sent.append(id)
                        length+=1

                Lengths.append(length)
                left=(maxlength-length)/2
                right=maxlength-left-length
                leftPad.append(left)
                rightPad.append(right)
                if left<0 or right<0:
                    print 'Too long sentence:\n'+tokens[i]
                    exit(0)   
                sent=[0]*left+sent+[0]*right
                data.append(sent)
            #line_control+=1
            #if line_control==1000:
            #    break
        read_file.close()
        '''
        #normalized lengths
        arr=numpy.array(Lengths)
        max=numpy.max(arr)
        min=numpy.min(arr)
        normalized_lengths=(arr-min)*1.0/(max-min)
        '''
        #return numpy.array(data),numpy.array(Y), numpy.array(Lengths), numpy.array(leftPad),numpy.array(rightPad) 
        return numpy.array(data),numpy.array(Y), numpy.array(Lengths), numpy.array(leftPad),numpy.array(rightPad) 

    indices_train, trainY, trainLengths, trainLeftPad, trainRightPad=load_train_file(trainFile, vocab)
    print 'train file loaded over, total pairs: ', len(trainLengths)/2
    indices_test, testY, testLengths, testLeftPad, testRightPad=load_test_file(testFile, vocab)
    print 'test file loaded over, total pairs: ', len(testLengths)/2
    
    #now, we need normaliza sentence length in the whole dataset (training and test)
    concate_matrix=numpy.concatenate((trainLengths, testLengths), axis=0)
    max=numpy.max(concate_matrix)
    min=numpy.min(concate_matrix)    
    normalized_trainLengths=(trainLengths-min)*1.0/(max-min)
    normalized_testLengths=(testLengths-min)*1.0/(max-min)

    
    def shared_dataset(data_y, borrow=True):
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),  # @UndefinedVariable
                                 borrow=borrow)
        return T.cast(shared_y, 'int64')
        #return shared_y


    #indices_train=shared_dataset(indices_train)
    #indices_test=shared_dataset(indices_test)
    train_set_Lengths=shared_dataset(trainLengths)
    test_set_Lengths=shared_dataset(testLengths)
    
    normalized_train_length=theano.shared(numpy.asarray(normalized_trainLengths, dtype=theano.config.floatX),  borrow=True)                           
    normalized_test_length = theano.shared(numpy.asarray(normalized_testLengths, dtype=theano.config.floatX),  borrow=True)       
    
    train_left_pad=shared_dataset(trainLeftPad)
    train_right_pad=shared_dataset(trainRightPad)
    test_left_pad=shared_dataset(testLeftPad)
    test_right_pad=shared_dataset(testRightPad)
                                
    train_set_y=shared_dataset(trainY)                             
    test_set_y = shared_dataset(testY)
    

    rval = [(indices_train,train_set_y, train_set_Lengths, normalized_train_length, train_left_pad, train_right_pad), (indices_test, test_set_y, test_set_Lengths, normalized_test_length, test_left_pad, test_right_pad)]
    return rval, word_ind-1

def load_mts(train_file, test_file):
    read_train=open(train_file, 'r')
    train_values=[]
    for line in read_train:
        tokens=map(float, line.strip().split())
        train_values.append(tokens)
        train_values.append(tokens)#repeat once
    read_train.close()
    read_test=open(test_file, 'r')
    test_values=[]
    for line in read_test:
        tokens=map(float, line.strip().split())
        test_values.append(tokens)
    read_test.close()
    
    train_values=theano.shared(numpy.asarray(train_values, dtype=theano.config.floatX), borrow=True)
    test_values=theano.shared(numpy.asarray(test_values, dtype=theano.config.floatX), borrow=True)
    
    return train_values, test_values

def load_mts_wikiQA(train_file, test_file):
    read_train=open(train_file, 'r')
    train_values=[]
    for line in read_train:
        tokens=map(float, line.strip().split())
        train_values.append(tokens)
    read_train.close()
    read_test=open(test_file, 'r')
    test_values=[]
    for line in read_test:
        tokens=map(float, line.strip().split())
        test_values.append(tokens)
    read_test.close()
    
    train_values=theano.shared(numpy.asarray(train_values, dtype=theano.config.floatX), borrow=True)
    test_values=theano.shared(numpy.asarray(test_values, dtype=theano.config.floatX), borrow=True)
    
    return train_values, test_values

def load_extra_features(train_file, test_file):
    read_train=open(train_file, 'r')
    train_values=[]
    for line in read_train:
        tokens=map(float, line.strip().split())
        train_values.append(tokens)
    read_train.close()
    read_test=open(test_file, 'r')
    test_values=[]
    for line in read_test:
        tokens=map(float, line.strip().split())
        test_values.append(tokens)
    read_test.close()
    
    train_values=theano.shared(numpy.asarray(train_values, dtype=theano.config.floatX), borrow=True)
    test_values=theano.shared(numpy.asarray(test_values, dtype=theano.config.floatX), borrow=True)
    
    return train_values, test_values
def load_wmf_wikiQA(train_file, test_file):
    read_train=open(train_file, 'r')
    train_values=[]
    for line in read_train:
        tokens=map(float, line.strip().split())
        train_values.append(tokens)
    read_train.close()
    read_test=open(test_file, 'r')
    test_values=[]
    for line in read_test:
        tokens=map(float, line.strip().split())
        test_values.append(tokens)
    read_test.close()
    
    train_values=theano.shared(numpy.asarray(train_values, dtype=theano.config.floatX), borrow=True)
    test_values=theano.shared(numpy.asarray(test_values, dtype=theano.config.floatX), borrow=True)
    
    return train_values, test_values
def load_MCTest_corpus(vocabFile, trainFile, testFile, max_truncate,maxlength, max_doc_length): #maxSentLength=45
    #first load word vocab
    read_vocab=open(vocabFile, 'r')
    vocab={}
    word_ind=1
    for line in read_vocab:
        tokens=line.strip().split()
        vocab[tokens[1]]=word_ind #word2id
        word_ind+=1
    read_vocab.close()
    #load train file
    def load_file(infile, word2id):   
        read_file=open(infile, 'r')
        data_D=[] #docs, will be a order-3 tensor
        data_Q=[] #question
        data_A=[] #answers
        Y=[]
        Label=[]
        Length_D=[] #true lengths of docs
        Length_D_s=[] # true lengths of each sentences in docs
        Length_Q=[]
        Length_A=[]
        leftPad_D=[]
        leftPad_D_s=[]
        leftPad_Q=[]
        leftPad_A=[]
        rightPad_D=[]
        rightPad_D_s=[]
        rightPad_Q=[]
        rightPad_A=[]                                

        line_control=0
        for line in read_file:
            tokens=line.strip().split('\t')  # question, answer, label
            Y.append(int(tokens[0]))
            Label.append(int(tokens[1])) 

            #doc
            data_D_s=[]# will be a matrix
            length_D_s=[]#a vector
            left_D_s=[] # vector
            right_D_s=[] #vector
            
            doc_len=len(tokens)-4 # remove two labels, one question, one answer           
            left_D=(max_doc_length-doc_len)/2
            for i in range(left_D):#pad empty sentences
                sent=[0]*maxlength
                #update four depository
                data_D_s.append(sent)
                length_D_s.append(0)
                left_D_s.append(maxlength/2)
                right_D_s.append(maxlength-maxlength/2)


            for s in tokens[2:-2]: #load valid sentences
                sent=[]
                words=s.strip().split()  
                for word in words:
                    id=word2id.get(word)
                    if id is not None:
                        sent.append(id)
                length=len(words)
                left=(maxlength-length)/2
                right=maxlength-left-length
                sent=[0]*left+sent+[0]*right
                #update four depository
                data_D_s.append(sent)
                length_D_s.append(length)
                left_D_s.append(left)
                right_D_s.append(right)
            
            right_D=max_doc_length-left_D-doc_len
            for i in range(right_D):#pad empty sentences
                sent=[0]*maxlength
                #update four depository
                data_D_s.append(sent)
                length_D_s.append(0)
                left_D_s.append(maxlength/2)
                right_D_s.append(maxlength-maxlength/2)            

            data_D.append(data_D_s) # add one slice
            Length_D.append(doc_len)
            leftPad_D.append(left_D)
            rightPad_D.append(right_D)                        
            #store above four depository
            Length_D_s.append(length_D_s)
            leftPad_D_s.append(left_D_s)
            rightPad_D_s.append(right_D_s)
            #Q
            words=tokens[-2].strip().split()
            len_q=len(words)
            left=(maxlength-len_q)/2
            right=maxlength-left-len_q          
            sent=[]
            for word in words:
                id=word2id.get(word)
                if id is not None:
                    sent.append(id)
            sent=[0]*left+sent+[0]*right
            data_Q.append(sent)
            Length_Q.append(len_q)
            leftPad_Q.append(left)
            rightPad_Q.append(right)             
            #A
            words=tokens[-1].strip().split()
            len_a=len(words)
            left=(maxlength-len_a)/2
            right=maxlength-left-len_a           
            sent=[]
            for word in words:
                id=word2id.get(word)
                if id is not None:
                    sent.append(id)
            sent=[0]*left+sent+[0]*right
            data_A.append(sent)            
            Length_A.append(len_a)
            leftPad_A.append(left)
            rightPad_A.append(right)             
            line_control+=1
            #if line_control==50:
            #    break
        read_file.close()
 
        
        results=[numpy.array(data_D), numpy.array(data_Q), numpy.array(data_A), numpy.array(Y), numpy.array(Label), 
                 numpy.array(Length_D),numpy.array(Length_D_s), numpy.array(Length_Q), numpy.array(Length_A),
                numpy.array(leftPad_D),numpy.array(leftPad_D_s), numpy.array(leftPad_Q), numpy.array(leftPad_A),
                numpy.array(rightPad_D),numpy.array(rightPad_D_s), numpy.array(rightPad_Q), numpy.array(rightPad_A)]
        return results, line_control



    train_data, train_size=load_file(trainFile, vocab)
    print 'train file loaded over'
    test_data, test_size=load_file(testFile, vocab)
    print 'test file loaded over'


    
    def shared_dataset(data_y, borrow=True):
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),  # @UndefinedVariable
                                 borrow=borrow)
        return T.cast(shared_y, 'int64')  
        #return shared_y
    
    train_list=[shared_dataset(matt) for matt in train_data]
    test_list=[shared_dataset(matt) for matt in test_data]       

    return train_list, train_size, test_list, test_size, word_ind-1

def load_MCTest_corpus_DQAAAA(vocabFile, trainFile, testFile, max_truncate,maxlength, max_doc_length): #maxSentLength=45
    #first load word vocab
    read_vocab=open(vocabFile, 'r')
    vocab={}
    word_ind=1
    for line in read_vocab:
        tokens=line.strip().split()
        vocab[tokens[1]]=word_ind #word2id
        word_ind+=1
    read_vocab.close()
    #load train file
    def load_file(infile, word2id):   
        read_file=open(infile, 'r')
        data_D=[] #docs, will be a order-3 tensor
        data_Q=[] #question
        data_A1=[] #positive answers
        data_A2=[]
        data_A3=[]
        data_A4=[]
#         Y=[]
        Label=[]
        Length_D=[] #true lengths of docs
        Length_D_s=[] # true lengths of each sentences in docs
        Length_Q=[]
        Length_A1=[]
        Length_A2=[]
        Length_A3=[]
        Length_A4=[]
        leftPad_D=[]
        leftPad_D_s=[]
        leftPad_Q=[]
        leftPad_A1=[]
        leftPad_A2=[]
        leftPad_A3=[]
        leftPad_A4=[]
        rightPad_D=[]
        rightPad_D_s=[]
        rightPad_Q=[]
        rightPad_A1=[]       
        rightPad_A2=[]  
        rightPad_A3=[]  
        rightPad_A4=[]                           

        line_control=0
        for line in read_file:
            tokens=line.strip().lower().split('\t')  # question, answer, label, lowercase always
#             Y.append(int(tokens[0]))
            Label.append(int(tokens[0])) #1 or 2 means single or multiple

            #doc
            data_D_s=[]# will be a matrix
            length_D_s=[]#a vector
            left_D_s=[] # vector
            right_D_s=[] #vector
            
            doc_len=len(tokens)-6 # remove one label, one question, four answers           
            left_D=(max_doc_length-doc_len)/2
            for i in range(left_D):#pad empty sentences
                sent=[0]*maxlength
                #update four depository
                data_D_s.append(sent)
                length_D_s.append(0)
                left_D_s.append(maxlength/2)
                right_D_s.append(maxlength-maxlength/2)


            for s in tokens[1:-5]: #load valid sentences
                sent=[]
                words=s.strip().split()  
                for word in words:
                    id=word2id.get(word)
                    if id is not None:
                        sent.append(id)
                length=len(words)
                left=(maxlength-length)/2
                right=maxlength-left-length
                sent=[0]*left+sent+[0]*right
                #update four depository
                data_D_s.append(sent)
                length_D_s.append(length)
                left_D_s.append(left)
                right_D_s.append(right)
            
            right_D=max_doc_length-left_D-doc_len
            for i in range(right_D):#pad empty sentences
                sent=[0]*maxlength
                #update four depository
                data_D_s.append(sent)
                length_D_s.append(0)
                left_D_s.append(maxlength/2)
                right_D_s.append(maxlength-maxlength/2)            

            data_D.append(data_D_s) # add one slice
            Length_D.append(doc_len)
            leftPad_D.append(left_D)
            rightPad_D.append(right_D)                        
            #store above four depository
            Length_D_s.append(length_D_s)
            leftPad_D_s.append(left_D_s)
            rightPad_D_s.append(right_D_s)
            #Q
            words=tokens[-5].strip().split()
            len_q=len(words)
            left=(maxlength-len_q)/2
            right=maxlength-left-len_q          
            sent=[]
            for word in words:
                id=word2id.get(word)
                if id is not None:
                    sent.append(id)
            sent=[0]*left+sent+[0]*right
            data_Q.append(sent)
            Length_Q.append(len_q)
            leftPad_Q.append(left)
            rightPad_Q.append(right)             
            #A1
            words=tokens[-4].strip().split()
            len_a=len(words)
            left=(maxlength-len_a)/2
            right=maxlength-left-len_a           
            sent=[]
            for word in words:
                id=word2id.get(word)
                if id is not None:
                    sent.append(id)
            sent=[0]*left+sent+[0]*right
            data_A1.append(sent)            
            Length_A1.append(len_a)
            leftPad_A1.append(left)
            rightPad_A1.append(right)         
            #A2
            words=tokens[-3].strip().split()
            len_a=len(words)
            left=(maxlength-len_a)/2
            right=maxlength-left-len_a           
            sent=[]
            for word in words:
                id=word2id.get(word)
                if id is not None:
                    sent.append(id)
            sent=[0]*left+sent+[0]*right
            data_A2.append(sent)            
            Length_A2.append(len_a)
            leftPad_A2.append(left)
            rightPad_A2.append(right)   
            #A1
            words=tokens[-2].strip().split()
            len_a=len(words)
            left=(maxlength-len_a)/2
            right=maxlength-left-len_a           
            sent=[]
            for word in words:
                id=word2id.get(word)
                if id is not None:
                    sent.append(id)
            sent=[0]*left+sent+[0]*right
            data_A3.append(sent)            
            Length_A3.append(len_a)
            leftPad_A3.append(left)
            rightPad_A3.append(right)   
            #A1
            words=tokens[-1].strip().split()
            len_a=len(words)
            left=(maxlength-len_a)/2
            right=maxlength-left-len_a           
            sent=[]
            for word in words:
                id=word2id.get(word)
                if id is not None:
                    sent.append(id)
            sent=[0]*left+sent+[0]*right
            data_A4.append(sent)            
            Length_A4.append(len_a)
            leftPad_A4.append(left)
            rightPad_A4.append(right)       
            line_control+=1
            #if line_control==50:
            #    break
        read_file.close()
 
        
        results=[numpy.array(data_D), numpy.array(data_Q), numpy.array(data_A1), numpy.array(data_A2), numpy.array(data_A3), numpy.array(data_A4), numpy.array(Label), 
                 numpy.array(Length_D),numpy.array(Length_D_s), numpy.array(Length_Q), numpy.array(Length_A1), numpy.array(Length_A2), numpy.array(Length_A3), numpy.array(Length_A4),
                numpy.array(leftPad_D),numpy.array(leftPad_D_s), numpy.array(leftPad_Q), numpy.array(leftPad_A1), numpy.array(leftPad_A2), numpy.array(leftPad_A3), numpy.array(leftPad_A4),
                numpy.array(rightPad_D),numpy.array(rightPad_D_s), numpy.array(rightPad_Q), numpy.array(rightPad_A1), numpy.array(rightPad_A2), numpy.array(rightPad_A3), numpy.array(rightPad_A4)]
        return results, line_control



    train_data, train_size=load_file(trainFile, vocab)
    print 'train file loaded over, train size:', train_size
    test_data, test_size=load_file(testFile, vocab)
    print 'test file loaded over, test_size:', test_size


    
    def shared_dataset(data_y, borrow=True):
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),  # @UndefinedVariable
                                 borrow=borrow)
        return T.cast(shared_y, 'int64')  
        #return shared_y
    
    train_list=[shared_dataset(matt) for matt in train_data]
    test_list=[shared_dataset(matt) for matt in test_data]       

    return train_list, train_size, test_list, test_size, word_ind-1

def load_MCTest_corpus_DSSSS(vocabFile, trainFile, testFile, max_truncate,maxlength, max_doc_length): #maxSentLength=45
    #first load word vocab
    read_vocab=open(vocabFile, 'r')
    vocab={}
    word_ind=1
    for line in read_vocab:
        tokens=line.strip().split()
        vocab[tokens[1]]=word_ind #word2id
        word_ind+=1
    read_vocab.close()
    #load train file
    def load_file(infile, word2id):   
        read_file=open(infile, 'r')
        data_D=[] #docs, will be a order-3 tensor
#         data_Q=[] #question
        data_A1=[] #positive answers
        data_A2=[]
        data_A3=[]
        data_A4=[]
#         Y=[]
        Label=[]
        Length_D=[] #true lengths of docs
        Length_D_s=[] # true lengths of each sentences in docs
#         Length_Q=[]
        Length_A1=[]
        Length_A2=[]
        Length_A3=[]
        Length_A4=[]
        leftPad_D=[]
        leftPad_D_s=[]
#         leftPad_Q=[]
        leftPad_A1=[]
        leftPad_A2=[]
        leftPad_A3=[]
        leftPad_A4=[]
        rightPad_D=[]
        rightPad_D_s=[]
#         rightPad_Q=[]
        rightPad_A1=[]       
        rightPad_A2=[]  
        rightPad_A3=[]  
        rightPad_A4=[]                           

        line_control=0
        for line in read_file:
            tokens=line.strip().split('\t')  # question, answer, label
#             Y.append(int(tokens[0]))
            print "len tokens : " , len(tokens)
            if len(tokens) <= 4:
                continue
            Label.append(int(tokens[0])) #1 or 2 means single or multiple

            #doc
            data_D_s=[]# will be a matrix
            length_D_s=[]#a vector
            left_D_s=[] # vector
            right_D_s=[] #vector
            
            doc_len=len(tokens)-5 # remove one label, four statements           
            left_D=(max_doc_length-doc_len)/2
            for i in range(left_D):#pad empty sentences
                sent=[0]*maxlength
                #update four depository
                data_D_s.append(sent)
                length_D_s.append(0)
                left_D_s.append(maxlength/2)
                right_D_s.append(maxlength-maxlength/2)


            for s in tokens[1:-4]: #load valid sentences
                sent=[]
                words=s.strip().split()  
                for word in words:
                    id=word2id.get(word)
                    if id is not None:
                        sent.append(id)
                length=len(words)
                left=(maxlength-length)/2
                right=maxlength-left-length
                sent=[0]*left+sent+[0]*right
                #update four depository
                data_D_s.append(sent)
                length_D_s.append(length)
                left_D_s.append(left)
                right_D_s.append(right)
            
            right_D=max_doc_length-left_D-doc_len
            for i in range(right_D):#pad empty sentences
                sent=[0]*maxlength
                #update four depository
                data_D_s.append(sent)
                length_D_s.append(0)
                left_D_s.append(maxlength/2)
                right_D_s.append(maxlength-maxlength/2)            

            data_D.append(data_D_s) # add one slice
            Length_D.append(doc_len)
            leftPad_D.append(left_D)
            rightPad_D.append(right_D)                        
            #store above four depository
            Length_D_s.append(length_D_s)
            leftPad_D_s.append(left_D_s)
            rightPad_D_s.append(right_D_s)
#             #Q
#             words=tokens[-5].strip().split()
#             len_q=len(words)
#             left=(maxlength-len_q)/2
#             right=maxlength-left-len_q          
#             sent=[]
#             for word in words:
#                 id=word2id.get(word)
#                 if id is not None:
#                     sent.append(id)
#             sent=[0]*left+sent+[0]*right
#             data_Q.append(sent)
#             Length_Q.append(len_q)
#             leftPad_Q.append(left)
#             rightPad_Q.append(right)             
            #A1
            words=tokens[-4].strip().split()
            len_a=len(words)
            left=(maxlength-len_a)/2
            right=maxlength-left-len_a           
            sent=[]
            for word in words:
                id=word2id.get(word)
                if id is not None:
                    sent.append(id)
            sent=[0]*left+sent+[0]*right
            data_A1.append(sent)            
            Length_A1.append(len_a)
            leftPad_A1.append(left)
            rightPad_A1.append(right)         
            #A2
            words=tokens[-3].strip().split()
            len_a=len(words)
            left=(maxlength-len_a)/2
            right=maxlength-left-len_a           
            sent=[]
            for word in words:
                id=word2id.get(word)
                if id is not None:
                    sent.append(id)
            sent=[0]*left+sent+[0]*right
            data_A2.append(sent)            
            Length_A2.append(len_a)
            leftPad_A2.append(left)
            rightPad_A2.append(right)   
            #A1
            words=tokens[-2].strip().split()
            len_a=len(words)
            left=(maxlength-len_a)/2
            right=maxlength-left-len_a           
            sent=[]
            for word in words:
                id=word2id.get(word)
                if id is not None:
                    sent.append(id)
            sent=[0]*left+sent+[0]*right
            data_A3.append(sent)            
            Length_A3.append(len_a)
            leftPad_A3.append(left)
            rightPad_A3.append(right)   
            #A1
            words=tokens[-1].strip().split()
            len_a=len(words)
            left=(maxlength-len_a)/2
            right=maxlength-left-len_a           
            sent=[]
            for word in words:
                id=word2id.get(word)
                if id is not None:
                    sent.append(id)
            sent=[0]*left+sent+[0]*right
            data_A4.append(sent)            
            Length_A4.append(len_a)
            leftPad_A4.append(left)
            rightPad_A4.append(right)       
            line_control+=1
            #if line_control==50:
            #    break
        read_file.close()
 
        
        results=[numpy.array(data_D), numpy.array(data_A1), numpy.array(data_A2), numpy.array(data_A3), numpy.array(data_A4), numpy.array(Label), 
                 numpy.array(Length_D),numpy.array(Length_D_s), numpy.array(Length_A1), numpy.array(Length_A2), numpy.array(Length_A3), numpy.array(Length_A4),
                numpy.array(leftPad_D),numpy.array(leftPad_D_s), numpy.array(leftPad_A1), numpy.array(leftPad_A2), numpy.array(leftPad_A3), numpy.array(leftPad_A4),
                numpy.array(rightPad_D),numpy.array(rightPad_D_s), numpy.array(rightPad_A1), numpy.array(rightPad_A2), numpy.array(rightPad_A3), numpy.array(rightPad_A4)]
        return results, line_control



    train_data, train_size=load_file(trainFile, vocab)
    print 'train file loaded over, train size:', train_size

    test_data, test_size=load_file(testFile, vocab)
    print 'test file loaded over, test_size:', test_size


    
    def shared_dataset(data_y, borrow=True):
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),  # @UndefinedVariable
                                 borrow=borrow)
        return T.cast(shared_y, 'int64')  
        #return shared_y
    
    train_list=[shared_dataset(matt) for matt in train_data]

    test_list=[shared_dataset(matt) for matt in test_data]       

    return train_list, train_size, test_list, test_size, word_ind-1
    
    
def load_MCTest_corpus_DPN_lambda(vocabFile, trainFile, testFile, max_truncate,maxlength, max_doc_length): #maxSentLength=45
    read_vocab = open(vocabFile,'r')
    vocab={}
    word_ind=1
    for line in read_vocab:
        tokens=line.strip().split()
        vocab[tokens[1]]=word_ind #word2id
        word_ind+=1
    read_vocab.close()
    
    def load_file(infile, word2id):   
        read_file=open(infile, 'r')
        data_D=[] #docs, will be a order-3 tensor
#         data_Q=[] #question
        data_A1=[] #positive answers
      #  data_A2=[]
#         data_A3=[]
#         data_A4=[]
#         Y=[]
        Label=[]
        Length_D=[] #true lengths of docs
        Length_D_s=[] # true lengths of each sentences in docs
#         Length_Q=[]
        Length_A1=[]
      #  Length_A2=[]
#         Length_A3=[]
#         Length_A4=[]
        leftPad_D=[]
        leftPad_D_s=[]
#         leftPad_Q=[]
        leftPad_A1=[]
      #  leftPad_A2=[]
#         leftPad_A3=[]
#         leftPad_A4=[]
        rightPad_D=[]
        rightPad_D_s=[]
#         rightPad_Q=[]
        rightPad_A1=[]       
       # rightPad_A2=[]  
#         rightPad_A3=[]  
#         rightPad_A4=[]                           
        lambda_label = []
        line_control=0
        num = 0
        for line in read_file:
            tokens=line.strip().split('\t')  # question, answer, label
#             Y.append(int(tokens[0]))
            #print tokens
            if len(tokens) <=4:
                continue
            lambda_label.append(int(tokens[0]))
            Label.append(int(tokens[1])) #1 or 2 means single or multiple
            #num += 1
            #print num
            #doc
            data_D_s=[]# will be a matrix
            length_D_s=[]#a vector
            left_D_s=[] # vector
            right_D_s=[] #vector
            
            doc_len=len(tokens)-3 # remove one label, two statements           
            left_D=(max_doc_length-doc_len)/2
            for i in range(left_D):#pad empty sentences
                sent=[0]*maxlength
                #update four depository
                data_D_s.append(sent)
                length_D_s.append(0)
                left_D_s.append(maxlength/2)
                right_D_s.append(maxlength-maxlength/2)


            for s in tokens[2:-1]: #load valid sentences
                s = s.lower()
               # print s
                sent=[]
                words=s.strip().split()  
                for word in words:
                    id=word2id.get(word)
                    if id is not None:
                        sent.append(id)
                    else:
                        sent.append(0)
                length=len(sent)
                left=(maxlength-length)/2
                right=maxlength-left-length
                sent=[0]*left+sent+[0]*right
                #update four depository
                data_D_s.append(sent)
                length_D_s.append(length)
                left_D_s.append(left)
                right_D_s.append(right)
               # print sent
               # raw_input()
            right_D=max_doc_length-left_D-doc_len
            for i in range(right_D):#pad empty sentences
                sent=[0]*maxlength
                #update four depository
                data_D_s.append(sent)
                length_D_s.append(0)
                left_D_s.append(maxlength/2)
                right_D_s.append(maxlength-maxlength/2)            

            data_D.append(data_D_s) # add one slice
            Length_D.append(doc_len)
            leftPad_D.append(left_D)
            rightPad_D.append(right_D)                        
            #store above four depository
            Length_D_s.append(length_D_s)
            leftPad_D_s.append(left_D_s)
            rightPad_D_s.append(right_D_s)
#             #Q
#             words=tokens[-5].strip().split()
#             len_q=len(words)
#             left=(maxlength-len_q)/2
#             right=maxlength-left-len_q          
#             sent=[]
#             for word in words:
#                 id=word2id.get(word)
#                 if id is not None:
#                     sent.append(id)
#             sent=[0]*left+sent+[0]*right
#             data_Q.append(sent)
#             Length_Q.append(len_q)
#             leftPad_Q.append(left)
#             rightPad_Q.append(right)             
            #A1
            words=tokens[-1].strip().split()
            #len_a=len(words)           
            sent=[]
            for word in words:
                id=word2id.get(word)
                if id is not None:
                    sent.append(id)
                else:
                    sent.append(0)
            len_a = len(sent)
            left=(maxlength-len_a)/2
            right=maxlength-left-len_a
            sent=[0]*left+sent+[0]*right
            data_A1.append(sent)            
            Length_A1.append(len_a)
            leftPad_A1.append(left)
            rightPad_A1.append(right)         
            #A2
            '''
            words=tokens[-1].strip().split()
            len_a=len(words)
            left=(maxlength-len_a)/2
            right=maxlength-left-len_a           
            sent=[]
            for word in words:
                id=word2id.get(word)
                if id is not None:
                    sent.append(id)
            sent=[0]*left+sent+[0]*right
            data_A2.append(sent)            
            Length_A2.append(len_a)
            leftPad_A2.append(left)
            rightPad_A2.append(right)  
            '''         
#             #A1
#             words=tokens[-2].strip().split()
#             len_a=len(words)
#             left=(maxlength-len_a)/2
#             right=maxlength-left-len_a           
#             sent=[]
#             for word in words:
#                 id=word2id.get(word)
#                 if id is not None:
#                     sent.append(id)
#             sent=[0]*left+sent+[0]*right
#             data_A3.append(sent)            
#             Length_A3.append(len_a)
#             leftPad_A3.append(left)
#             rightPad_A3.append(right)   
#             #A1
#             words=tokens[-1].strip().split()
#             len_a=len(words)
#             left=(maxlength-len_a)/2
#             right=maxlength-left-len_a           
#             sent=[]
#             for word in words:
#                 id=word2id.get(word)
#                 if id is not None:
#                     sent.append(id)
#             sent=[0]*left+sent+[0]*right
#             data_A4.append(sent)            
#             Length_A4.append(len_a)
#             leftPad_A4.append(left)
#             rightPad_A4.append(right)       
            line_control+=1
            #if line_control==50:
            #    break
        read_file.close()
 
        results=[numpy.array(data_D), numpy.array(data_A1),  numpy.array(Label), numpy.array(lambda_label),
                 numpy.array(Length_D),numpy.array(Length_D_s),  numpy.array(Length_A1),
                numpy.array(leftPad_D),numpy.array(leftPad_D_s), numpy.array(leftPad_A1),
                numpy.array(rightPad_D),numpy.array(rightPad_D_s),  numpy.array(rightPad_A1)]
        return results, Label, line_control



    train_data, train_Label,train_size=load_file(trainFile, vocab)
    print 'train file loaded over, train size:', train_size

    test_data, test_Label,test_size=load_file(testFile, vocab)
    #print 'test file loaded over, test_size:', test_size
    #print 'Lable', train_data[2]
    #print 'data_d', train_data[0]
    #print 'lambda_label', train_data[3]
    #raw_input()

    
    def shared_dataset(data_y, borrow=True):
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),  # @UndefinedVariable
                                 borrow=borrow)
        return T.cast(shared_y, 'int64')  
        #return shared_y
   
    
    train_list=[shared_dataset(matt) for matt in train_data]

    test_list=[shared_dataset(matt) for matt in test_data]       

    return train_list, train_Label,train_size, test_list,test_Label,test_size, word_ind-1
    
def load_MCTest_corpus_DPN(vocabFile, trainFile, testFile, max_truncate,maxlength, max_doc_length): #maxSentLength=45
    #first load word vocab
    read_vocab=open(vocabFile, 'r')
    vocab={}
    word_ind=1
    for line in read_vocab:
        tokens=line.strip().split()
        vocab[tokens[1]]=word_ind #word2id
        word_ind+=1
    read_vocab.close()
    #load train file
    def load_file(infile, word2id):   
        read_file=open(infile, 'r')
        data_D=[] #docs, will be a order-3 tensor
#         data_Q=[] #question
        data_A1=[] #positive answers
      #  data_A2=[]
#         data_A3=[]
#         data_A4=[]
#         Y=[]
        Label=[]
        Length_D=[] #true lengths of docs
        Length_D_s=[] # true lengths of each sentences in docs
#         Length_Q=[]
        Length_A1=[]
      #  Length_A2=[]
#         Length_A3=[]
#         Length_A4=[]
        leftPad_D=[]
        leftPad_D_s=[]
#         leftPad_Q=[]
        leftPad_A1=[]
      #  leftPad_A2=[]
#         leftPad_A3=[]
#         leftPad_A4=[]
        rightPad_D=[]
        rightPad_D_s=[]
#         rightPad_Q=[]
        rightPad_A1=[]       
       # rightPad_A2=[]  
#         rightPad_A3=[]  
#         rightPad_A4=[]                           

        line_control=0
        for line in read_file:
            tokens=line.strip().split('\t')  # question, answer, label
#             Y.append(int(tokens[0]))
            if len(tokens) <=4:
                continue
            Label.append(int(tokens[0])) #1 or 2 means single or multiple

            #doc
            data_D_s=[]# will be a matrix
            length_D_s=[]#a vector
            left_D_s=[] # vector
            right_D_s=[] #vector
            
            doc_len=len(tokens)-2 # remove one label, two statements           
            left_D=(max_doc_length-doc_len)/2
            for i in range(left_D):#pad empty sentences
                sent=[0]*maxlength
                #update four depository
                data_D_s.append(sent)
                length_D_s.append(0)
                left_D_s.append(maxlength/2)
                right_D_s.append(maxlength-maxlength/2)


            for s in tokens[1:-1]: #load valid sentences
                s = s.lower()
               # print s
                sent=[]
                words=s.strip().split()  
                for word in words:
                    id=word2id.get(word)
                    if id is not None:
                        sent.append(id)
                    else:
                        sent.append(0)
                length=len(sent)
                left=(maxlength-length)/2
                right=maxlength-left-length
                sent=[0]*left+sent+[0]*right
                #update four depository
                data_D_s.append(sent)
                length_D_s.append(length)
                left_D_s.append(left)
                right_D_s.append(right)
               # print sent
               # raw_input()
            right_D=max_doc_length-left_D-doc_len
            for i in range(right_D):#pad empty sentences
                sent=[0]*maxlength
                #update four depository
                data_D_s.append(sent)
                length_D_s.append(0)
                left_D_s.append(maxlength/2)
                right_D_s.append(maxlength-maxlength/2)            

            data_D.append(data_D_s) # add one slice
            Length_D.append(doc_len)
            leftPad_D.append(left_D)
            rightPad_D.append(right_D)                        
            #store above four depository
            Length_D_s.append(length_D_s)
            leftPad_D_s.append(left_D_s)
            rightPad_D_s.append(right_D_s)
#             #Q
#             words=tokens[-5].strip().split()
#             len_q=len(words)
#             left=(maxlength-len_q)/2
#             right=maxlength-left-len_q          
#             sent=[]
#             for word in words:
#                 id=word2id.get(word)
#                 if id is not None:
#                     sent.append(id)
#             sent=[0]*left+sent+[0]*right
#             data_Q.append(sent)
#             Length_Q.append(len_q)
#             leftPad_Q.append(left)
#             rightPad_Q.append(right)             
            #A1
            words=tokens[-1].strip().split()
            #len_a=len(words)           
            sent=[]
            for word in words:
                id=word2id.get(word)
                if id is not None:
                    sent.append(id)
                else:
                    sent.append(0)
            len_a = len(sent)
            left=(maxlength-len_a)/2
            right=maxlength-left-len_a
            sent=[0]*left+sent+[0]*right
            data_A1.append(sent)            
            Length_A1.append(len_a)
            leftPad_A1.append(left)
            rightPad_A1.append(right)         
            #A2
            '''
            words=tokens[-1].strip().split()
            len_a=len(words)
            left=(maxlength-len_a)/2
            right=maxlength-left-len_a           
            sent=[]
            for word in words:
                id=word2id.get(word)
                if id is not None:
                    sent.append(id)
            sent=[0]*left+sent+[0]*right
            data_A2.append(sent)            
            Length_A2.append(len_a)
            leftPad_A2.append(left)
            rightPad_A2.append(right)  
            '''         
#             #A1
#             words=tokens[-2].strip().split()
#             len_a=len(words)
#             left=(maxlength-len_a)/2
#             right=maxlength-left-len_a           
#             sent=[]
#             for word in words:
#                 id=word2id.get(word)
#                 if id is not None:
#                     sent.append(id)
#             sent=[0]*left+sent+[0]*right
#             data_A3.append(sent)            
#             Length_A3.append(len_a)
#             leftPad_A3.append(left)
#             rightPad_A3.append(right)   
#             #A1
#             words=tokens[-1].strip().split()
#             len_a=len(words)
#             left=(maxlength-len_a)/2
#             right=maxlength-left-len_a           
#             sent=[]
#             for word in words:
#                 id=word2id.get(word)
#                 if id is not None:
#                     sent.append(id)
#             sent=[0]*left+sent+[0]*right
#             data_A4.append(sent)            
#             Length_A4.append(len_a)
#             leftPad_A4.append(left)
#             rightPad_A4.append(right)       
            line_control+=1
            #if line_control==50:
            #    break
        read_file.close()
 
        results=[numpy.array(data_D), numpy.array(data_A1),  numpy.array(Label), 
                 numpy.array(Length_D),numpy.array(Length_D_s),  numpy.array(Length_A1),
                numpy.array(leftPad_D),numpy.array(leftPad_D_s), numpy.array(leftPad_A1),
                numpy.array(rightPad_D),numpy.array(rightPad_D_s),  numpy.array(rightPad_A1)]
        return results, Label, line_control



    train_data, train_Label,train_size=load_file(trainFile, vocab)
    print 'train file loaded over, train size:', train_size

    test_data, test_Label,test_size=load_file(testFile, vocab)
    #print 'test file loaded over, test_size:', test_size
    #print 'Lable', train_data[2]
    #print 'data_d', train_data[0]
    #print 'length_d', train_data[3]
    #raw_input()

    
    def shared_dataset(data_y, borrow=True):
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),  # @UndefinedVariable
                                 borrow=borrow)
        return T.cast(shared_y, 'int64')  
        #return shared_y
   
    
    train_list=[shared_dataset(matt) for matt in train_data]

    test_list=[shared_dataset(matt) for matt in test_data]       

    return train_list, train_Label,train_size, test_list,test_Label, test_size, word_ind-1

def load_MCTest_corpus_DPNQ(vocabFile, trainFile, testFile, max_truncate,maxlength, max_doc_length): #maxSentLength=45
    #first load word vocab
    read_vocab=open(vocabFile, 'r')
    vocab={}
    word_ind=1
    for line in read_vocab:
        tokens=line.strip().split()
        vocab[tokens[1]]=word_ind #word2id
        word_ind+=1
    read_vocab.close()
    #load train file
    def load_file(infile, word2id):   
        read_file=open(infile, 'r')
        data_D=[] #docs, will be a order-3 tensor
#         data_Q=[] #question
        data_A1=[] #positive answers
        data_A2=[]
        data_A3=[]
#         data_A4=[]
#         Y=[]
        Label=[]
        Length_D=[] #true lengths of docs
        Length_D_s=[] # true lengths of each sentences in docs
#         Length_Q=[]
        Length_A1=[]
        Length_A2=[]
        Length_A3=[]
#         Length_A4=[]
        leftPad_D=[]
        leftPad_D_s=[]
#         leftPad_Q=[]
        leftPad_A1=[]
        leftPad_A2=[]
        leftPad_A3=[]
#         leftPad_A4=[]
        rightPad_D=[]
        rightPad_D_s=[]
#         rightPad_Q=[]
        rightPad_A1=[]       
        rightPad_A2=[]  
        rightPad_A3=[]  
#         rightPad_A4=[]                           

        line_control=0
        for line in read_file:
            tokens=line.strip().split('\t')  # question, answer, label
#             Y.append(int(tokens[0]))
            Label.append(int(tokens[0])) #1 or 2 means single or multiple

            #doc
            data_D_s=[]# will be a matrix
            length_D_s=[]#a vector
            left_D_s=[] # vector
            right_D_s=[] #vector
            
            doc_len=len(tokens)-4 # remove one label, two statements, one question           
            left_D=(max_doc_length-doc_len)/2
            for i in range(left_D):#pad empty sentences
                sent=[0]*maxlength
                #update four depository
                data_D_s.append(sent)
                length_D_s.append(0)
                left_D_s.append(maxlength/2)
                right_D_s.append(maxlength-maxlength/2)


            for s in tokens[1:-3]: #load valid sentences
                sent=[]
                words=s.strip().split()  
                for word in words:
                    id=word2id.get(word)
                    if id is not None:
                        sent.append(id)
                length=len(words)
                left=(maxlength-length)/2
                right=maxlength-left-length
                sent=[0]*left+sent+[0]*right
                #update four depository
                data_D_s.append(sent)
                length_D_s.append(length)
                left_D_s.append(left)
                right_D_s.append(right)
            
            right_D=max_doc_length-left_D-doc_len
            for i in range(right_D):#pad empty sentences
                sent=[0]*maxlength
                #update four depository
                data_D_s.append(sent)
                length_D_s.append(0)
                left_D_s.append(maxlength/2)
                right_D_s.append(maxlength-maxlength/2)            

            data_D.append(data_D_s) # add one slice
            Length_D.append(doc_len)
            leftPad_D.append(left_D)
            rightPad_D.append(right_D)                        
            #store above four depository
            Length_D_s.append(length_D_s)
            leftPad_D_s.append(left_D_s)
            rightPad_D_s.append(right_D_s)
#             #Q
#             words=tokens[-5].strip().split()
#             len_q=len(words)
#             left=(maxlength-len_q)/2
#             right=maxlength-left-len_q          
#             sent=[]
#             for word in words:
#                 id=word2id.get(word)
#                 if id is not None:
#                     sent.append(id)
#             sent=[0]*left+sent+[0]*right
#             data_Q.append(sent)
#             Length_Q.append(len_q)
#             leftPad_Q.append(left)
#             rightPad_Q.append(right)             
            #A1
            words=tokens[-3].strip().split()
            len_a=len(words)
            left=(maxlength-len_a)/2
            right=maxlength-left-len_a           
            sent=[]
            for word in words:
                id=word2id.get(word)
                if id is not None:
                    sent.append(id)
            sent=[0]*left+sent+[0]*right
            data_A1.append(sent)            
            Length_A1.append(len_a)
            leftPad_A1.append(left)
            rightPad_A1.append(right)         
            #A2
            words=tokens[-2].strip().split()
            len_a=len(words)
            left=(maxlength-len_a)/2
            right=maxlength-left-len_a           
            sent=[]
            for word in words:
                id=word2id.get(word)
                if id is not None:
                    sent.append(id)
            sent=[0]*left+sent+[0]*right
            data_A2.append(sent)            
            Length_A2.append(len_a)
            leftPad_A2.append(left)
            rightPad_A2.append(right)   
            #A3
            words=tokens[-1].strip().split()
            len_a=len(words)
            left=(maxlength-len_a)/2
            right=maxlength-left-len_a           
            sent=[]
            for word in words:
                id=word2id.get(word)
                if id is not None:
                    sent.append(id)
            sent=[0]*left+sent+[0]*right
            data_A3.append(sent)            
            Length_A3.append(len_a)
            leftPad_A3.append(left)
            rightPad_A3.append(right)   
#             #A1
#             words=tokens[-1].strip().split()
#             len_a=len(words)
#             left=(maxlength-len_a)/2
#             right=maxlength-left-len_a           
#             sent=[]
#             for word in words:
#                 id=word2id.get(word)
#                 if id is not None:
#                     sent.append(id)
#             sent=[0]*left+sent+[0]*right
#             data_A4.append(sent)            
#             Length_A4.append(len_a)
#             leftPad_A4.append(left)
#             rightPad_A4.append(right)       
            line_control+=1
            #if line_control==50:
            #    break
        read_file.close()
 
        
        results=[numpy.array(data_D), numpy.array(data_A1), numpy.array(data_A2), numpy.array(data_A3), numpy.array(Label), 
                 numpy.array(Length_D),numpy.array(Length_D_s), numpy.array(Length_A1), numpy.array(Length_A2), numpy.array(Length_A3),
                numpy.array(leftPad_D),numpy.array(leftPad_D_s), numpy.array(leftPad_A1), numpy.array(leftPad_A2), numpy.array(leftPad_A3),
                numpy.array(rightPad_D),numpy.array(rightPad_D_s), numpy.array(rightPad_A1), numpy.array(rightPad_A2), numpy.array(rightPad_A3)]
        return results, line_control



    train_data, train_size=load_file(trainFile, vocab)
    print 'train file loaded over, train size:', train_size

    test_data, test_size=load_file(testFile, vocab)
    print 'test file loaded over, test_size:', test_size


    
    def shared_dataset(data_y, borrow=True):
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),  # @UndefinedVariable
                                 borrow=borrow)
        return T.cast(shared_y, 'int64')  
        #return shared_y
    
    train_list=[shared_dataset(matt) for matt in train_data]

    test_list=[shared_dataset(matt) for matt in test_data]       

    return train_list, train_size, test_list, test_size, word_ind-1