#!/usr/bin/env python
# coding=utf-8
# Copyright 2018 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import itertools
import os
import time

import numpy as np
import numpy
import tensorflow as tf
import thumt.data.record as record
import thumt.data.dataset as dataset
import thumt.data.vocab as vocabulary
import thumt.models as models
import thumt.utils.search as search

def normalize(matrix):
    matrix = numpy.abs(matrix)
    total = numpy.sum(matrix, -1)
    return matrix / numpy.expand_dims(total, -1)   

def get_rlv_encoder_vector(w_x_newh, w_h_newh, enc_h, back=False):
    '''
        w_h_newh: [len_src, 1]
        w_x_newh: [len_src, 1]
        enc_h: [1, len_src, dim]
    '''
    len_src = w_x_newh.shape[0]
    r = numpy.zeros((len_src, len_src), dtype = 'float32') 
    print(r.shape)
    R = numpy.zeros((len_src, len_src), dtype="float32")
    for i in range(len_src):
        for j in range(i+1):
            if i == j:
                r[i,j] = w_x_newh[i][0]
            else:
                if i == 0:
                    tmp = 0
                else:
                    tmp = r[i-1,j]
                # relevance h_i, x_j
                r[i][j] = tmp * w_h_newh[i][0]
    if back:
        r = r[::-1, ::-1]
    for i in range(len_src): # hidden
        for j in range(len_src): # src
            R[i][j] = r[i][j] * numpy.sum(enc_h[0,i]) 
    # print out 
    R = normalize(R)
    print('encoder:', R)
    return r, R

def to_text(vocab, mapping, indice, params):
    print('idce', indice)
    decoded = []
    for idx in indice:
        if idx == mapping[params.eos]:
            break
        decoded.append(vocab[idx])

    decoded = " ".join(decoded)
    return decoded

def parse_args():
    parser = argparse.ArgumentParser(
        description="Translate using existing NMT models",
        usage="translator.py [<args>] [-h | --help]"
    )

    # input files
    parser.add_argument("--input", type=str, nargs=2, required=True,
                        help="Path of input file")
    parser.add_argument("--checkpoints", type=str, nargs="+", required=True,
                        help="Path of trained models")
    parser.add_argument("--vocabulary", type=str, nargs=2, required=True,
                        help="Path of source and target vocabulary")

    # model and configuration
    parser.add_argument("--models", type=str, required=True, nargs="+",
                        help="Name of the model")
    parser.add_argument("--parameters", type=str,
                        help="Additional hyper parameters")

    return parser.parse_args()


def default_parameters():
    params = tf.contrib.training.HParams(
        input=None,
        output=None,
        vocabulary=None,
        model=None,

        batch_size=1,
        max_length=25,
        length_multiplier=1,
        mantissa_bits=2,
        buffer_size=10000,
        constant_batch_size=True,
        # vocabulary specific
        pad="<pad>",
        bos="<bos>",
        eos="<eos>",
        unk="<unk>",
        mapping=None,
        append_eos=False,
        # decoding
        top_beams=1,
        beam_size=4,
        decode_alpha=0.6,
        decode_length=50,
        decode_batch_size=32,
        decode_constant=5.0,
        decode_normalize=False,
        device_list=[0],
        num_threads=6
    )

    return params


def merge_parameters(params1, params2):
    params = tf.contrib.training.HParams()

    for (k, v) in params1.values().iteritems():
        params.add_hparam(k, v)

    params_dict = params.values()

    for (k, v) in params2.values().iteritems():
        if k in params_dict:
            # Override
            setattr(params, k, v)
        else:
            params.add_hparam(k, v)

    return params


def import_params(model_dir, model_name, params):
    model_dir = os.path.abspath(model_dir)
    m_name = os.path.join(model_dir, model_name + ".json")

    if not tf.gfile.Exists(m_name):
        return params

    with tf.gfile.Open(m_name) as fd:
        tf.logging.info("Restoring model parameters from %s" % m_name)
        json_str = fd.readline()
        params.parse_json(json_str)

    return params


def override_parameters(params, args):
    if args.parameters:
        params.parse(args.parameters)

    params.vocabulary = {
        "source": vocabulary.load_vocabulary(args.vocabulary[0]),
        "target": vocabulary.load_vocabulary(args.vocabulary[1])
    }
    params.vocabulary["source"] = vocabulary.process_vocabulary(
        params.vocabulary["source"], params
    )
    params.vocabulary["target"] = vocabulary.process_vocabulary(
        params.vocabulary["target"], params
    )

    control_symbols = [params.pad, params.bos, params.eos, params.unk]

    params.mapping = {
        "source": vocabulary.get_control_mapping(
            params.vocabulary["source"],
            control_symbols
        ),
        "target": vocabulary.get_control_mapping(
            params.vocabulary["target"],
            control_symbols
        )
    }

    return params


def session_config(params):
    optimizer_options = tf.OptimizerOptions(opt_level=tf.OptimizerOptions.L1,
                                            do_function_inlining=False)
    graph_options = tf.GraphOptions(optimizer_options=optimizer_options)
    config = tf.ConfigProto(allow_soft_placement=True,
                            graph_options=graph_options)
    if params.device_list:
        device_str = ",".join([str(i) for i in params.device_list])
        config.gpu_options.visible_device_list = device_str

    return config


def set_variables(var_list, value_dict, prefix):
    ops = []
    for var in var_list:
        for name in value_dict:
            var_name = "/".join([prefix] + list(name.split("/")[1:]))

            if var.name[:-2] == var_name:
                tf.logging.info("restoring %s -> %s" % (name, var.name))
                with tf.device("/cpu:0"):
                    op = tf.assign(var, value_dict[name])
                    ops.append(op)
                break

    return ops


def main(args):
    tf.logging.set_verbosity(tf.logging.INFO)
    # Load configs
    model_cls_list = [models.get_model(model) for model in args.models]
    params_list = [default_parameters() for _ in range(len(model_cls_list))]
    params_list = [
        merge_parameters(params, model_cls.get_parameters())
        for params, model_cls in zip(params_list, model_cls_list)
    ]
    params_list = [
        import_params(args.checkpoints[i], args.models[i], params_list[i])
        for i in range(len(args.checkpoints))
    ]
    params_list = [
        override_parameters(params_list[i], args)
        for i in range(len(model_cls_list))
    ]

    # Build Graph
    with tf.Graph().as_default():
        model_var_lists = []

        # Load checkpoints
        for i, checkpoint in enumerate(args.checkpoints):
            print("Loading %s" % checkpoint)
            var_list = tf.train.list_variables(checkpoint)
            values = {}
            reader = tf.train.load_checkpoint(checkpoint)

            for (name, shape) in var_list:
                if not name.startswith(model_cls_list[i].get_name()):
                    continue

                if name.find("losses_avg") >= 0:
                    continue

                tensor = reader.get_tensor(name)
                values[name] = tensor

            model_var_lists.append(values)

        # Build models
        model_fns = []

        for i in range(len(args.checkpoints)):
            name = model_cls_list[i].get_name()
            model = model_cls_list[i](params_list[i], name + "_%d" % i)
            model_fn = model.get_relevance_func()
            model_fns.append(model_fn)

        params = params_list[0]
        # Build input queue
        features = dataset.get_training_input(args.input, params)
        relevances = model_fns[0](features, params)

        assign_ops = []

        all_var_list = tf.trainable_variables()

        for i in range(len(args.checkpoints)):
            un_init_var_list = []
            name = model_cls_list[i].get_name()

            for v in all_var_list:
                if v.name.startswith(name + "_%d" % i):
                    un_init_var_list.append(v)

            ops = set_variables(un_init_var_list, model_var_lists[i],
                                name + "_%d" % i)
            assign_ops.extend(ops)

        assign_op = tf.group(*assign_ops)

        sess_creator = tf.train.ChiefSessionCreator(
            config=session_config(params)
        )



        results = []
        num = 10
        count = 0
        hooks = [tf.train.LoggingTensorHook({}, every_n_iter=1)]
        with tf.train.MonitoredSession(session_creator=sess_creator, hooks=hooks) as sess:
            # Restore variables
            sess.run(assign_op)
            src_seq, trg_seq, rlv_info, loss = sess.run(relevances)
            start = time.time()
            while count < num:#not sess.should_stop():
                src_seq, trg_seq, rlv_info, loss = sess.run(relevances)
                print('--result--')
                print('loss:', loss)
                for i in range(src_seq.shape[0]):
                    src = to_text(params.vocabulary["source"], params.mapping["source"], src_seq[i], params)
                    trg = to_text(params.vocabulary["target"], params.mapping["target"], trg_seq[i], params)
                    print('sentence %d' %i)
                    print('src:', src)
                    print('src_idx:', src_seq[i])
                    print('trg:', trg)
                    print('trg_idx:', trg_seq[i])
                    print('result:',rlv_info["result"][i])
                count += 1
            end = time.time()
            print('total time:',end-start)

if __name__ == "__main__":
    main(parse_args())
