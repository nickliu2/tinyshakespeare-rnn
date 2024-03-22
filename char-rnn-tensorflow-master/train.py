#!/usr/bin/env python

from __future__ import print_function

import argparse
import time
import os
from six.moves import cPickle
import csv
import numpy as np


parser = argparse.ArgumentParser(
                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# Data and model checkpoints directories
parser.add_argument('--data_dir', type=str, default='data/tinyshakespeare',
                    help='data directory containing input.txt with training examples')
parser.add_argument('--save_dir', type=str, default='save',
                    help='directory to store checkpointed models')
parser.add_argument('--log_dir', type=str, default='logs',
                    help='directory to store tensorboard logs')
parser.add_argument('--save_every', type=int, default=1000,
                    help='Save frequency. Number of passes between checkpoints of the model.')
parser.add_argument('--init_from', type=str, default=None,
                    help="""continue training from saved model at this path (usually "save").
                        Path must contain files saved by previous training process:
                        'config.pkl'        : configuration;
                        'chars_vocab.pkl'   : vocabulary definitions;
                        'checkpoint'        : paths to model file(s) (created by tf).
                                              Note: this file contains absolute paths, be careful when moving files around;
                        'model.ckpt-*'      : file(s) with model definition (created by tf)
                         Model params must be the same between multiple runs (model, rnn_size, num_layers and seq_length).
                    """)
# Model params
parser.add_argument('--model', type=str, default='lstm',
                    help='lstm, rnn, gru, or nas')
parser.add_argument('--rnn_size', type=int, default=128,
                    help='size of RNN hidden state')
parser.add_argument('--num_layers', type=int, default=2,
                    help='number of layers in the RNN')
# Optimization
parser.add_argument('--seq_length', type=int, default=50,
                    help='RNN sequence length. Number of timesteps to unroll for.')
parser.add_argument('--batch_size', type=int, default=50,
                    help="""minibatch size. Number of sequences propagated through the network in parallel.
                            Pick batch-sizes to fully leverage the GPU (e.g. until the memory is filled up)
                            commonly in the range 10-500.""")
parser.add_argument('--num_epochs', type=int, default=10,
                    help='number of epochs. Number of full passes through the training examples.')
parser.add_argument('--grad_clip', type=float, default=5.,
                    help='clip gradients at this value')
parser.add_argument('--learning_rate', type=float, default=0.002,
                    help='learning rate')
parser.add_argument('--decay_rate', type=float, default=0.97,
                    help='decay rate for rmsprop')
parser.add_argument('--output_keep_prob', type=float, default=1.0,
                    help='probability of keeping weights in the hidden layer')
parser.add_argument('--input_keep_prob', type=float, default=1.0,
                    help='probability of keeping weights in the input layer')
parser.add_argument('--logname', type=str, default=time.strftime("%Y-%m-%d-%H-%M-%S"),
                    help='name of directory for recorded log')
args = parser.parse_args()

import tensorflow as tf
from utils import TextLoader
from model import Model

def evaluate_model(sess, model, data_loader):
    total_loss = 0.0
    data_loader.reset_batch_pointer()

    for b in range(data_loader.num_batches):
        x, y = data_loader.next_batch()
        # Ensure the state is initialized with the correct batch size for evaluation
        # Adjust this line to match the evaluation batch size, which seems to be 1 in this case
        state = sess.run(model.cell.zero_state(data_loader.batch_size, tf.float32))  # Use the actual batch size here

        feed = {model.input_data: x, model.targets: y, model.initial_state: state}
        loss, state = sess.run([model.cost, model.final_state], feed)
        total_loss += loss

    average_loss = total_loss / data_loader.num_batches
    average_perplexity = np.exp(average_loss)
    return average_loss, average_perplexity

def train(args):
    data_loader = TextLoader("data/tinyshakespeare_train", args.batch_size, args.seq_length)
    val_data_loader = TextLoader("data/tinyshakespeare_val", args.batch_size, args.seq_length)
    test_data_loader = TextLoader("data/tinyshakespeare_test", args.batch_size, args.seq_length)
    args.vocab_size = data_loader.vocab_size

    # check compatibility if training is continued from previously saved model
    if args.init_from is not None:
        # check if all necessary files exist
        assert os.path.isdir(args.init_from)," %s must be a a path" % args.init_from
        assert os.path.isfile(os.path.join(args.init_from,"config.pkl")),"config.pkl file does not exist in path %s"%args.init_from
        assert os.path.isfile(os.path.join(args.init_from,"chars_vocab.pkl")),"chars_vocab.pkl.pkl file does not exist in path %s" % args.init_from
        ckpt = tf.train.latest_checkpoint(args.init_from)
        assert ckpt, "No checkpoint found"

        # open old config and check if models are compatible
        with open(os.path.join(args.init_from, 'config.pkl'), 'rb') as f:
            saved_model_args = cPickle.load(f)
        need_be_same = ["model", "rnn_size", "num_layers", "seq_length"]
        for checkme in need_be_same:
            assert vars(saved_model_args)[checkme]==vars(args)[checkme],"Command line argument and saved model disagree on '%s' "%checkme

        # open saved vocab/dict and check if vocabs/dicts are compatible
        with open(os.path.join(args.init_from, 'chars_vocab.pkl'), 'rb') as f:
            saved_chars, saved_vocab = cPickle.load(f)
        assert saved_chars==data_loader.chars, "Data and loaded model disagree on character set!"
        assert saved_vocab==data_loader.vocab, "Data and loaded model disagree on dictionary mappings!"

    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)
    with open(os.path.join(args.save_dir, 'config.pkl'), 'wb') as f:
        cPickle.dump(args, f)
    with open(os.path.join(args.save_dir, 'chars_vocab.pkl'), 'wb') as f:
        cPickle.dump((data_loader.chars, data_loader.vocab), f)

    model = Model(args)

    with tf.Session() as sess:
        # instrument for tensorboard
        summaries = tf.summary.merge_all()
        writer = tf.summary.FileWriter(
                os.path.join(args.log_dir, args.logname))
        writer.add_graph(sess.graph)

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables())
        # restore model
        if args.init_from is not None:
            saver.restore(sess, ckpt)
        total_perplexity = 0
        for e in range(args.num_epochs):
            sess.run(tf.assign(model.lr,
                               args.learning_rate * (args.decay_rate ** e)))
            data_loader.reset_batch_pointer()
            state = sess.run(model.initial_state)
            for b in range(data_loader.num_batches):
                start = time.time()
                x, y = data_loader.next_batch()
                feed = {model.input_data: x, model.targets: y}
                if isinstance(model.cell.state_size, tuple):
                    for i, layer_state_size in enumerate(model.cell.state_size):
                        if isinstance(layer_state_size, tf.contrib.rnn.LSTMStateTuple):
                            # LSTM cell in a layer of MultiRNNCell
                            feed[model.initial_state[i][0]] = state[i].c  # c state
                            feed[model.initial_state[i][1]] = state[i].h  # h state
                        else:
                            # GRU or basic RNN cell in a layer of MultiRNNCell
                            feed[model.initial_state[i]] = state[i]
                else:
                    # Single layer GRU or RNN (not in MultiRNNCell, unlikely based on your description)
                    feed[model.initial_state] = state

                # instrument for tensorboard
                summ, train_loss, state, _, batch_perplexity = sess.run([summaries, model.cost, model.final_state, model.train_op, model.perplexity], feed)
                writer.add_summary(summ, e * data_loader.num_batches + b)
                total_perplexity += batch_perplexity
                train_perplexity = total_perplexity / (e * data_loader.num_batches + b)

                
                end = time.time()
                print("{}/{} (epoch {}), train_loss = {:.3f}, train_perplexity = {:.3f}, time/batch = {:.3f}"
                  .format(e * data_loader.num_batches + b,
                          args.num_epochs * data_loader.num_batches,
                          e, train_loss, train_perplexity, end - start))
                if (e * data_loader.num_batches + b) % args.save_every == 0\
                        or (e == args.num_epochs-1 and
                            b == data_loader.num_batches-1):
                    # save for the last result
                    checkpoint_path = os.path.join(args.save_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path,
                               global_step=e * data_loader.num_batches + b)
                    
                    # save training perplexity
                    print("model saved to {}, average train perplexity:{}".format(checkpoint_path, train_perplexity))

                    ## save val perplexity
                    val_loss, val_perplexity = evaluate_model(sess, model, val_data_loader)
                    print("Epoch: {}, Val Loss: {:.3f}, Val Perplexity: {:.3f}".format(e, val_loss, val_perplexity))

                    # save testing perplexity
                    test_loss, test_perplexity = evaluate_model(sess, model, test_data_loader)
                    print("Epoch: {}, Test Loss: {:.3f}, Test Perplexity: {:.3f}".format(e, test_loss, test_perplexity))
                    
                    # Optionally, log validation metrics to TensorBoard or another logging system
                    val_summary = tf.Summary(value=[
                        tf.Summary.Value(tag="val_loss", simple_value=val_loss),
                        tf.Summary.Value(tag="val_perplexity", simple_value=val_perplexity)])
                    writer.add_summary(val_summary, e * data_loader.num_batches + b)


                    test_summary = tf.Summary(value=[
                        tf.Summary.Value(tag="test_loss", simple_value=test_loss),
                        tf.Summary.Value(tag="test_perplexity", simple_value=test_perplexity)])
                    writer.add_summary(test_summary, e * data_loader.num_batches + b)
                    # Define the path to your CSV file
                    csv_file_path = 'summary.csv'
                    
                    # Check if the CSV file exists and if we need to write headers
                    write_headers = not os.path.isfile(csv_file_path)
                    
                    with open(csv_file_path, 'a', newline='') as csvfile:
                        fieldnames = ['e','rnn_size', 'num_layers', 'seq_length', 'train_perplexity', 'val_perplexity', 'test_perplexity']
                        csvwriter = csv.DictWriter(csvfile, fieldnames=fieldnames)
                            # Write the header only if the file was newly created
                        if write_headers:
                            csvwriter.writeheader()
                            
                        # Write the data
                        csvwriter.writerow({
                            'e': e,
                            'rnn_size': args.rnn_size,
                            'num_layers': args.num_layers,
                            'seq_length': args.seq_length,
                            'train_perplexity': train_perplexity,
                            'val_perplexity': val_perplexity,
                            'test_perplexity': test_perplexity
                        })

if __name__ == '__main__':
    train(args)
