from __future__ import print_function
import numpy as np
import tensorflow as tf

import argparse
import codecs
import time
import os
from six.moves import cPickle

from utils import TextLoader
from model import Model

from six import text_type

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='save',
                       help='model directory to store checkpointed models')
    parser.add_argument('--text', type=str, default='data/tinyshakespeare/input.txt',
                        help='filename of text to evaluate on')
    args = parser.parse_args()
    eval(args)

def eval(args):
    with open(os.path.join(args.save_dir, 'config.pkl'), 'rb') as f:
        saved_args = cPickle.load(f)
    saved_args.batch_size = 1
    saved_args.seq_length = 200
    with open(os.path.join(args.save_dir, 'chars_vocab.pkl'), 'rb') as f:
        chars, vocab = cPickle.load(f)
    model = Model(saved_args, False)

    with codecs.open(args.text, 'r', encoding='utf-8') as f:
        text = f.read()

    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        saver = tf.train.Saver(tf.all_variables())
        ckpt = tf.train.get_checkpoint_state(args.save_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            ppl = model.eval(sess, chars, vocab, text)
            print('perplexity: {0}'.format(ppl))

if __name__ == '__main__':
    main()