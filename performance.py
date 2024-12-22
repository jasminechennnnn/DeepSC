# !usr/bin/env python
# -*- coding:utf-8 _*-
"""
@Author: Huiqiang Xie
@File: performance.py
@Time: 2021/4/1 11:48
"""
import os
import json
import torch
import argparse
import numpy as np
from dataset import EurDataset, collate_data
from models.transceiver import DeepSC
from torch.utils.data import DataLoader
from utils import BleuScore, SNR_to_noise, greedy_decode, SeqtoText
from tqdm import tqdm
from sklearn.preprocessing import normalize
# from bert4keras.backend import keras
# from bert4keras.models import build_bert_model
# from bert4keras.tokenizers import Tokenizer
from w3lib.html import remove_tags
from datetime import datetime, timedelta, timezone
import pytz

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', default='data/train_data.pkl', type=str)
parser.add_argument('--vocab-file', default='data/vocab.json', type=str)
parser.add_argument('--checkpoint-path', default='checkpoints/', type=str)
parser.add_argument('--channel', default='Rayleigh', type=str)
parser.add_argument('--MAX-LENGTH', default=30, type=int)
parser.add_argument('--MIN-LENGTH', default=4, type=int)
parser.add_argument('--d-model', default=128, type = int)
parser.add_argument('--dff', default=512, type=int)
parser.add_argument('--num-layers', default=4, type=int)
parser.add_argument('--num-heads', default=8, type=int)
parser.add_argument('--batch-size', default=64, type=int)
parser.add_argument('--epochs', default=2, type = int)
parser.add_argument('--bert-config-path', default='bert/cased_L-12_H-768_A-12/bert_config.json', type = str)
parser.add_argument('--bert-checkpoint-path', default='bert/cased_L-12_H-768_A-12/bert_model.ckpt', type = str)
parser.add_argument('--bert-dict-path', default='bert/cased_L-12_H-768_A-12/vocab.txt', type = str)
parser.add_argument('--lamb', default=0.0009, type=float, help='weight for MI loss')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# using pre-trained model to compute the sentence similarity
# class Similarity():
#     def __init__(self, config_path, checkpoint_path, dict_path):
#         self.model1 = build_bert_model(config_path, checkpoint_path, with_pool=True)
#         self.model = keras.Model(inputs=self.model1.input,
#                                  outputs=self.model1.get_layer('Encoder-11-FeedForward-Norm').output)
#         # build tokenizer
#         self.tokenizer = Tokenizer(dict_path, do_lower_case=True)
#
#     def compute_similarity(self, real, predicted):
#         token_ids1, segment_ids1 = [], []
#         token_ids2, segment_ids2 = [], []
#         score = []
#
#         for (sent1, sent2) in zip(real, predicted):
#             sent1 = remove_tags(sent1)
#             sent2 = remove_tags(sent2)
#
#             ids1, sids1 = self.tokenizer.encode(sent1)
#             ids2, sids2 = self.tokenizer.encode(sent2)
#
#             token_ids1.append(ids1)
#             token_ids2.append(ids2)
#             segment_ids1.append(sids1)
#             segment_ids2.append(sids2)
#
#         token_ids1 = keras.preprocessing.sequence.pad_sequences(token_ids1, maxlen=32, padding='post')
#         token_ids2 = keras.preprocessing.sequence.pad_sequences(token_ids2, maxlen=32, padding='post')
#
#         segment_ids1 = keras.preprocessing.sequence.pad_sequences(segment_ids1, maxlen=32, padding='post')
#         segment_ids2 = keras.preprocessing.sequence.pad_sequences(segment_ids2, maxlen=32, padding='post')
#
#         vector1 = self.model.predict([token_ids1, segment_ids1])
#         vector2 = self.model.predict([token_ids2, segment_ids2])
#
#         vector1 = np.sum(vector1, axis=1)
#         vector2 = np.sum(vector2, axis=1)
#
#         vector1 = normalize(vector1, axis=0, norm='max')
#         vector2 = normalize(vector2, axis=0, norm='max')
#
#         dot = np.diag(np.matmul(vector1, vector2.T))  # a*b
#         a = np.diag(np.matmul(vector1, vector1.T))  # a*a
#         b = np.diag(np.matmul(vector2, vector2.T))
#
#         a = np.sqrt(a)
#         b = np.sqrt(b)
#
#         output = dot / (a * b)
#         score = output.tolist()
#
#         return score


def performance(args, SNR, net):
    # similarity = Similarity(args.bert_config_path, args.bert_checkpoint_path, args.bert_dict_path)
    bleu_scores = {
        '1gram': BleuScore(1, 0, 0, 0),
        '2gram': BleuScore(0, 1, 0, 0),
        '3gram': BleuScore(0, 0, 1, 0),
        '4gram': BleuScore(0, 0, 0, 1),
        'avg': BleuScore(0.25, 0.25, 0.25, 0.25)
    }

    test_eur = EurDataset('test')
    test_iterator = DataLoader(test_eur, batch_size=args.batch_size, num_workers=0,
                               pin_memory=True, collate_fn=collate_data)

    StoT = SeqtoText(token_to_idx, end_idx)
    scores = {key: [] for key in bleu_scores.keys()}
    # score2 = []
    net.eval()
    with torch.no_grad():
        for _ in range(args.epochs):
            Tx_word = []
            Rx_word = []

            for snr in tqdm(SNR):
                word = []
                target_word = []
                noise_std = SNR_to_noise(snr)

                for sents in test_iterator:

                    sents = sents.to(device)
                    # src = batch.src.transpose(0, 1)[:1]
                    target = sents

                    out = greedy_decode(net, sents, noise_std, args.MAX_LENGTH, pad_idx,
                                        start_idx, args.channel)

                    sentences = out.cpu().numpy().tolist()
                    result_string = list(map(StoT.sequence_to_text, sentences))
                    word = word + result_string

                    target_sent = target.cpu().numpy().tolist()
                    result_string = list(map(StoT.sequence_to_text, target_sent))
                    target_word = target_word + result_string

                Tx_word.append(word)
                Rx_word.append(target_word)
            
            for gram_type, bleu_scorer in bleu_scores.items():
                epoch_scores = []
                for sent1, sent2 in zip(Tx_word, Rx_word):
                    epoch_scores.append(bleu_scorer.compute_blue_score(sent1, sent2))
                epoch_scores = np.array(epoch_scores)
                epoch_mean = np.mean(epoch_scores, axis=1)
                scores[gram_type].append(epoch_mean)
            
            # sim_score = []
            # for sent1, sent2 in zip(Tx_word, Rx_word):
                # sim_score.append(similarity.compute_similarity(sent1, sent2)) # 7*num_sent

            # sim_score = np.array(sim_score)
            # sim_score = np.mean(sim_score, axis=1)
            # score2.append(sim_score)

    # score2 = np.mean(np.array(score2), axis=0)
    final_scores = {gram_type: np.mean(np.array(score_list), axis=0) 
                   for gram_type, score_list in scores.items()}

    eval_dir = os.path.join(os.path.dirname(model_path), 'evaluation')
    os.makedirs(eval_dir, exist_ok=True)

    current_time = datetime.now(pytz.timezone('Asia/Taipei'))
    result_file = os.path.join(eval_dir, f'{current_time.strftime("%Y%m%d_%H%M%S")}_bleu_scores.txt')

    with open(result_file, 'w', encoding='utf-8') as f:
        f.write(f'Evaluation Time: {current_time.strftime("%Y-%m-%d %H:%M:%S")}\n')
        f.write(f'Model Path: {model_path}\n')
        f.write(f'Channel: {args.channel}\n')
        f.write(f'Lambda: {args.lamb}\n')
        f.write('-' * 50 + '\n')
        f.write(f'SNR values: {SNR}\n')
        f.write('-' * 50 + '\n')
        f.write('BLEU Scores:\n')
        for gram_type, score in final_scores.items():
            f.write(f'{gram_type}: {score}\n')

    return final_scores

if __name__ == '__main__':
    args = parser.parse_args()
    SNR = [0,3,6,9,12,15,18]

    args.vocab_file = '' + args.vocab_file
    vocab = json.load(open(args.vocab_file, 'rb'))
    token_to_idx = vocab['token_to_idx']
    idx_to_token = dict(zip(token_to_idx.values(), token_to_idx.keys()))
    num_vocab = len(token_to_idx)
    pad_idx = token_to_idx["<PAD>"]
    start_idx = token_to_idx["<START>"]
    end_idx = token_to_idx["<END>"]

    """ define optimizer and loss function """
    deepsc = DeepSC(args.num_layers, num_vocab, num_vocab,
                        num_vocab, num_vocab, args.d_model, args.num_heads,
                        args.dff, 0.1).to(device)

    # (default) find the latest checkpoint
    if args.checkpoint_path == "checkpoints/":
        subdirs = [d for d in os.listdir(args.checkpoint_path) if os.path.isdir(os.path.join(args.checkpoint_path, d))]
    
        if not subdirs:
            raise FileNotFoundError("No checkpoint directories found in checkpoints/")
        
        latest_dir = sorted(subdirs)[-1]
        model_path = os.path.join(args.checkpoint_path, latest_dir, 'best.pth')
        print("Loading the latest checkpoint from: ", model_path)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No best.pth found in {os.path.join(args.checkpoint_path, latest_dir)}")
    # specific checkpoint  
    else:
        model_path = os.path.join(args.checkpoint_path, 'best.pth')
        print(f"Loading checkpoint from: {model_path}") 
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No best.pth found in {args.checkpoint_path}")

    checkpoint = torch.load(model_path, weights_only=True)
    deepsc.load_state_dict(checkpoint)
    
    print('Start evaluating the model...')
    bleu_score = performance(args, SNR, deepsc)
    print(bleu_score)
    #similarity.compute_similarity(sent1, real)
