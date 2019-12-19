import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.cm as cm
import skimage.transform

from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms
from PIL import Image

import argparse

import models
import dataset as dset

from tqdm import tqdm

import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def beam_search(encoder, decoder, img, word_map, beam_size=3):
    """
    Reads an image and captions it with beam search.
    :param encoder: encoder model
    :param decoder: decoder model
    :param image_path: path to image
    :param word_map: word map
    :param beam_size: number of sequences to consider at each decode-step
    :return: caption, weights for visualization
    """

    k = beam_size
    vocab_size = len(word_map)

    # Read image and process
    image = transforms.ToTensor()(img).to(device)

    # Encode
    image = image.unsqueeze(0)  # (1, 1, 64, 64)
    encoder_out = encoder(image)  # (1, enc_image_size, enc_image_size, encoder_dim)
    enc_image_size = encoder_out.size(1)
    encoder_dim = encoder_out.size(3)

    # Flatten encoding
    encoder_out = encoder_out.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
    num_pixels = encoder_out.size(1)

    # We'll treat the problem as having a batch size of k
    encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)  # (k, num_pixels, encoder_dim)

    # Tensor to store top k previous words at each step; now they're just <start>
    k_prev_words = torch.zeros(k, 1, dtype=torch.long, device=device)  # (k, 1)

    # Tensor to store top k sequences; now they're just <start>
    seqs = k_prev_words  # (k, 1)

    # Tensor to store top k sequences' scores; now they're just 0
    top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

    # Tensor to store top k sequences' alphas; now they're just 1s
    seqs_alpha = torch.ones(k, 1, enc_image_size, enc_image_size).to(device)  # (k, 1, enc_image_size, enc_image_size)

    # Lists to store completed sequences, their alphas and scores
    complete_seqs = list()
    complete_seqs_alpha = list()
    complete_seqs_scores = list()

    # Start decoding
    step = 1
    h, c = decoder.init_hidden_state(encoder_out)

    # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
    for step in range(10):

        embeddings = decoder.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)

        awe, alpha = decoder.attention(encoder_out, h)  # (s, encoder_dim), (s, num_pixels)

        alpha = alpha.view(-1, enc_image_size, enc_image_size)  # (s, enc_image_size, enc_image_size)

        gate = decoder.sigmoid(decoder.f_beta(h))  # gating scalar, (s, encoder_dim)
        awe = gate * awe

        h, c = decoder.rnn(torch.cat([embeddings, awe], dim=1), (h, c))  # (s, decoder_dim)

        scores = decoder.fc(h)  # (s, vocab_size)
        scores = F.log_softmax(scores, dim=1)

        # Add
        scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

        # For the first step, all k points will have the same scores (since same k previous words, h, c)
        if step == 1:
            top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
        else:
            # Unroll and find top scores, and their unrolled indices
            top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

        # Convert unrolled indices to actual indices of scores
        prev_word_inds = top_k_words / vocab_size  # (s)
        next_word_inds = top_k_words % vocab_size  # (s)

        # Add new words to sequences, alphas
        seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)
        seqs_alpha = torch.cat([seqs_alpha[prev_word_inds], alpha[prev_word_inds].unsqueeze(1)],
                               dim=1)  # (s, step+1, enc_image_size, enc_image_size)

        # Which sequences are incomplete (didn't reach <end>)?
        incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                           next_word != word_map['<end>']]
        complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

        # Set aside complete sequences
        if len(complete_inds) > 0:
            complete_seqs.extend(seqs[complete_inds].tolist())
            complete_seqs_alpha.extend(seqs_alpha[complete_inds].tolist())
            complete_seqs_scores.extend(top_k_scores[complete_inds])
        k -= len(complete_inds)  # reduce beam length accordingly

        # Proceed with incomplete sequences
        if k == 0:
            break
        seqs = seqs[incomplete_inds]
        seqs_alpha = seqs_alpha[incomplete_inds]
        h = h[prev_word_inds[incomplete_inds]]
        c = c[prev_word_inds[incomplete_inds]]
        encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
        top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
        k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)


    i = complete_seqs_scores.index(max(complete_seqs_scores))
    seq = complete_seqs[i]
    alphas = complete_seqs_alpha[i]

    return seq, alphas, complete_seqs




def main():
    global device
    parser = argparse.ArgumentParser(description='Neuro Cangjie')

    parser.add_argument('--model', '-m', help='path to model', default='logs/cangjie5.pth.tar')
    parser.add_argument('--fonts', '-f', nargs='+', default=['data/hanazono/HanaMinA.ttf', 'data/hanazono/HanaMinB.ttf'])
    parser.add_argument('--codemap', '-cm', help='path to code map', default='data/codemap_cangjie5.txt')
    parser.add_argument('--beam_size', '-b', default=5, type=int, help='beam size for beam search')
    parser.add_argument('--dont_smooth', dest='smooth', action='store_false', help='do not smooth alpha overlay')
    parser.add_argument('--use_cpu', action='store_true', help='use cpu for model inference')

    args = parser.parse_args()

    if args.use_cpu:
        device = torch.device('cpu')

    # Load model
    checkpoint = torch.load(args.model, map_location=device)
    encoder = models.Encoder(encode_channels=256).to(device)
    decoder = models.Decoder(128, 256, 256, 26 + 2, dataset.char_num, encoder_dim=256, dropout=0.5).to(device)
    decoder.load_state_dict(checkpoint['decoder'])
    encoder.load_state_dict(checkpoint['encoder'])
    decoder.eval()
    encoder.eval()

    word_map, rev_word_map = utils.load_map(args.codemap)

    glyph = dset.Glyph(args.fonts)
    while True:
        ch = input('>> ')[0]
        img = glyph.draw(ch)
        img.save('exp.png')

        # Encode, decode with attention and beam search
        seq, alphas, all_seq = beam_search(encoder, decoder, img, word_map, args.beam_size)
        for s in all_seq:
            codes = [rev_word_map[ind] for ind in s]
            print(codes)
        alphas = torch.FloatTensor(alphas)

        # Visualize caption and attention of best sequence
        plt = utils.visualize_att(img, seq, alphas, rev_word_map, args.smooth)
        plt.savefig('result.png')
if __name__ == '__main__':
    main()