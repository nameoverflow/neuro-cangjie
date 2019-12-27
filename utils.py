from PIL import ImageFont, ImageDraw, Image
from fontTools.ttLib import TTFont
import matplotlib; matplotlib.use('agg')

import torch
import os
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import skimage.transform

from torchvision import transforms

def load_map(path):
    with open(path) as f:
        m_rev = [s.strip() for s in f.readlines()]
    m = {v: i for i, v in enumerate(m_rev)}
    return m, m_rev

class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(scores, targets, k):
    """
    Computes top-k accuracy, from predicted and true labels.
    :param scores: scores from the model
    :param targets: true labels
    :param k: k in top-k accuracy
    :return: top-k accuracy
    """

    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size)


def save_checkpoint(epoch, encoder, decoder, encoder_optimizer, decoder_optimizer, acc, is_best, save_dir):
    """
    Saves model checkpoint.
    :param data_name: base name of processed dataset
    :param epoch: epoch number
    :param epochs_since_improvement: number of epochs since last improvement in BLEU-4 score
    :param encoder: encoder model
    :param decoder: decoder model
    :param encoder_optimizer: optimizer to update encoder's weights, if fine-tuning
    :param decoder_optimizer: optimizer to update decoder's weights
    :param acc: validation acc
    :param is_best: is this checkpoint the best so far?
    """
    state = {'epoch': epoch,
             'acc': acc,
             'encoder': encoder.state_dict(),
             'decoder': decoder.state_dict(),
             'encoder_optimizer': encoder_optimizer.state_dict(),
             'decoder_optimizer': decoder_optimizer.state_dict()}
    filename = os.path.join(save_dir, 'checkpoint.pth.tar')
    torch.save(state, filename)
    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    if is_best:
        torch.save(state, filename + '.best.%d'%epoch)


def visualize_att(image, seq, alphas, rev_word_map, smooth=True):
    """
    Visualizes caption with weights at every word.
    Adapted from paper authors' repo: https://github.com/kelvinxu/arctic-captions/blob/master/alpha_visualization.ipynb
    :param image_path: path to image that has been captioned
    :param seq: caption
    :param alphas: weights
    :param rev_word_map: reverse word mapping, i.e. ix2word
    :param smooth: smooth weights?
    """
    image = image.resize([13 * 12, 13 * 12], Image.LANCZOS)
    image = Image.fromarray(255 - np.array(image), mode='L')

    words = [rev_word_map[ind] for ind in seq]

    plt.clf()
    for t in range(len(words)):
        if t > 50:
            break
        plt.subplot(np.ceil(len(words) / 5.), 5, t + 1)

        plt.text(0, 1, '%s' % (words[t]), color='black', backgroundcolor='white', fontsize=12)
        plt.imshow(image)
        current_alpha = alphas[t, :]
        if smooth:
            alpha = skimage.transform.pyramid_expand(current_alpha.numpy(), upscale=12, sigma=8)
        else:
            alpha = skimage.transform.resize(current_alpha.numpy(), [13 * 12, 13 * 12])
        if t == 0:
            plt.imshow(alpha, alpha=0)
        else:
            plt.imshow(alpha, alpha=0.8)
        plt.set_cmap(cm.Greys_r)
        plt.axis('off')

    return plt.gcf()
 
