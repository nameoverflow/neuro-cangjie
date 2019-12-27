import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as T
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.tensorboard import SummaryWriter
from PIL import Image

import argparse
import datetime

import models
import dataset as dset

from tqdm import tqdm

from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PredLogger(object):
    def __init__(self, dataset, savedir, codemap):
        self.dataset = dataset
        self.log = open(os.path.join(savedir, 'pred'), 'a')
        self.map, self.map_rev = load_map(codemap)

    def __call__(self, pred, chidx, gt, prefix):
        with torch.no_grad():
            p, c = pred[0], chidx[0]
            code = p.topk(1, dim=-1).indices.flatten().tolist()
            code = [self.map_rev[c] for c in code]
            gt = [self.map_rev[c] for c in gt[0].tolist()]
            c = self.dataset.chs[c.item()]
            self.log.writelines([prefix + '%s: %s, gt: %s\n'%(c, code, gt)])


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--loader_workers', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=1200)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--save_interval', type=int, default=10)
    parser.add_argument('--save_dir', type=str, default='logs/')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--table', type=str, default='data/Cangjie5.txt')
    parser.add_argument('--codemap', type=str, default='data/codemap_cangjie5.txt')
    parser.add_argument('--fonts', nargs='+', default=['data/hanazono/HanaMinA.ttf', 'data/hanazono/HanaMinB.ttf'])
    parser.add_argument('--encoder_lr', type=float, default=1e-3)
    parser.add_argument('--decoder_lr', type=float, default=1e-3)
    parser.add_argument('--alpha_c', type=float, default=1.)
    parser.add_argument('--grad_clip', type=float, default=5.)
    args = parser.parse_args()
    args.save_dir = os.path.join(args.save_dir, datetime.datetime.now().strftime("%m-%d-%Y-%H:%M:%S"))
    os.makedirs(args.save_dir)

    glyph = dset.Glyph(args.fonts)
    dataset = dset.CodeTableDataset(glyph, table=args.table, codemap=args.codemap)
    train_length = int(len(dataset) * 0.7)
    train_set, val_set = torch.utils.data.random_split(dataset, [train_length, len(dataset) - train_length])
    train_loader = torch.utils.data.DataLoader(train_set, args.batch_size, True,
                                               collate_fn=dset.collate_batch,
                                               num_workers=args.loader_workers,
                                               pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_set, args.batch_size, False,
                                             collate_fn=dset.collate_batch,
                                             num_workers=args.loader_workers,
                                             pin_memory=True)

    train_augment = T.Compose([
        T.RandomResizedCrop(64, (0.5, 1/0.9), (3./6., 4./ 3.)),
        T.RandomRotation((-10, 10)),
        T.ToTensor()
    ])

    encoder = models.Encoder(encode_channels=256).to(device)
    encoder_optim = torch.optim.Adam(encoder.parameters(), lr=args.encoder_lr)
    decoder = models.Decoder(128, 256, 256, 26 + 2, encoder_dim=256, dropout=0.7).to(device)
    decoder_optim = torch.optim.Adam(decoder.parameters(), lr=args.decoder_lr)
    epoch_start = 0
    if args.resume != None:
        print('loading checkpoint: %s'%args.resume)
        checkpoint = torch.load(args.resume, map_location=device)
        decoder.load_state_dict(checkpoint['decoder'])
        decoder_optim.load_state_dict(checkpoint['decoder_optimizer'])
        encoder.load_state_dict(checkpoint['encoder'])
        encoder_optim.load_state_dict(checkpoint['encoder_optimizer'])
        epoch_start = checkpoint['epoch']

    criterion = nn.CrossEntropyLoss().to(device)
    logger = PredLogger(dataset, args.save_dir, args.codemap)
    writer = SummaryWriter(args.save_dir)

    best_acc = 0

    for epoch in tqdm(range(epoch_start, args.epochs), position=0):
        with dataset.transform(train_augment):
            train(train_loader=train_loader,
                encoder=encoder,
                decoder=decoder,
                criterion=criterion,
                encoder_optimizer=encoder_optim,
                decoder_optimizer=decoder_optim,
                epoch=epoch,
                logger=logger,
                writer=writer,
                args=args)
        acc, imgs, scores, alphas = validate(val_loader=val_loader,
                       encoder=encoder,
                       decoder=decoder,
                       criterion=criterion,
                       epoch=epoch,
                       logger=logger,
                       writer=writer,
                       args=args)
        
        is_best = best_acc < acc # and epoch > 0
        best_acc = max(acc, best_acc)
        
        if epoch % args.save_interval == args.save_interval - 1 or is_best:
            save_checkpoint(epoch, encoder, decoder, encoder_optim, decoder_optim, acc, is_best, args.save_dir)
            vis = visualize_att(T.ToPILImage()(imgs[0].cpu()), scores[0].topk(1, dim=-1).indices.flatten().tolist(), alphas[0].view(-1, 13, 13).cpu(), logger.map_rev)
            vis.savefig(os.path.join(args.save_dir, 'val_visualize_%d.png'%epoch))


def train(train_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, epoch, logger, writer, args):

    decoder.train()  # train mode (dropout and batchnorm is used)
    encoder.train()

    losses = AverageMeter()  # loss (per word decoded)
    aux_top1 = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()  # top5 accuracy

    # Batches
    progress = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
    for i, (imgs, caps, caplens, chidx) in progress:
        # Move to GPU, if available
        imgs = imgs.to(device)
        caps = caps.to(device)
        caplens = caplens.to(device)
        chidx = chidx.to(device)

        # Forward prop.
        feature = encoder(imgs)
        scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(feature, caps, caplens)

        # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
        targets = caps_sorted[:, 1:]
        # print(scores.topk(1, dim=-1).indices.view(len(scores), -1))

        logger(scores, chidx[sort_ind], targets, 'train: ')
        # Remove timesteps that we didn't decode at, or are pads
        # pack_padded_sequence is an easy trick to do this
        scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
        targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data

        # Calculate loss
        loss = criterion(scores, targets)
        # loss_aux = criterion(aux_out, chidx)

        # Add doubly stochastic attention regularization
        loss += args.alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

        # Back prop.
        decoder_optimizer.zero_grad()
        if encoder_optimizer is not None:
            encoder_optimizer.zero_grad()
        # (loss + 10 * loss_aux).backward()
        loss.backward()

        # Clip gradients
        if args.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), args.grad_clip)
            if encoder_optimizer is not None:
                torch.nn.utils.clip_grad_norm_(encoder.parameters(), args.grad_clip)

        # Update weights
        decoder_optimizer.step()
        if encoder_optimizer is not None:
            encoder_optimizer.step()

        # Keep track of metrics
        losses.update(loss.item(), sum(decode_lengths))
        top5.update(accuracy(scores, targets, 5), sum(decode_lengths))
        top1.update(accuracy(scores, targets, 1), sum(decode_lengths))
        progress.set_description("train loss: %.4f, top1: %2.2f%%, top5: %2.2f%%"%(losses.avg, top1.avg, top5.avg))

    writer.add_scalar('Loss/train', losses.avg, epoch)
    writer.add_scalar('Accuracy/train', top1.avg, epoch)


def validate(val_loader, encoder, decoder, criterion, epoch, logger, writer, args):
    """
    Performs one epoch's validation.
    :param val_loader: DataLoader for validation data.
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :return: BLEU-4 score
    """
    decoder.eval()  # eval mode (no dropout or batchnorm)
    if encoder is not None:
        encoder.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    aux_top1 = AverageMeter()

    with torch.no_grad():
        progress = tqdm(enumerate(val_loader), total=len(val_loader), leave=False)
        for i, (imgs, caps, caplens, chidx) in progress:
            # Move to device, if available
            imgs = imgs.to(device)
            caps = caps.to(device)
            caplens = caplens.to(device)
            chidx = chidx.to(device)

            # Forward prop.
            if encoder is not None:
                feature = encoder(imgs)
            scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(feature, caps, caplens)

            # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
            targets = caps_sorted[:, 1:]

            logger(scores, chidx[sort_ind], targets, 'val: ')
            # Remove timesteps that we didn't decode at, or are pads
            # pack_padded_sequence is an easy trick to do this
            scores_packed = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
            targets_packed = pack_padded_sequence(targets, decode_lengths, batch_first=True).data

            # Calculate loss
            loss = criterion(scores_packed, targets_packed)

            # Add doubly stochastic attention regularization
            loss += args.alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

            # Keep track of metrics
            losses.update(loss.item(), sum(decode_lengths))
            top5.update(accuracy(scores_packed, targets_packed, 5), sum(decode_lengths))
            top1.update(accuracy(scores_packed, targets_packed, 1), sum(decode_lengths))
            progress.set_description("valid loss: %.4f, top1: %2.2f%%, top5: %2.2f%%"%(losses.avg, top1.avg, top5.avg))
    writer.add_scalar('Loss/val', losses.avg, epoch)
    writer.add_scalar('Accuracy/val', top1.avg, epoch)


    return top1.avg, imgs[sort_ind], scores, alphas

if __name__ == '__main__':
    main()
