import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as T
import random
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.tensorboard import SummaryWriter
from PIL import Image

import argparse
import datetime

from models.transformer import CangjieTransformer
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
            p, c = pred[:, 0], chidx[0]
            code = p.topk(1, dim=-1).indices.flatten().tolist()
            code = [self.map_rev[c] for c in code]
            gt = [self.map_rev[c] for c in gt[:, 0].tolist()]
            c = self.dataset.chs[c.item()]
            self.log.writelines([prefix + '%s: %s, gt: %s\n'%(c, code, gt)])


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--loader_workers', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=120)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--save_interval', type=int, default=10)
    parser.add_argument('--save_dir', type=str, default='logs/')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--table', type=str, default='data/Cangjie5.txt')
    parser.add_argument('--codemap', type=str, default='data/codemap_cangjie5.txt')
    parser.add_argument('--fonts', nargs='+', default=['data/hanazono/HanaMinA.ttf', 'data/hanazono/HanaMinB.ttf'])
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--grad_clip', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=114514, help="random seed for training")
    parser.add_argument('--alpha_codelen', type=float, default=1.)

    args = parser.parse_args()
    args.save_dir = os.path.join(args.save_dir, datetime.datetime.now().strftime("transformer-%m-%d-%Y-%H:%M:%S"))
    os.makedirs(args.save_dir)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    glyph = dset.Glyph(args.fonts)
    dataset = dset.CodeTableDataset(glyph, table=args.table, codemap=args.codemap)
    # args.pad = dataset.codemap['<pad>']
    train_length = int(len(dataset) * 0.7)
    train_set, val_set = torch.utils.data.random_split(dataset, [train_length, len(dataset) - train_length])
    train_loader = torch.utils.data.DataLoader(train_set, args.batch_size, True,
                                               collate_fn=dset.collate_batch_pad(dataset.codemap['<pad>']),
                                               num_workers=args.loader_workers,
                                               pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_set, args.batch_size, False,
                                             collate_fn=dset.collate_batch_pad(dataset.codemap['<pad>']),
                                             num_workers=args.loader_workers,
                                             pin_memory=True)

    train_augment = T.Compose([
        T.RandomResizedCrop(64, (0.5, 1/0.9), (3./6., 4./ 3.)),
        T.RandomRotation((-10, 10)),
        T.ToTensor()
    ])

    # encoder = Encoder(encode_channels=256).to(device)
    # encoder_optim = torch.optim.Adam(encoder.parameters(), lr=args.encoder_lr)
    # decoder = Decoder(dataset.codemap, 256, 4, 256, 4, dropout=0.5).to(device)
    # decoder_optim = torch.optim.Adam(decoder.parameters(), lr=args.decoder_lr)
    model = CangjieTransformer(dataset.codemap, 256, 4, 256, 4, dropout=0.2).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    # optim = torch.optim.SGD(model.parameters(), args.lr, momentum=0.8)
    epoch_start = 0
    if args.resume != None:
        print('loading checkpoint: %s'%args.resume)
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model'])
        optim.load_state_dict(checkpoint['optimizer'])
        epoch_start = checkpoint['epoch']

    sche = torch.optim.lr_scheduler.CosineAnnealingLR(optim, args.epochs, last_epoch=epoch_start-1)

    criterion = nn.CrossEntropyLoss(ignore_index=dataset.codemap['<pad>']).to(device)
    logger = PredLogger(dataset, args.save_dir, args.codemap)
    writer = SummaryWriter(args.save_dir)

    best_acc = 0

    for epoch in tqdm(range(epoch_start, args.epochs), position=0):
        with dataset.transform(train_augment):
            train(train_loader=train_loader,
                model=model,
                criterion=criterion,
                optimizer=optim,
                epoch=epoch,
                logger=logger,
                writer=writer,
                args=args)
        acc, imgs, scores = validate(val_loader=val_loader,
                       model=model,
                       criterion=criterion,
                       epoch=epoch,
                       logger=logger,
                       writer=writer,
                       args=args)
        sche.step()
        is_best = best_acc < acc # and epoch > 0
        best_acc = max(acc, best_acc)
        
        if epoch % args.save_interval == 0:
            data = {'epoch': epoch,
                    'acc': acc,
                    'model': model.state_dict(),
                    'optimizer': optim.state_dict()}
            save_checkpoint(data, epoch, is_best, args.save_dir)

            # vis = visualize_att(T.ToPILImage()(imgs[0].cpu()), scores[0].topk(1, dim=-1).indices.flatten().tolist(), alphas[0].view(-1, 13, 13).cpu(), logger.map_rev)
            # vis.savefig(os.path.join(args.save_dir, 'val_visualize_%d.png'%epoch))


def train(train_loader, model, criterion, optimizer, epoch, logger, writer, args):

    model.train()

    losses = AverageMeter()  # loss (per word decoded)
    aux_top1 = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()  # top5 accuracy

    # Batches
    progress = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
    for i, (imgs, caps, caplens, chidx) in progress:
        # Move to GPU, if available
        tgt = caps[:-1].to(device)
        tgt_y = caps[1:].to(device).permute(1, 0)

        imgs = imgs.to(device)
        # caps = caps.to(device)
        caplens = caplens.to(device)
        chidx = chidx.to(device)

        # Forward prop.
        # scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(feature, caps, caplens)
        scores, aux_codelen = model(imgs, tgt)
        # print(scores.size())

        # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
        # targets = caps_sorted[:, 1:]
        # print(scores.topk(1, dim=-1).indices.view(len(scores), -1))

        logger(scores, chidx, caps, 'train: ')
        # Remove timesteps that we didn't decode at, or are pads
        # pack_padded_sequence is an easy trick to do this

        # Calculate loss
        # loss = criterion(scores.permute(1, 0, 2).view(-1, scores.size(-1)), tgt_y.view(-1, tgt_y.size(-1)))
        scores = scores.permute(1, 0, 2).reshape(-1, scores.size(-1))
        tgt_y = tgt_y.reshape(-1)
        # print(scores.size(), tgt_y.size())
        loss = criterion(scores, tgt_y)
        # loss_aux = criterion(aux_out, chidx)
        loss += args.alpha_codelen * F.cross_entropy(aux_codelen, caplens - 3)

        # Add doubly stochastic attention regularization
        # loss += args.alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

        # Back prop.
        optimizer.zero_grad()
        # (loss + 10 * loss_aux).backward()
        loss.backward()

        # Clip gradients
        if args.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)


        # Update weights
        optimizer.step()

        # Keep track of metrics
        losses.update(loss.item(), sum(caplens - 1))
        top5.update(accuracy(scores, tgt_y, 5), sum(caplens - 1))
        top1.update(accuracy(scores, tgt_y, 1), sum(caplens - 1))
        progress.set_description("train loss: %.4f, top1: %2.2f%%, top5: %2.2f%%"%(losses.avg, top1.avg, top5.avg))

    writer.add_scalar('Loss/train', losses.avg, epoch)
    writer.add_scalar('Accuracy/train', top1.avg, epoch)


def validate(val_loader, model, criterion, epoch, logger, writer, args):
    """
    Performs one epoch's validation.
    :param val_loader: DataLoader for validation data.
    :param model: model
    :param criterion: loss layer
    :return: BLEU-4 score
    """
    model.eval()  # eval mode (no dropout or batchnorm)

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    aux_top1 = AverageMeter()

    with torch.no_grad():
        progress = tqdm(enumerate(val_loader), total=len(val_loader), leave=False)
        for i, (imgs, caps, caplens, chidx) in progress:
            # Move to device, if available
            tgt = caps[:-1].to(device)
            tgt_y = caps[1:].to(device).permute(1, 0)

            imgs = imgs.to(device)
            caps = caps.to(device)
            caplens = caplens.to(device)
            chidx = chidx.to(device)

            # Forward prop.
            # scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(feature, caps, caplens)
            scores, aux_codelen = model(imgs, tgt)

            # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
            # targets = caps_sorted[:, 1:]

            logger(scores, chidx, caps, 'val: ')
            # Remove timesteps that we didn't decode at, or are pads
            # pack_padded_sequence is an easy trick to do this

            # Calculate loss
            # loss = criterion(scores, tgt_y)
            # loss = criterion(scores.permute(1, 0, 2), tgt_y)
            # loss = criterion(scores.permute(1, 0, 2).reshape(-1, scores.size(-1)), tgt_y.reshape(-1, tgt_y.size(-1)))
            scores = scores.permute(1, 0, 2).reshape(-1, scores.size(-1))
            tgt_y = tgt_y.reshape(-1)
            loss = criterion(scores, tgt_y)


            # Add doubly stochastic attention regularization
            # loss += args.alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

            # Keep track of metrics
            losses.update(loss.item(), sum(caplens - 1))
            top5.update(accuracy(scores, tgt_y, 5), sum(caplens - 1))
            top1.update(accuracy(scores, tgt_y, 1), sum(caplens - 1))
            progress.set_description("valid loss: %.4f, top1: %2.2f%%, top5: %2.2f%%"%(losses.avg, top1.avg, top5.avg))
    writer.add_scalar('Loss/val', losses.avg, epoch)
    writer.add_scalar('Accuracy/val', top1.avg, epoch)


    return top1.avg, imgs, scores

if __name__ == '__main__':
    main()
