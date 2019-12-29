# pylint: disable=no-member, invalid-name, missing-docstring, redefined-builtin, arguments-differ, line-too-long
import argparse
import copy
import os
import random
from collections import defaultdict

import sklearn.metrics
import torch
import torch.nn as nn
import tqdm
from astropy.io import fits

from d4cnn.archi.efficientnet import equivariantize_network
from d4cnn.datasets.gg2 import GG2, target_transform


def image_transform(images):
    images = [fits.open(file, memmap=False)[0].data for file in images]
    images = [torch.from_numpy(x.byteswap().newbyteorder()) for x in images]

    # normalize the second moment of the channels to 1
    normalize = [3.5239e+10, 1.5327e+09, 1.8903e+09, 1.2963e+09]
    images = [x.mul(n) for x, n in zip(images, normalize)]

    # stack the 3 channels of small resolution together
    vis, j, y, h = images
    vis, nisp = vis[None], torch.stack([j, y, h])

    upsample = torch.nn.Upsample(size=225, mode='bilinear', align_corners=True)
    vis = upsample(vis[None])[0]
    nisp = upsample(nisp[None])[0]

    return torch.cat([vis, nisp])


def execute(args):
    # define model
    torch.manual_seed(args.init_seed)
    f = torch.hub.load(args.github, args.model, pretrained=False)

    equivariantize_network(f, in_channels=4)
    f.classifier = torch.nn.Linear(f.classifier.in_features, 1)
    f.to(args.device)
    print("{} parameters in total".format(sum(p.numel() for p in f.parameters())))

    # evaluation
    def evaluate(dataset, desc):
        loader = torch.utils.data.DataLoader(dataset, batch_size=args.bs, num_workers=2)

        with torch.no_grad():
            ote = []
            yte = []
            for x, y in tqdm.tqdm(loader, desc=desc):
                x, y = x.to(args.device), y.to(dtype=x.dtype, device=args.device)
                f.train()
                ote += [f(x).flatten()]
                yte += [y]

        return {
            'output': torch.cat(ote).cpu(),
            'labels': torch.cat(yte).cpu(),
        }

    # criterion and optimizer
    criterion = nn.SoftMarginLoss()
    optimizer = torch.optim.Adam(f.parameters(), lr=args.lr)

    # datasets
    dataset = GG2(args.root, transform=image_transform, target_transform=target_transform)
    print("{} images in total".format(len(dataset)))

    torch.manual_seed(args.data_seed)
    trainset, testset, _ = torch.utils.data.random_split(dataset, (args.ntr, args.nte, len(dataset) - args.ntr - args.nte))

    # training
    dummy_trainset = copy.deepcopy(trainset)
    dummy_trainset.dataset.transform = None

    trainloader = torch.utils.data.DataLoader(trainset, sampler=BalancedBatchSampler(dummy_trainset), batch_size=args.bs, drop_last=True, num_workers=4)

    results = []
    torch.manual_seed(args.batch_seed)

    avg_loss = RunningOp(1000 // args.bs, lambda x: sum(x) / len(x))
    avg_acc = RunningOp(1000 // args.bs, lambda x: sum(x) / len(x))

    for epoch in range(args.epoch):
        t = tqdm.tqdm(total=len(trainloader), desc='[epoch {}] training'.format(epoch + 1))
        for x, y in trainloader:
            x, y = x.to(args.device), y.to(dtype=x.dtype, device=args.device)

            f.train()
            out = f(x).flatten()
            loss = criterion(out, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            t.update(1)
            t.set_postfix_str("loss={0[loss]:.2f} acc={0[acc]:.2f}".format({
                'loss': avg_loss(loss.item()),
                'acc': avg_acc((out * y > 0).double().mean().item()),
            }))

        t.close()

        results += [{
            'epoch': epoch,
            'train': evaluate(trainset, '[epoch {}] eval trainset'.format(epoch + 1)),
            'test': evaluate(testset, '[epoch {}] eval testset'.format(epoch + 1)),
        }]

        auctr = sklearn.metrics.roc_auc_score(results[-1]['train']['labels'], results[-1]['train']['output'])
        aucte = sklearn.metrics.roc_auc_score(results[-1]['test']['labels'], results[-1]['test']['output'])
        print("train={:.2f}% test={:.2f}%".format(100 * auctr, 100 * aucte))

        yield {
            'args': args,
            'epochs': results,
            'state': f.state_dict(),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init_seed", type=int, required=True)
    parser.add_argument("--data_seed", type=int, required=True)
    parser.add_argument("--batch_seed", type=int, required=True)

    parser.add_argument("--ntr", type=int, required=True)
    parser.add_argument("--nte", type=int, required=True)

    parser.add_argument("--bs", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)

    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--device", type=str, required=True)
    parser.add_argument("--root", type=str, required=True)

    parser.add_argument("--github", type=str, default='rwightman/gen-efficientnet-pytorch')
    parser.add_argument("--model", type=str, default='tf_mobilenetv3_small_minimal_100')

    parser.add_argument("--pickle", type=str, required=True)
    args = parser.parse_args()

    torch.save(args, args.pickle)
    try:
        for res in execute(args):
            with open(args.pickle, 'wb') as f:
                torch.save(args, f)
                torch.save(res, f)
    except:
        os.remove(args.pickle)
        raise


class RunningOp:
    def __init__(self, n, op):
        self.x = []
        self.n = n
        self.op = op

    def __call__(self, x):
        self.x.append(x)
        self.x = self.x[-self.n:]
        return self.op(self.x)


def inf_shuffle(xs):
    while xs:
        random.shuffle(xs)
        for x in xs:
            yield x


class BalancedBatchSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, dataset):
        super().__init__(dataset)
        indices = defaultdict(list)
        for i, (_, label) in enumerate(dataset):
            indices[label].append(i)
        self.indices = list(indices.values())

        self.n = max(len(ids) for ids in self.indices) * len(self.indices)

    def __iter__(self):
        m = 0
        for xs in zip(*(inf_shuffle(xs) for xs in self.indices)):
            for i in xs:  # yield one index of each label
                yield i
                m += 1
                if m >= self.n:
                    return

    def __len__(self):
        return self.n


if __name__ == "__main__":
    main()
