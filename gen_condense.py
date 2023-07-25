import os
import sys
import time
import random
import argparse
import numpy as np
import pickle
import random

import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

import models.resnet as RN
import models.convnet as CN
import models.resnet_ap as RNAP
import models.densenet_cifar as DN
from gan_model import Generator, Discriminator
from utils import AverageMeter, accuracy, Normalize, Logger, rand_bbox
from augment import DiffAug
from data import get_pacs_datasets

from all_models import get_model

def str2bool(v):
    """Cast string to boolean"""
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

def gen_noisy_batch(args, clip_embeddings, label_batch=None):
    # hold_out_domain : 'p', 'a', 'c', 's' 
    # clip_embeddings : dict
    int2label = {
           0 : "dog",
           1 : "elephant",
           2 : "giraffe",
           3 : "guitar",
           4 : "horse",
           5 : "house",
           6 : "person"
        }

    domain_to_foldername = {
            "p": "photo",
            "a": "painting",
            "c": "cartoon",
            "s": "sketch"
        }
    count = 0
    embeddings = []
    keys = clip_embeddings.keys()    # key = f"a {DOMAIN} of a {CLASS}"

    if label_batch == None:
      while count != args.batch_size:
          key = random.sample(keys, 1)[0]
          if key.split(" ")[1] != domain_to_foldername[args.holdout_domain]:
              # print(key)
              count += 1
              embeddings.append(clip_embeddings[key])
    else:
      domains = ["p", "a", "c", "s"]
      domains.pop(domains.index(args.holdout_domain))
      for cls in label_batch:
        dom = domain_to_foldername[random.sample(domains, 1)]
        cls_name = int2label[cls]
        key = f"a {dom} of a {cls_name}"
        embeddings.append(clip_embeddings[key])

    assert len(embeddings) == args.batch_size    
    embeddings = torch.stack((embeddings))
    noise = torch.normal(0, torch.std(embeddings).item() / 100., size=(args.batch_size, 512))
    noisy_vectors_batch = embeddings + noise
    return noisy_vectors_batch

def load_data(args):
    """Obtain data"""

    # TODO: allow for PACS dataset
    transform_train = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    if args.data == "cifar10":
        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.491, 0.482, 0.447), (0.202, 0.199, 0.201)),
            ]
        )
        trainset = datasets.CIFAR10(
            root=args.data_dir, train=True, download=True, transform=transform_train
        )
        testset = datasets.CIFAR10(
            root=args.data_dir, train=False, download=True, transform=transform_test
        )
    elif args.data == "svhn":
        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.437, 0.444, 0.473), (0.198, 0.201, 0.197)),
            ]
        )
        trainset = datasets.SVHN(
            os.path.join(args.data_dir, "svhn"),
            split="train",
            download=True,
            transform=transform_train,
        )
        testset = datasets.SVHN(
            os.path.join(args.data_dir, "svhn"),
            split="test",
            download=True,
            transform=transform_test,
        )
    elif args.data == "fashion":
        transform_train = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        )
        transform_test = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.286,), (0.353,))]
        )

        trainset = datasets.FashionMNIST(
            args.data_dir, train=True, download=True, transform=transform_train
        )
        testset = datasets.FashionMNIST(
            args.data_dir, train=False, download=True, transform=transform_train
        )
    elif args.data == "mnist":
        transform_train = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        )
        transform_test = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.131,), (0.308,))]
        )

        trainset = datasets.MNIST(
            args.data_dir, train=True, download=True, transform=transform_train
        )
        testset = datasets.MNIST(
            args.data_dir, train=False, download=True, transform=transform_train
        )

    elif args.data == "pacs":
        trainset, testset = get_pacs_datasets(args.data_dir, args.holdout_domain, test_split=args.test_split, seed=args.seed)

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    return trainloader, testloader


# def define_model(args, num_classes, e_model=None):
#     """Obtain model for training and validating"""
#     if e_model:
#         model = e_model
#     else:
#         model = args.match_model

#     if args.data == "mnist" or args.data == "fashion":
#         nch = 1
#     else:
#         nch = 3

#     if model == "convnet":
#         return CN.ConvNet(num_classes, channel=nch)
#     elif model == "resnet10":
#         return RN.ResNet(args.data, 10, num_classes, nch=nch)
#     elif model == "resnet18":
#         return RN.ResNet(args.data, 18, num_classes, nch=nch)
#     elif model == "resnet34":
#         return RN.ResNet(args.data, 34, num_classes, nch=nch)
#     elif model == "resnet50":
#         return RN.ResNet(args.data, 50, num_classes, nch=nch)
#     elif model == "resnet101":
#         return RN.ResNet(args.data, 101, num_classes, nch=nch)
#     elif model == "resnet10_ap":
#         return RNAP.ResNetAP(args.data, 10, num_classes, nch=nch)
#     elif model == "resnet18_ap":
#         return RNAP.ResNetAP(args.data, 18, num_classes, nch=nch)
#     elif model == "resnet34_ap":
#         return RNAP.ResNetAP(args.data, 34, num_classes, nch=nch)
#     elif model == "resnet50_ap":
#         return RNAP.ResNetAP(args.data, 50, num_classes, nch=nch)
#     elif model == "resnet101_ap":
#         return RNAP.ResNetAP(args.data, 101, num_classes, nch=nch)
#     elif model == "densenet":
#         return DN.densenet_cifar(num_classes)


def calc_gradient_penalty(args, discriminator, img_real, img_syn):
    """Gradient penalty from Wasserstein GAN"""
    LAMBDA = 10
    n_size = img_real.shape[-1]
    batch_size = img_real.shape[0]
    n_channels = img_real.shape[1]

    alpha = torch.rand(batch_size, 1)
    alpha = alpha.expand(batch_size, int(img_real.nelement() / batch_size)).contiguous()
    alpha = alpha.view(batch_size, n_channels, n_size, n_size)
    alpha = alpha.cuda()

    img_syn = img_syn.view(batch_size, n_channels, n_size, n_size)
    interpolates = alpha * img_real.detach() + ((1 - alpha) * img_syn.detach())

    interpolates = interpolates.cuda()
    interpolates.requires_grad_(True)

    disc_interpolates, _ = discriminator(interpolates)

    gradients = torch.autograd.grad(
        outputs=disc_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty


def remove_aug(augtype, remove_aug):
    aug_list = []
    for aug in augtype.split("_"):
        if aug not in remove_aug.split("_"):
            aug_list.append(aug)

    return "_".join(aug_list)


def diffaug(args, device="cuda"):
    """Differentiable augmentation for condensation"""
    aug_type = args.aug_type
    if args.data == "cifar10":
        normalize = Normalize(
            (0.491, 0.482, 0.447), (0.202, 0.199, 0.201), device="cuda"
        )
    elif args.data == "svhn":
        normalize = Normalize(
            (0.437, 0.444, 0.473), (0.198, 0.201, 0.197), device="cuda"
        )
    elif args.data == "fashion":
        normalize = Normalize((0.286,), (0.353,), device="cuda")
    elif args.data == "mnist":
        normalize = Normalize((0.131,), (0.308,), device="cuda")
    else:
        normalize = torch.nn.Identity()
    print("Augmentataion Matching: ", aug_type)
    augment = DiffAug(strategy=aug_type, batch=True)
    aug_batch = transforms.Compose([normalize, augment])

    if args.mixup_net == "cut":
        aug_type = remove_aug(aug_type, "cutout")
    print("Augmentataion Net update: ", aug_type)
    augment_rand = DiffAug(strategy=aug_type, batch=False)
    aug_rand = transforms.Compose([normalize, augment_rand])

    return aug_batch, aug_rand


def train(
    args,
    epoch,
    generator,
    discriminator,
    optim_g,
    optim_d,
    trainloader,
    criterion,
    aug,
    aug_rand,
    clip_embeddings
):
    """The main training function for the generator"""
    generator.train()
    gen_losses = AverageMeter()
    disc_losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    for batch_idx, (img_real, lab_real) in enumerate(trainloader):
        img_real = img_real.cuda()
        lab_real = lab_real.cuda()

        # train the generator
        discriminator.eval()
        optim_g.zero_grad()

        # obtain the noise with one-hot class labels
        noise = gen_noisy_batch(args, clip_embeddings)
        noise = noise.cuda()

        img_syn = generator(noise)
        gen_source, gen_class = discriminator(img_syn)
        gen_source = gen_source.mean()
        gen_class = criterion(gen_class, lab_real)
        gen_loss = -gen_source + gen_class

        gen_loss.backward()
        optim_g.step()

        # train the discriminator
        discriminator.train()
        optim_d.zero_grad()
        lab_syn = torch.randint(args.num_classes, (args.batch_size,))
        # noise = torch.normal(0, 1, (args.batch_size, args.dim_noise))
        # lab_onehot = torch.zeros((args.batch_size, args.num_classes))
        # lab_onehot[torch.arange(args.batch_size), lab_syn] = 1
        # noise[torch.arange(args.batch_size), : args.num_classes] = lab_onehot[
        #     torch.arange(args.batch_size)
        # ]
        noise = gen_noisy_batch(args, clip_embeddings, lab_syn)
        noise = noise.cuda()
        lab_syn = lab_syn.cuda()

        with torch.no_grad():
            img_syn = generator(noise)

        disc_fake_source, disc_fake_class = discriminator(img_syn)
        disc_fake_source = disc_fake_source.mean()
        disc_fake_class = criterion(disc_fake_class, lab_syn)

        disc_real_source, disc_real_class = discriminator(img_real)
        acc1, acc5 = accuracy(disc_real_class.data, lab_real, topk=(1, 5))
        disc_real_source = disc_real_source.mean()
        disc_real_class = criterion(disc_real_class, lab_real)

        gradient_penalty = calc_gradient_penalty(args, discriminator, img_real, img_syn)

        disc_loss = (
            disc_fake_source
            - disc_real_source
            + disc_fake_class
            + disc_real_class
            + gradient_penalty
        )
        disc_loss.backward()
        optim_d.step()

        gen_losses.update(gen_loss.item())
        disc_losses.update(disc_loss.item())
        top1.update(acc1.item())
        top5.update(acc5.item())

        if (batch_idx + 1) % args.print_freq == 0:
            print(
                "[Train Epoch {} Iter {}] G Loss: {:.3f}({:.3f}) D Loss: {:.3f}({:.3f}) D Acc: {:.3f}({:.3f})".format(
                    epoch,
                    batch_idx + 1,
                    gen_losses.val,
                    gen_losses.avg,
                    disc_losses.val,
                    disc_losses.avg,
                    top1.val,
                    top1.avg,
                )
            )


def test(args, model, testloader, criterion):
    """Calculate accuracy"""
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    for batch_idx, (img, lab) in enumerate(testloader):
        img = img.cuda()
        lab = lab.cuda()

        with torch.no_grad():
            output = model(img)
        loss = criterion(output, lab)
        acc1, acc5 = accuracy(output.data, lab, topk=(1, 5))
        losses.update(loss.item(), output.shape[0])
        top1.update(acc1.item(), output.shape[0])
        top5.update(acc5.item(), output.shape[0])

    return top1.avg, top5.avg, losses.avg


def validate(args, generator, testloader, criterion, aug_rand, clip_embeddings):
    """Validate the generator performance"""
    all_best_top1 = []
    all_best_top5 = []
    for e_model in args.eval_model:
        print("Evaluating {}".format(e_model))
        # model = define_model(args, args.num_classes, e_model).cuda()
        model = get_model(e_model, args.num_classes).cuda()
        model.train()
        optim_model = torch.optim.SGD(
            model.parameters(),
            args.eval_lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )

        generator.eval()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        best_top1 = 0.0
        best_top5 = 0.0
        for epoch_idx in range(args.epochs_eval):
            for batch_idx in range(10 * args.ipc // args.batch_size + 1):
                # obtain pseudo samples with the generator
                lab_syn = torch.randint(args.num_classes, (args.batch_size,))
                noise = noise = gen_noisy_batch(args, clip_embeddings)
                noise = noise.cuda()
                lab_syn = lab_syn.cuda()

                with torch.no_grad():
                    img_syn = generator(noise)
                    img_syn = aug_rand(img_syn)

                if np.random.rand(1) < args.mix_p and args.mixup_net == "cut":
                    lam = np.random.beta(args.beta, args.beta)
                    rand_index = torch.randperm(len(img_syn)).cuda()

                    lab_syn_b = lab_syn[rand_index]
                    bbx1, bby1, bbx2, bby2 = rand_bbox(img_syn.size(), lam)
                    img_syn[:, :, bbx1:bbx2, bby1:bby2] = img_syn[
                        rand_index, :, bbx1:bbx2, bby1:bby2
                    ]
                    ratio = 1 - (
                        (bbx2 - bbx1)
                        * (bby2 - bby1)
                        / (img_syn.size()[-1] * img_syn.size()[-2])
                    )

                    output = model(img_syn)
                    loss = criterion(output, lab_syn) * ratio + criterion(
                        output, lab_syn_b
                    ) * (1.0 - ratio)
                else:
                    output = model(img_syn)
                    loss = criterion(output, lab_syn)

                acc1, acc5 = accuracy(output.data, lab_syn, topk=(1, 5))

                losses.update(loss.item(), img_syn.shape[0])
                top1.update(acc1.item(), img_syn.shape[0])
                top5.update(acc5.item(), img_syn.shape[0])

                optim_model.zero_grad()
                loss.backward()
                optim_model.step()

            if (epoch_idx + 1) % args.test_interval == 0:
                test_top1, test_top5, test_loss = test(
                    args, model, testloader, criterion
                )
                print(
                    "[Test Epoch {}] Top1: {:.3f} Top5: {:.3f}".format(
                        epoch_idx + 1, test_top1, test_top5
                    )
                )
                if test_top1 > best_top1:
                    best_top1 = test_top1
                    best_top5 = test_top5

        all_best_top1.append(best_top1)
        all_best_top5.append(best_top5)

    return all_best_top1, all_best_top5


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ipc", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--epochs-eval", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4) # 
    parser.add_argument("--eval-lr", type=float, default=3e-4) #0.01
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-5) # 5e-4
    parser.add_argument("--eval-model", type=str, nargs="+", default=["efficientnet_b0"])
    parser.add_argument("--dim-noise", type=int, default=512)  # 512 + 256
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--print-freq", type=int, default=50)
    parser.add_argument("--eval-interval", type=int, default=10)
    parser.add_argument("--test-interval", type=int, default=200)
    parser.add_argument("--data", type=str, default="pacs")
    parser.add_argument(
        "--holdout-domain",
        type=str,
        default="p",
        help="Must be one of `p`,`a`,`c`,`s` ",
    )
    parser.add_argument("--num-classes", type=int, default=7)
    parser.add_argument('--clip-embeddings', type=str, default='./DiM/embeds/pacs/clip_embeddings.pickle')
    parser.add_argument("--data-dir", type=str, default="./data/PACS")
    parser.add_argument("--output-dir", type=str, default="./results/")
    parser.add_argument("--logs-dir", type=str, default="./logs/")
    parser.add_argument("--aug-type", type=str, default="color_crop_cutout")
    parser.add_argument("--mixup-net", type=str, default="cut")
    parser.add_argument("--bias", type=str2bool, default=False)
    parser.add_argument("--fc", type=str2bool, default=False)
    parser.add_argument("--mix-p", type=float, default=-1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--test-split", type=float, default=0.2)
    parser.add_argument("--tag", type=str, default="test")
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    args.output_dir = args.output_dir + args.tag
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if not os.path.exists(args.output_dir + "/outputs"):
        os.makedirs(args.output_dir + "/outputs")

    if not os.path.exists(args.logs_dir):
        os.makedirs(args.logs_dir)
    args.logs_dir = args.logs_dir + args.tag
    if not os.path.exists(args.logs_dir):
        os.makedirs(args.logs_dir)
    sys.stdout = Logger(os.path.join(args.logs_dir, "logs.txt"))

    print(args)

    clip_embeddings = pickle.load(open(args.clip_embeddings, 'rb'))

    trainloader, testloader = load_data(args)

    generator = Generator(args.dim_noise).cuda()
    discriminator = Discriminator(args.num_classes).cuda()

    optim_g = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0, 0.9))
    optim_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0, 0.9))
    criterion = nn.CrossEntropyLoss()

    aug, aug_rand = diffaug(args)

    best_top1s = np.zeros((len(args.eval_model),))
    best_top5s = np.zeros((len(args.eval_model),))
    best_epochs = np.zeros((len(args.eval_model),))
    for epoch in range(args.epochs):
        print(f"####### Epoch {epoch}")
        generator.train()
        discriminator.train()
        train(
            args,
            epoch,
            generator,
            discriminator,
            optim_g,
            optim_d,
            trainloader,
            criterion,
            aug,
            aug_rand,
            clip_embeddings
        )

        # save image for visualization
        generator.eval()
        test_label = torch.tensor(list(range(args.num_classes)) * 10)
        test_noise = gen_noisy_batch(args, clip_embeddings)
        test_noise = test_noise.cuda()
        test_img_syn = (generator(test_noise))
        test_img_syn = make_grid(test_img_syn, nrow=10)
        save_image(
            test_img_syn,
            os.path.join(args.output_dir, "outputs/img_{}.png".format(epoch)),
        )
        generator.train()

        # if (epoch + 1) % args.eval_interval == 0:
        model_dict = {
            "generator": generator.state_dict(),
            "discriminator": discriminator.state_dict(),
            "optim_g": optim_g.state_dict(),
            "optim_d": optim_d.state_dict(),
        }
        torch.save(
            model_dict,
            os.path.join(args.output_dir, "model_dict_{}.pth".format(epoch)),
        )
        print("img and data saved!")

        top1s, top5s = validate(args, generator, testloader, criterion, aug_rand, clip_embeddings)
        for e_idx, e_model in enumerate(args.eval_model):
            if top1s[e_idx] > best_top1s[e_idx]:
                best_top1s[e_idx] = top1s[e_idx]
                best_top5s[e_idx] = top5s[e_idx]
                best_epochs[e_idx] = epoch
            print(
                "Current Best Epoch for {}: {}, Top1: {:.3f}, Top5: {:.3f}".format(
                    e_model,
                    best_epochs[e_idx],
                    best_top1s[e_idx],
                    best_top5s[e_idx],
                )
            )
