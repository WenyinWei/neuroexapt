import os
import sys
import time
import glob
import numpy as np
import torch
import logging
import argparse
import torch.nn as nn
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
from neuroexapt.core.model import Network
from neuroexapt.core.architect import Architect
# Use the new utils
from neuroexapt.utils.utils import AvgrageMeter, accuracy

# Add progress bar support
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

def log_gpu_usage(msg=""):
    """Log current GPU usage and memory."""
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / 1024**3  # GB
        gpu_memory_cached = torch.cuda.memory_reserved() / 1024**3  # GB
        gpu_utilization = torch.cuda.utilization() if hasattr(torch.cuda, 'utilization') else 'N/A'
        print(f"{msg} GPU: {gpu_memory:.2f}GB used, {gpu_memory_cached:.2f}GB cached, Util: {gpu_utilization}")
    else:
        print(f"{msg} GPU not available")

parser = argparse.ArgumentParser("fast_cifar")
parser.add_argument('--data', type=str, default='./data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=64, help='batch size (larger for better GPU utilization)')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=10, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=6, help='total number of layers (reduced for speed)')
parser.add_argument('--potential_layers', type=int, default=2, help='number of potential layers (reduced for speed)')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='FAST_EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.8, help='portion of training data (increased for speed)')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
parser.add_argument('--num_workers', type=int, default=4, help='number of data loading workers (increased for speed)')
parser.add_argument('--pin_memory', action='store_true', default=True, help='pin memory for faster data loading')
parser.add_argument('--disable_visualization', action='store_true', default=False, help='disable visualization for speed')
parser.add_argument('--fast_mode', action='store_true', default=False, help='enable fast mode optimizations')


def main():
    args = parser.parse_args()
    args.save = 'search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
    if not os.path.exists(args.save):
        os.makedirs(args.save)
    
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    # Performance optimizations
    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True  # Optimize for consistent input sizes
    cudnn.deterministic = False  # Allow non-deterministic algorithms for speed
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    
    # Set tensor core optimizations
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    
    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)
    
    log_gpu_usage("Initial")

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()

    model = Network(args.init_channels, 10, args.layers, args.potential_layers)
    model = model.cuda()
    
    # Use mixed precision if available
    scaler = torch.cuda.amp.GradScaler()
    
    logging.info("param size = %fM", sum(p.numel() for p in model.parameters())/1e6)
    log_gpu_usage("After model creation")

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay)

    # Optimized data transforms
    import torchvision.transforms as transforms
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)

    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(args.train_portion * num_train))

    # Use larger batch size and more workers for better GPU utilization
    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.SubsetRandomSampler(indices[:split]),
        pin_memory=args.pin_memory, num_workers=args.num_workers,
        persistent_workers=True if args.num_workers > 0 else False)

    valid_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.SubsetRandomSampler(indices[split:num_train]),
        pin_memory=args.pin_memory, num_workers=args.num_workers,
        persistent_workers=True if args.num_workers > 0 else False)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.epochs, eta_min=args.learning_rate_min)

    architect = Architect(model, args)
    architect.criterion = criterion
    
    logging.info("Initial architecture: %d layers, %d potential layers", args.layers, args.potential_layers)
    logging.info("Model parameters: %.2fM", sum(p.numel() for p in model.parameters())/1e6)
    
    # Create epoch progress bar if available
    if TQDM_AVAILABLE:
        epoch_pbar = tqdm(range(args.epochs), desc="Epochs")
        epoch_range = epoch_pbar
    else:
        epoch_pbar = None
        epoch_range = range(args.epochs)

    best_valid_acc = 0.0
    
    for epoch in epoch_range:
        lr = scheduler.get_lr()[0]
        logging.info('epoch %d lr %e', epoch, lr)
        
        # Show architecture parameters occasionally
        if epoch % 5 == 0:
            logging.info('alphas_normal max = %.3f', torch.softmax(model.alphas_normal, dim=-1).max().item())
            logging.info('alphas_reduce max = %.3f', torch.softmax(model.alphas_reduce, dim=-1).max().item())
            if hasattr(model, 'alphas_gates') and model.alphas_gates:
                gate_activations = [torch.sigmoid(gate).item() for gate in model.alphas_gates]
                logging.info('gate_activations = %s', [f"{g:.3f}" for g in gate_activations])

        # Training
        train_acc, train_obj = train(train_queue, valid_queue, model, architect, criterion, optimizer, scaler, lr, args)
        logging.info('train_acc %.2f', train_acc)

        # Validation
        valid_acc, valid_obj = infer(valid_queue, model, criterion)
        logging.info('valid_acc %.2f', valid_acc)
        
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            logging.info('New best validation accuracy: %.2f', best_valid_acc)

        # Update epoch progress bar if available
        if epoch_pbar is not None:
            epoch_pbar.set_postfix({
                'Train_Acc': f'{train_acc:.1f}%',
                'Valid_Acc': f'{valid_acc:.1f}%',
                'Best': f'{best_valid_acc:.1f}%',
                'LR': f'{lr:.4f}'
            })
        
        # Update learning rate scheduler
        scheduler.step()

        # Memory management
        if hasattr(architect, 'cleanup_gradients'):
            architect.cleanup_gradients()

        # Log GPU usage periodically
        if epoch % 5 == 0:
            log_gpu_usage(f"Epoch {epoch}")

        # Save checkpoint occasionally
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            save_path = os.path.join(args.save, f'checkpoint_epoch_{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_valid_acc': best_valid_acc,
            }, save_path)
            logging.info("Checkpoint saved to %s", save_path)

    genotype = model.genotype()
    logging.info('Final Genotype = %s', genotype)
    logging.info('Best validation accuracy: %.2f', best_valid_acc)
    log_gpu_usage("Final")


def train(train_queue, valid_queue, model, architect, criterion, optimizer, scaler, lr, args):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    top5 = AvgrageMeter()

    # Create progress bar if available
    if TQDM_AVAILABLE:
        train_iter = tqdm(train_queue, desc="Training", leave=False)
    else:
        train_iter = train_queue

    for step, (input, target) in enumerate(train_iter):
        model.train()
        n = input.size(0)

        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # Get validation data for architecture update
        try:
            input_search, target_search = next(iter(valid_queue))
        except StopIteration:
            continue
            
        input_search = input_search.cuda(non_blocking=True)
        target_search = target_search.cuda(non_blocking=True)
        
        # Update architecture (less frequently for speed)
        if step % 2 == 0:  # Update architecture every 2 steps
            architect.step(input, target, input_search, target_search, lr, optimizer, unrolled=args.unrolled)

        # Update weights with mixed precision
        optimizer.zero_grad()
        
        if args.fast_mode:
            # Use mixed precision for speed
            with torch.cuda.amp.autocast():
                logits = model(input)
                loss = criterion(logits, target)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(input)
            loss = criterion(logits, target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

        prec1, prec5 = accuracy(logits, target, topk=(1, 5))
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        # Update progress bar if available
        if TQDM_AVAILABLE:
            train_iter.set_postfix({
                'Loss': f'{objs.avg:.3f}',
                'Top1': f'{top1.avg:.1f}%',
                'Top5': f'{top5.avg:.1f}%'
            })

        if step % args.report_freq == 0:
            logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    top5 = AvgrageMeter()
    model.eval()

    # Create progress bar if available
    if TQDM_AVAILABLE:
        valid_iter = tqdm(valid_queue, desc="Validation", leave=False)
    else:
        valid_iter = valid_queue

    with torch.no_grad():
        for step, (input, target) in enumerate(valid_iter):
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            logits = model(input)
            loss = criterion(logits, target)

            prec1, prec5 = accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

            # Update progress bar if available
            if TQDM_AVAILABLE:
                valid_iter.set_postfix({
                    'Loss': f'{objs.avg:.3f}',
                    'Top1': f'{top1.avg:.1f}%',
                    'Top5': f'{top5.avg:.1f}%'
                })

    return top1.avg, objs.avg


if __name__ == '__main__':
    main() 