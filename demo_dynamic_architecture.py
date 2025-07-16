
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
from neuroexapt.utils.visualization import ascii_model_graph

# Add progress bar support
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

def log_memory_usage(msg=""):
    """Log current GPU memory usage."""
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        memory_reserved = torch.cuda.memory_reserved() / 1024**3   # GB
        logging.info(f"{msg} GPU Memory: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved")
    else:
        logging.info(f"{msg} GPU not available")

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='./data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
parser.add_argument('--potential_layers', type=int, default=4, help='number of potential layers to add')
parser.add_argument('--num_workers', type=int, default=2, help='number of data loading workers')
parser.add_argument('--resume_path', type=str, default=None, help='path to resume from checkpoint')
parser.add_argument('--early_stopping_patience', type=int, default=10, help='epochs to wait before early stopping')
parser.add_argument('--grad_accumulation_steps', type=int, default=1, help='gradient accumulation steps')
parser.add_argument('--validation_frequency', type=int, default=1, help='frequency of validation checks')


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

    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled=True
    torch.cuda.manual_seed(args.seed)
    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)
    if not args.unrolled:
        logging.warning("Note: DARTS' second-order approximation is disabled (unrolled=False).")
        logging.warning("This speeds up search but may affect results. For final runs, consider --unrolled.")


    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()

    model = Network(args.init_channels, 10, args.layers, args.potential_layers)
    model = model.cuda()
    logging.info("param size = %fM", sum(p.numel() for p in model.parameters())/1e6)

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay)

    # transform, _ = utils._data_transforms_cifar10(args)
    # For simplicity, use standard transforms
    import torchvision.transforms as transforms
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)

    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(args.train_portion * num_train))

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.SubsetRandomSampler(indices[:split]),
        pin_memory=True, num_workers=args.num_workers)

    valid_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.SubsetRandomSampler(indices[split:num_train]),
        pin_memory=True, num_workers=args.num_workers)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.epochs, eta_min=args.learning_rate_min)

    architect = Architect(model, args)
    architect.criterion = criterion  # Set the criterion for the architect
    
    start_epoch = 0
    # Load checkpoint if resume_path is provided
    if args.resume_path and os.path.exists(args.resume_path):
        logging.info("Resuming from checkpoint: %s", args.resume_path)
        checkpoint = torch.load(args.resume_path)
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        architect.optimizer.load_state_dict(checkpoint['architect_optimizer_state_dict'])
        logging.info("Resumed from epoch %d", start_epoch)
    else:
        # Visualize initial architecture only on a fresh run (simplified)
        logging.info("Initial architecture: %d layers, %d potential layers", args.layers, args.potential_layers)
        logging.info("Model parameters: %.2fM", sum(p.numel() for p in model.parameters())/1e6)
        # Only show detailed visualization for very small models
        if args.layers <= 4:
            ascii_model_graph(model, force_show=True, sample_input=torch.randn(1, 3, 32, 32))


    # Create epoch progress bar if available
    if TQDM_AVAILABLE:
        epoch_pbar = tqdm(range(start_epoch, args.epochs), desc="Epochs", initial=start_epoch, total=args.epochs)
        epoch_range = epoch_pbar
    else:
        epoch_pbar = None
        epoch_range = range(start_epoch, args.epochs)

    # Early stopping variables
    best_valid_acc = 0.0
    patience_counter = 0
    
    for epoch in epoch_range:
        lr = scheduler.get_lr()[0]
        logging.info('epoch %d lr %e', epoch, lr)

        # Show architecture parameters
        logging.info('alphas_normal = %s', torch.softmax(model.alphas_normal, dim=-1))
        logging.info('alphas_reduce = %s', torch.softmax(model.alphas_reduce, dim=-1))
        if args.potential_layers > 0:
            logging.info('alphas_gates = %s', [torch.sigmoid(p) for p in model.alphas_gates])

        # training
        train_acc, train_obj = train(train_queue, valid_queue, model, architect, criterion, optimizer, lr, args)
        logging.info('train_acc %f', train_acc)

        # validation (only run if needed)
        if epoch % args.validation_frequency == 0:
            valid_acc, valid_obj = infer(valid_queue, model, criterion)
            logging.info('valid_acc %f', valid_acc)
            
            # Early stopping check
            if valid_acc > best_valid_acc:
                best_valid_acc = valid_acc
                patience_counter = 0
                logging.info('New best validation accuracy: %f', best_valid_acc)
            else:
                patience_counter += 1
                if patience_counter >= args.early_stopping_patience:
                    logging.info('Early stopping triggered at epoch %d', epoch)
                    break
        else:
            # Use previous validation accuracy for progress bar
            valid_acc, valid_obj = best_valid_acc, 0.0
        
        # Update epoch progress bar if available
        if epoch_pbar is not None:
            epoch_pbar.set_postfix({
                'Train_Acc': f'{train_acc:.1f}%',
                'Valid_Acc': f'{valid_acc:.1f}%',
                'LR': f'{lr:.4f}'
            })
        
        # Update learning rate scheduler
        scheduler.step()

        # Clean up gradients to prevent memory leaks
        if hasattr(architect, 'cleanup_gradients'):
            architect.cleanup_gradients()

        # Log memory usage periodically
        if epoch % 10 == 0:
            log_memory_usage(f"Epoch {epoch}")

        # Visualize and save checkpoint (simplified visualization)
        if epoch % 10 == 0 or epoch == args.epochs - 1:  # Only visualize every 10 epochs or at the end
            logging.info("Visualizing architecture at epoch %d:", epoch)
            if args.layers <= 4:  # Only show detailed visualization for small models
                ascii_model_graph(model, force_show=False, sample_input=torch.randn(1, 3, 32, 32))
            else:
                logging.info("Model has %d layers, skipping detailed visualization", args.layers)
        
        # Show architecture statistics instead of full visualization
        if hasattr(model, 'alphas_gates') and model.alphas_gates:
            gate_activations = [torch.sigmoid(gate).item() for gate in model.alphas_gates]
            logging.info("Gate activations: %s", [f"{g:.3f}" for g in gate_activations])
        
        save_path = os.path.join(args.save, 'checkpoint.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'architect_optimizer_state_dict': architect.optimizer.state_dict(),
        }, save_path)
        logging.info("Checkpoint saved to %s", save_path)

    genotype = model.genotype()
    logging.info('Final Genotype = %s', genotype)

def train(train_queue, valid_queue, model, architect, criterion, optimizer, lr, args):
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

        input = Variable(input, requires_grad=False).cuda()
        target = Variable(target, requires_grad=False).cuda(non_blocking=True)

        # get a random minibatch from the search queue with replacement
        input_search, target_search = next(iter(valid_queue))
        input_search = Variable(input_search, requires_grad=False).cuda()
        target_search = Variable(target_search, requires_grad=False).cuda(non_blocking=True)
        
        # Update architecture
        architect.step(input, target, input_search, target_search, lr, optimizer, unrolled=args.unrolled)

        # Update weights (with gradient accumulation)
        if step % args.grad_accumulation_steps == 0:
            optimizer.zero_grad()
        
        logits = model(input)
        loss = criterion(logits, target)
        
        # Scale loss by accumulation steps
        loss = loss / args.grad_accumulation_steps
        loss.backward()
        
        if (step + 1) % args.grad_accumulation_steps == 0:
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
            input = Variable(input).cuda()
            target = Variable(target).cuda(non_blocking=True)

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

            if step % 50 == 0:
                logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg

if __name__ == '__main__':
    main() 