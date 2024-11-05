import argparse, math, os, time, pdb
import torchvision
import torchvision.transforms as transforms
import models, torch
from models.normalizer import Normalize_layer
from train import train
from evaluate import evaluate


def print_log(print_string, log):
    print("{}".format(print_string))
    log.write('{}\n'.format(print_string))
    log.flush()
def time_string():
  ISOTIMEFORMAT='%Y-%m-%d %X'
  string = '[{}]'.format(time.strftime( ISOTIMEFORMAT, time.gmtime(time.time()) ))
  return string


def parse_args():
    parser = argparse.ArgumentParser(description="AdaNI for adversarial robustness")
    # dataset and model config
    parser.add_argument('--dataset', type=str,  default='cifar10', choices=['cifar10', 'cifar100'])
    parser.add_argument('--baseline', type=str, choices=['resnet_v1', 'resnet_v2'], default='resnet_v1')
    parser.add_argument('--res_v1_depth', type=int, default=20, help='depth of the res v1')
    parser.add_argument('--res_v2_num_blocks', type=int, nargs=4, default=[2,2,2,2], help='num blocks for each of the four layers of Res V2')
    # training optimization parameters
    parser.add_argument('--epochs', type=int, default=350, help='training epochs number')
    parser.add_argument('--batch_size', type=int, default=128, help='Training batch size.')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='The Learning Rate.')
    parser.add_argument('--lr_schedule', type=int, nargs='+', default=[150,250], help='epochs in which lr is decreased')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--model_base', type=str, required=True, help='path to where save the models')
    parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
    parser.add_argument('--gpu_id', type=int, default=0, help='device range [0,ngpu-1]')
    parser.add_argument('--noise_add_delay', type=int, default=10, help='number of epochs to delay noise injection')#10
    parser.add_argument('--adv_train_delay', type=int, default=20, help='number of epochs to delay adversarial training')#20
    parser.add_argument('--gamma', type=float, default=1e-4, help='parameter gamma in equation (7)')
    return parser.parse_args()


def main(args):
    '''
        pipeline for training and evaluating robust deep convolutional models with Learn2Perturb
    '''
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if args.ngpu == 1:
        # make only devices indexed by #gpu_id visible
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    ## create the folder to save models
    if not os.path.exists(args.model_base):
        os.makedirs(args.model_base)

    log_test = open(os.path.join(args.model_base, 'log_test.txt'), 'w')
    log_train = open(os.path.join(args.model_base, 'log_train.txt'), 'w')

    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                    std=[0.2023, 0.1994, 0.2010])

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    if args.dataset == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    else: # cifar100
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    if args.dataset == 'cifar10':
        n_classes = 10
    else: 
        n_classes = 100

    if args.baseline == 'resnet_v1':
        model = models.resnet_v1.adani_resnet_v1(depth= args.res_v1_depth, num_classes= n_classes)
    else:
        model = models.resnet_v2.adani_resnet_v1(num_blocks= args.res_v2_num_blocks, num_classes= n_classes)
    criterion = torch.nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        model.cuda()
        criterion.cuda()
    
    if args.baseline == 'resnet_v1':
        layers = [model.stage_1, model.stage_2, model.stage_3]
    else:
        layers = [model.layer1, model.layer2, model.layer3, model.layer4]

    model_sigma_maps = []
    model_sigma_maps.append(model.cn1_sigma_map)

    for layer in layers:
        for block in layer:
            model_sigma_maps.append(block.sigma_map)

    normal_param = [
        param for name, param in model.named_parameters()
        if not 'sigma' in name
    ]

    sigma_param = [
        param for name, param in model.named_parameters()
        if 'sigma' in name
    ]

    optimizer1 = torch.optim.SGD(normal_param,
        lr=args.learning_rate,
        momentum=args.momentum, weight_decay=args.weight_decay,
        nesterov=True
    )

    optimizer2 = torch.optim.SGD(sigma_param,
        lr=args.learning_rate,
        momentum=args.momentum, weight_decay=0,
        nesterov=True
    )


    for epoch in range(args.epochs):
        print("epoch: {} / {} ...".format(epoch+1, args.epochs))
        print("    Training:")
        train(model, trainloader, epoch, optimizer1, optimizer2, criterion, layers, model_sigma_maps, args)


        if (epoch +1) > 300:
            print("    Evaluation:")
            acc_clean = evaluate(model, testloader, attack=None)
            acc_pgd = evaluate(model, testloader, attack='pgd')
            acc_fgsm = evaluate(model, testloader, attack='fgsm')
            print_log('{:s} [Epoch={:03d}/{:03d}]'.format(time_string(), epoch, args.epochs)
                + ' [Clean : Accuracy={:.2f}]'.format(acc_clean), log_test)
            print_log('{:s} [Epoch={:03d}/{:03d}]'.format(time_string(), epoch, args.epochs)
                + ' [PGD : Accuracy={:.2f}]'.format(acc_pgd), log_test)
            print_log('{:s} [Epoch={:03d}/{:03d}]'.format(time_string(), epoch, args.epochs)
                + ' [FGSM : Accuracy={:.2f}]'.format(acc_fgsm), log_test)

        elif (epoch +1) % 25 == 0:
            print("    Evaluation:")
            acc_clean = evaluate(model, testloader, attack=None)
            acc_pgd = evaluate(model, testloader, attack='pgd')
            acc_fgsm = evaluate(model, testloader, attack='fgsm')
            print_log('{:s} [Epoch={:03d}/{:03d}]'.format(time_string(), epoch, args.epochs)
                + ' [Clean : Accuracy={:.2f}]'.format(acc_clean), log_test)
            print_log('{:s} [Epoch={:03d}/{:03d}]'.format(time_string(), epoch, args.epochs)
                + ' [PGD : Accuracy={:.2f}]'.format(acc_pgd), log_test)
            print_log('{:s} [Epoch={:03d}/{:03d}]'.format(time_string(), epoch, args.epochs)
                + ' [FGSM : Accuracy={:.2f}]'.format(acc_fgsm), log_test)

        if (epoch +1) % 25 == 0:
            path = args.model_base + str(epoch + 1) + ".pt"
            torch.save(model, path)


if __name__ =='__main__':
    args = parse_args()
    main(args)
