import torch
import torch, os,  argparse, shutil, time, torchvision, cv2
from attacks.pgd import pgd
from attacks.fgsm import fgsm
from attacks.eot import EOT_FGSM, EOT_PGD
import torchvision.transforms as transforms


def loaddata():
    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                std=[0.2023, 0.1994, 0.2010])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])
    testset = torchvision.datasets.CIFAR10(root='./data/', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=16, shuffle=False, num_workers=2)
    return testloader


def evaluate(model, testloader, attack=None, epsilon = 8/255):
    model.eval()
    correct = 0
    total = 0   
   
    for data in testloader:
        inputs, labels = data[0], data[1]
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            labels = labels.cuda()

        if attack == 'pgd':
            perturbed_image = pgd(model, inputs, labels, epsilon = epsilon)
            outputs = model(perturbed_image, labels, mixBN=False)[0]
            input_info = "Accuracy under pgd attack with epsilon = " + str(epsilon)
        elif attack == 'fgsm':
            perturbed_image = fgsm(model, inputs, labels, epsilon = epsilon)
            outputs = model(perturbed_image, labels, mixBN=False)[0]
            input_info = "Accuracy under fgsm attack with epsilon = " + str(epsilon)
        elif attack == 'EOT-fgsm':
            perturbed_image = EOT_FGSM(model, inputs, labels, epsilon = epsilon)
            outputs = model(perturbed_image, labels, mixBN=False)[0]
            input_info = "Accuracy under EOT-fgsm attack with epsilon = " + str(epsilon)
        elif attack == 'EOT-pgd':
            perturbed_image = EOT_PGD(model, inputs, labels, epsilon = epsilon)
            outputs = model(perturbed_image, labels, mixBN=False)[0]
            input_info = "Accuracy under EOT-pgd attack with epsilon = " + str(epsilon)
        elif attack is None:
            outputs = model(inputs, labels, mixBN=False)[0]
            input_info = "Clean data accuracy"
        else:
            raise NotImplementedError("Attack not supported!")

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    acc = correct/total
    print('        {} = {}'.format(input_info, 100*acc))
    return 100*acc


if __name__=="__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'  
    model=torch.load('./checkpoints/cifar10_adani/350.pt')
    testloader = loaddata()

    acc_clean = evaluate(model, testloader, attack=None, epsilon = 8/255)
    acc_fgsm = evaluate(model, testloader, attack='EOT-fgsm', epsilon = 8/255)
    acc_pgd = evaluate(model, testloader, attack='EOT-pgd', epsilon = 8/255)
    print(acc_clean, acc_fgsm, acc_pgd)
