import torch
from torchvision.transforms import v2
from models import CMLineardo, erm, BayesianNet
from tools import entropy, stack_features, cross_entropy, plot_dictionary2, entropy_drop_out, plot_dictionary3, calc_ent_batched
from torchvision import transforms
from cmnist_ram import ColoredMNISTRAM, train_batched, test_batched
from torch.utils.data import Subset
import numpy as np


def main(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    al_iters = 10
    al_batch_size = 100
    use_cuda = True
    log = {'train_acc':[], 'ent1': [], 'cross_ent_1': [],'ent2': [], 'cross_ent_2': [], 'test_acc':[], 'num points': [], 'causal acc': [], 'sp acc': []}
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    trans = transforms.Compose([
        transforms.ToTensor()
    ])
    # training datasets
    dataset1_train = ColoredMNISTRAM(root='./data', spurious_noise=0.0, 
                                     causal_noise=0.5,
                                     transform=trans, start_idx=0, num_samples=5000, 
                                     flip_sp=False)
    dataset2_train = ColoredMNISTRAM(root='./data', spurious_noise=0.0, 
                                     causal_noise=0.0,
                                     transform=trans, start_idx=5000, num_samples=500, flip_sp=True)
    dataset_train = torch.utils.data.ConcatDataset([dataset1_train, dataset2_train])
    
    # unseen dataset
    dataset1_unseen = ColoredMNISTRAM(root='./data', spurious_noise=0, 
                                     causal_noise=0,
                                     transform=trans, start_idx=6000, num_samples=5000,
                                    flip_sp=False)
    dataset2_unseen = ColoredMNISTRAM(root='./data', spurious_noise=0, 
                                     causal_noise=0,
                                     transform=trans, start_idx=7000, num_samples=5000,
                                      flip_sp=True)
    dataloader1_unseen = torch.utils.data.DataLoader(dataset1_unseen,
                                                      batch_size=64, **kwargs)
    dataloader2_unseen = torch.utils.data.DataLoader(dataset2_unseen,
                                                      batch_size=64, **kwargs)
    test_loader = torch.utils.data.DataLoader(ColoredMNISTRAM(root='./data',
                                                              train=False, spurious_noise=0.5, 
                                                              causal_noise=0.0,
                                                              transform=trans,
                                                              start_idx=0, 
                                                              num_samples=5000,
                                                              flip_sp=True),
                                              batch_size=64, shuffle=True, **kwargs)
    data_causal = ColoredMNISTRAM(root='./data', spurious_noise=0.5, 
                                     causal_noise=0,
                                     transform=trans, start_idx=8000, num_samples=5000, 
                                     flip_sp=False)
    data_sp = ColoredMNISTRAM(root='./data', spurious_noise=0.0, 
                              causal_noise=0.5,
                              transform=trans, start_idx=9000, 
                              num_samples=500, flip_sp=False)

    dataloader_causal = torch.utils.data.DataLoader(data_causal,
                                                    batch_size=64, **kwargs)
    dataloader_sp = torch.utils.data.DataLoader(data_sp,
                                                batch_size=64, **kwargs)

    num_epochs = 150
    for i in range(1, al_iters):
        print('al iteration: ', i)
        num_epochs += 100
        np.random.seed(seed)
        torch.manual_seed(seed)
        model = BayesianNet(num_classes=2)
        num_points = i*al_batch_size
        total_data_size = len(dataset1_train)
        spacing = total_data_size // num_points
        if spacing == 0:
            spacing = 1
        dataloader_train_sub = torch.utils.data.DataLoader(Subset(
            dataset1_train, [i for i in range(0, total_data_size, spacing)]),
                                                      batch_size=64, shuffle=True, **kwargs)        
        proportion_correct_train, proportion_correct_test = train_batched(
            model=model, dataloader=dataloader_train_sub,
            dataloader_test=test_loader, lr=0.001, epochs=num_epochs)
        ent1, cross_ent1 = calc_ent_batched(model, dataloader1_unseen, num_models=100)
        ent2, cross_ent2 = calc_ent_batched(model, dataloader2_unseen, num_models=100)
        causal_correct = test_batched(model, dataloader_causal)
        sp_correct = test_batched(model, dataloader_sp)

        log['train_acc'].append(proportion_correct_train)
        log['test_acc'].append(proportion_correct_test)
        log['cross_ent_1'].append(cross_ent1)
        log['cross_ent_2'].append(cross_ent2)
        log['num points'].append(num_points)
        log['ent1'].append(ent1)
        log['ent2'].append(ent2)
        log['causal acc'].append(causal_correct)
        log['sp acc'].append(sp_correct)
        
        print('num points', num_points)
        print('ent env 1 ', ent1)
        print('ent env 2 ', ent2)
        print('xent env 1 ', cross_ent1)
        print('xent env 2 ', cross_ent2)
        print('If causal acc is low, it means sp model is learned')
        print('causal acc', causal_correct)
        print('If sp acc is low, it means causal model is learned')
        print('sp acc', sp_correct)
        print('train', proportion_correct_train)
        print('test', proportion_correct_test)
    plot_dictionary3(log)
    return log

if __name__ == "__main__":
    seed = 0
    log = main(seed)
