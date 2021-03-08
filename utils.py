import numpy as np
from torch.utils.data.sampler import Sampler

class IdentitySampler(Sampler):
    """Sample person identities evenly in each batch.
        Args:
            train_color_label, train_thermal_label: labels of each modalities
            color_pos, thermal_pos: positions of each identity
            batch_num_identities: batch size
    """
    def __init__(self, train_color_label, train_thermal_label, color_pos, thermal_pos, num_of_same_id_in_batch, batch_num_identities, dataset):
        uni_label = np.unique(train_color_label)
        self.n_classes = len(uni_label)
        N = np.maximum(len(train_color_label), len(train_thermal_label))
        # Doing as much batch as we can divide the dataset in number of batch
        for j in range(int(N / (batch_num_identities * num_of_same_id_in_batch)) + 1):
            # We choose randomly 8 identities
            batch_idx = np.random.choice(uni_label, batch_num_identities, replace=False)
            # print(f"batch idx {batch_idx}")

            for i in range(batch_num_identities):
                # We choose randomly 4 images (num_of_same_id_in_batch) for the i=8 identitities
                if dataset == "Tworld" or dataset == "RegDB":
                    sample_color = np.random.choice(color_pos[batch_idx[i]], num_of_same_id_in_batch)
                    sample_thermal = sample_color
                elif dataset == "SYSU" :
                    sample_color = np.random.choice(color_pos[batch_idx[i]], num_of_same_id_in_batch)
                    sample_thermal = np.random.choice(thermal_pos[batch_idx[i]], num_of_same_id_in_batch)
                if j == 0 and i == 0:
                    index1 = sample_color
                    index2 = sample_thermal
                else:
                    index1 = np.hstack((index1, sample_color))
                    index2 = np.hstack((index2, sample_thermal))
        # print(f"index1 : {index1}")
        # print(f"index2 : {index2}")
        self.index1 = index1
        self.index2 = index2
        self.N = N

    def __iter__(self):
        return iter(np.arange(len(self.index1)))

    def __len__(self):
        return self.N


class IdentityFeatureSampler(Sampler):
    """Sample person identities evenly in each batch.
        Args:
            train_color_label, train_thermal_label: labels of each modalities
            color_pos, thermal_pos: positions of each identity
            batch_num_identities: batch size
    """

    def __init__(self, train_features_label, features_pos, num_of_same_id_in_batch, batch_num_identities, dataset):
        uni_label = np.unique(train_features_label)
        self.n_classes = len(uni_label)
        N = len(train_features_label)
        # Doing as much batch as we can divide the dataset in number of batch
        for j in range(int(N / (batch_num_identities * num_of_same_id_in_batch)) + 1):
            # We choose randomly 8 identities
            batch_idx = np.random.choice(uni_label, batch_num_identities, replace=False)
            # print(f"batch idx {batch_idx}")

            for i in range(batch_num_identities):
                # We choose randomly num_of_same_id_in_batch=4 concatenated features  for the i(batch_num_identities)=8 identitities
                if dataset == "Tworld" or dataset == "RegDB":
                    sample_features = np.random.choice(features_pos[batch_idx[i]], num_of_same_id_in_batch)
                if j == 0 and i == 0:
                    # This way in a batch we compare first feature id with all 4*8 features (Need to verify the type of index 1 and 2 here.
                    index1 = [sample_features[0] for w in range(num_of_same_id_in_batch)]

                    index2 = sample_features
                else:
                    index1 = np.hstack((index1, [index1[0] for w in range(len(sample_features))]))
                    index2 = np.hstack((index2, sample_features))

        print(f"index1 : {index1}")
        print(f"index2 : {index2}")
        print(f"index1 info: {index1[:4]}")
        print(f"index2 info: {index2[:4]}")
        print(f"index1 info: {index1[0]")
        print(f"index2 info: {index2[3]}")
        self.index1 = index1
        self.index2 = index2
        self.N = N

    def __iter__(self):
        return iter(np.arange(len(self.index1)))

    def __len__(self):
        return self.N



class AverageMeter(object):
    """Computes and stores the average and current value"""

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

#Tests avec un lr diminuant moins
def adjust_learning_rate(optimizer, epoch, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if epoch < 10:
        lr = lr * (epoch + 1) / 10
    elif epoch >= 10 and epoch < 20:
        lr = lr
    elif epoch >= 20 and epoch < 50:
        lr = lr * 0.1
    elif epoch >= 50:
        lr = lr * 0.01

    optimizer.param_groups[0]['lr'] = 0.1 * lr
    for i in range(len(optimizer.param_groups) - 1):
        optimizer.param_groups[i + 1]['lr'] = lr
    return lr
