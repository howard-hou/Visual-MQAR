from torch.utils.data import DataLoader


class VisualMQARDataset(DataLoader):

    def __init__(self, dataset, batch_size, shuffle=True):
        super(VisualMQARDataset, self).__init__(dataset, batch_size, shuffle)
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle




