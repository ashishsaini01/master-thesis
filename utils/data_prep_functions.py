from fastai.vision.all import *
from fastai.data.load import _FakeLoader, _loaders

def nosplit(o): return L(int(i) for i in range(len(o))), L()

def make_dataloaders_from_numpy_data(image, label=[], batch_size=64,
    num_workers=4, ood=False, train=True):
    def pass_index(idx):
        return idx

    def get_x(i):
        # NOTE: This is a grayscale image that appears to just work with a network expecting RGB.
        # I suspect this is due to tensor broadcasting rules.
        return image[i]
    
    if ood == False:
      def get_y(i):
          return label[i]
      if train:

        dblock = DataBlock(
            blocks=(ImageBlock, CategoryBlock),
            get_items=pass_index,
            splitter=RandomSplitter(valid_pct=0.2, seed=42),
            get_x=get_x,
            get_y=get_y)
      else:
        dblock = DataBlock(
            blocks=(ImageBlock, CategoryBlock),
            get_items=pass_index,
            splitter=nosplit,
            get_x=get_x,
            get_y=get_y)
    else:
      dblock = DataBlock(
          blocks=(ImageBlock),
          get_items=pass_index,
          splitter=RandomSplitter(valid_pct=0.2, seed=42),
          get_x=get_x)

    # pass in a list of index
    num_images = image.shape[0]
    dls = dblock.dataloaders(list(range(num_images)), batch_size=batch_size, shuffle=True,
    num_workers=num_workers, pin_memory=True)

    return dls

class NumpyDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, transform=None):
        self.data = np.load(data_path)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features = self.data[idx]

        # Convert to PIL Image if not already
        if not isinstance(features, Image.Image):
            features = Image.fromarray(features)

        if self.transform:
            features = self.transform(features)

        return features
    

def return_pair_dict_domainnet(images, labels):

    array_images = [images[key] for key in images.keys()]

    #Shuffle the indices
    indices = list(range(len(images)))
    random.shuffle(indices) 

    # Initialize the list to store dictionaries
    image_list_dict = []

    # Iterate over the shuffled indices
    for i in indices:
        image_dict = {}  # Create a new dictionary in each iteration
        image_dict['images'] = array_images[i]
        image_dict['labels'] = labels[i]
        image_list_dict.append(image_dict)
    
    return image_list_dict


#to make mixed dls
def arrayisin(arr, arr_list):
    "Checks if `arr` is in `arr_list`"
    for a in arr_list:
        if np.array_equal(arr, a):
            return True
    return False
class MixedDL():
    def __init__(self, *dls, device='cuda:0'):
        "Accepts any number of `DataLoaders` and a device"
        self.device = device
        for dl in dls: dl.shuffle_fn = self.shuffle_fn
        self.dls = dls
        self.count = 0
        self.fake_l = _FakeLoader(self, False, 0, 0, 0, 0)
        self._get_idxs()

    def __len__(self):
        return min(len(self.dls[0]), len(self.dls[1]))

    def _get_vals(self, x):
        "Checks for duplicates in batches"
        idxs, new_x = [], []
        for i, o in enumerate(x): x[i] = o.cpu().numpy().flatten()
        for idx, o in enumerate(x):
            if not arrayisin(o, new_x):
                idxs.append(idx)
                new_x.append(o)
        return idxs

    def _get_idxs(self):
          "Get `x` and `y` indicies for batches of data"
          dl_dict = dict(zip(range(0,len(self.dls)), [dl.n_inp for dl in self.dls]))
          inps = L([])
          outs = L([])
          for key, n_inp in dl_dict.items():
              b = next(iter(self.dls[key]))
              inps += L(b[:n_inp])
              outs += L(b[n_inp:])
          self.x_idxs = self._get_vals(inps)
          self.y_idxs = self._get_vals(outs)


    def __iter__(self):
        for b in zip(*[_loaders[i.fake_l.num_workers==0](i.fake_l) for i in self.dls]):
            inps = []
            outs = []
            if self.device is not None:
                b = to_device(b, self.device)
            for batch, dl in zip(b, self.dls):
                batch = dl.after_batch(batch)
                inps += batch[:dl.n_inp]
                outs += batch[dl.n_inp:]
            inps = torch.cat(inps, dim=0).to(torch.float32)
            outs = torch.cat(outs, dim=0).to(torch.float32)
            yield (inps, outs)

    def one_batch(self):
        "Grab one batch of data"
        with self.fake_l.no_multiproc(): res = first(self)
        if hasattr(self, 'it'): delattr(self, 'it')
        return res

    def shuffle_fn(self, idxs):
        "Shuffle the internal `DataLoaders`"
        if self.count == 0:
            self.rng = self.dls[0].rng.sample(idxs, len(idxs))
            self.count += 1
            return self.rng
        if self.count == 1:
            self.count = 0
            return self.rng

    def show_batch(self):
        "Show a batch of data"
        for dl in self.dls:
            dl.show_batch()

    def to(self, device):
        self.device = device

