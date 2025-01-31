import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from fastai.vision.all import *


# for real domain training data
ds, ds_info = tfds.load('domainnet/real', split='train', with_info = True)

# Extract data and labels from the numpy dataset
data = [item['image'] for item in tfds.as_numpy(ds)]
labels = [item['label'] for item in tfds.as_numpy(ds)]

np.savez('domainnet_real_train_images.npz', *data)
np.save('domainnet_real_train_labels.npy', np.array(labels))

# for real domain test data
ds, ds_info = tfds.load('domainnet/real', split='test', with_info = True)

# Extract data and labels from the numpy dataset
data = [item['image'] for item in tfds.as_numpy(ds)]
labels = [item['label'] for item in tfds.as_numpy(ds)]

np.savez('domainnet_real_test_images.npz', *data)
np.save('domainnet_real_test_labels.npy', np.array(labels))

# for infograph domain training data
ds, ds_info = tfds.load('domainnet/infograph', split='train', with_info = True)

# Extract data and labels from the numpy dataset
data = [item['image'] for item in tfds.as_numpy(ds)]
labels = [item['label'] for item in tfds.as_numpy(ds)]

np.savez('domainnet_infograph_train_images.npz', *data)
np.save('domainnet_infograph_train_labels.npy', np.array(labels))

# for infograph domain test data
ds, ds_info = tfds.load('domainnet/infograph', split='test', with_info = True)

# Extract data and labels from the numpy dataset
data = [item['image'] for item in tfds.as_numpy(ds)]
labels = [item['label'] for item in tfds.as_numpy(ds)]

np.savez('domainnet_infograph_test_images.npz', *data)
np.save('domainnet_infograph_test_labels.npy', np.array(labels))