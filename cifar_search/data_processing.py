import os
import random
import shutil

data_path = '/ubda/home/16904228r/data/Imagenet/train'
# Split images (75%/15%/10%) and save to temporary folders:
for subfolder in os.listdir(data_path):

    # Making a list of all files in current subfolder:
    original_path = f'{data_path}/{subfolder}'
    original_data = os.listdir(original_path)

    # Number of samples in each group:
    n_samples = len(original_data)
    list_of_examples = list(range(n_samples))
    random.shuffle(list_of_examples)
    list_of_examples_train = list_of_examples[:round(len(list_of_examples) * 0.1)]
    # valid_samples = int(n_samples * 0.9)

    train_path = f'/ubda/home/16904228r/data/Imagenet/train_partial/{subfolder}'

    # New class subfolder for training:
    os.chdir('/ubda/home/16904228r/data/Imagenet/train_partial')
    if os.path.exists(subfolder):
        shutil.rmtree(subfolder, ignore_errors=True)
    os.mkdir(subfolder)

    # Training images:
    for image in list_of_examples_train:
        original_file = f'{original_path}/{original_data[image]}'
        new_file = f'{train_path}/{original_data[image]}'
        shutil.copyfile(original_file, new_file)

    # New class subfolder for validation:
    # os.chdir('/ubda/home/16904228r/data/mushroom/valid')
    # os.mkdir(subfolder)

    # Validation images:
    # for image in range(train_samples, valid_samples):
    #     original_file = f'{original_path}/{original_data[image]}'
    #     new_file = f'{valid_path}/{original_data[image]}'
    #     shutil.copyfile(original_file, new_file)

    # New class subfolder for testing:
    # os.chdir('/ubda/home/16904228r/data/mushroom/test')
    # if os.path.exists(subfolder):
    #     shutil.rmtree(subfolder, ignore_errors=True)
    # os.mkdir(subfolder)

    # Test images:
    # for image in range(train_samples, n_samples):
    #     original_file = f'{original_path}/{original_data[image]}'
    #     new_file = f'{test_path}/{original_data[image]}'
    #     shutil.copyfile(original_file, new_file)