from utils import convert

TRAIN_SET = './DataSets/MNIST/processed/training.pt'
TEST_SET = './DataSets/MNIST/processed/test.pt'
SAVE_PATH = './Images/train'
NUM_TRAIN = 5
NUM_TEST = 5

convert.toImages(TRAIN_SET, SAVE_PATH, NUM_TRAIN)