from utils import convert

DATA_PATH = './DataSets'
IMAGES_PATH = './Images'
NUM_TRAIN = 5
NUM_TEST = 5

convert.toImages(DATA_PATH, IMAGES_PATH, NUM_TRAIN, NUM_TEST)