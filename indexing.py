import time
from argparse import ArgumentParser
import numpy as np

import faiss

from src.feature_extraction import MyResnet50, MyVGG16, MyXception
from src.indexing_faiss import get_faiss_indexer
from src.dataloader import MyDataLoader

feature_root = './dataset/feature'
def main():

    parser = ArgumentParser()
    parser.add_argument("--feature_extractor", required=True, type=str, default='Resnet50')
    parser.add_argument("--dataset", required=True, type=str, default='oxford')

    print('Start indexing .......')
    start = time.time()

    args = parser.parse_args()
    if (args.dataset == 'oxford'):
        image_root = './dataset/oxford'
    else:
        image_root = './dataset/paris'

    # Load module feature extraction 
    if (args.feature_extractor == 'Resnet50'):
        extractor = MyResnet50()
    elif (args.feature_extractor == 'VGG16'):
        extractor = MyVGG16()
    elif (args.feature_extractor == 'Xception'):
        extractor = MyXception()
    else:
        print("No matching model found")
        return

    dataset = MyDataLoader(image_root)
    index = get_faiss_indexer(extractor.shape)

    for images in dataset.image_list:
        features = extractor.extract_features(images)
        index.add(features)
    
    # Save features
    if (args.dataset == 'oxford'):
        faiss.write_index(index, feature_root + '/oxford/' + args.feature_extractor + '.index.bin')
    else:
        faiss.write_index(index, feature_root + '/paris/' + args.feature_extractor + '.index.bin')
    
    end = time.time()
    print('Finish in ' + str(end - start) + ' seconds')

if __name__ == '__main__':
    main()