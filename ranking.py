import os
import time
import pathlib
from PIL import Image
from argparse import ArgumentParser

import faiss

from src.feature_extraction import MyResnet50, MyVGG16, MyXception, MyEfficient
# from src.dataloader import get_transformation

ACCEPTED_IMAGE_EXTS = ['.jpg', '.png']


def get_image_list(image_root):
    image_root = pathlib.Path(image_root)
    image_list = list()
    for image_path in os.listdir(image_root):
        image_list.append(image_path[:-4])
    image_list = sorted(image_list, key = lambda x: x)
    return image_list

def main():

    parser = ArgumentParser()
    parser.add_argument("--feature_extractor", required=True, type=str, default='Resnet50')
    parser.add_argument("--dataset", required=True, type=str, default='paris')
    parser.add_argument("--top_k", required=False, type=int, default=15)
    parser.add_argument("--crop", required=False, type=bool, default=False)

    print('Ranking .......')
    start = time.time()

    args = parser.parse_args()
    
    if (args.dataset == 'oxford'):
        image_root = './dataset/oxford'
        query_root = './dataset/groundtruth/oxford'
        evaluate_root = './dataset/evaluation/oxford'
        feature_root = './dataset/feature/oxford'
    else:
        image_root = './dataset/paris'
        query_root = './dataset/groundtruth/paris'
        evaluate_root = './dataset/evaluation/paris'
        feature_root = './dataset/feature/paris'

    # Load module feature extraction 
    if (args.feature_extractor == 'Resnet50'):
        extractor = MyResnet50()
    elif (args.feature_extractor == 'VGG16'):
        extractor = MyVGG16()
    elif (args.feature_extractor == 'Xception'):
        extractor = MyXception()
    elif (args.feature_extractor == 'Efficient'):
        extractor = MyEfficient()
    else:
        print("No matching model found")
        return

    img_list = get_image_list(image_root)
    # transform = get_transformation()

    for path_file in os.listdir(query_root):
        if (path_file[-9:-4] == 'query'):
            rank_list = []

            with open(query_root + '/' + path_file, "r") as file:
                img_query, left, top, right, bottom = file.read().split()

            if(args.dataset == 'oxford'):
                img_query = img_query[5:]

            test_image_path = pathlib.Path(image_root + '/' + img_query + '.jpg')
            pil_image = Image.open(test_image_path)
            pil_image = pil_image.convert('RGB')

            path_crop = 'original'
            if (args.crop):
                pil_image=pil_image.crop((float(left), float(top), float(right), float(bottom)))
                path_crop = 'crop'

            feat = extractor.extract_features1(pil_image)

            indexer = faiss.read_index(feature_root + '/' + args.feature_extractor + '.index.bin')

            _, indices = indexer.search(feat, k=args.top_k)  

            for index in indices[0]:
                rank_list.append(str(img_list[index]))

            with open(evaluate_root + '/' + path_crop + '/' + args.feature_extractor + '/' + path_file[:-10] + '.txt', "w") as file:
                file.write("\n".join(rank_list))

    end = time.time()
    print('Finish in ' + str(end - start) + ' seconds')

if __name__ == '__main__':
    main()