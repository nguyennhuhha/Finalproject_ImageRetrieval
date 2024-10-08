import os


def compute_AP(pos_set, ranked_list):
    relevant = 0.0
    total_precision = 0.0
    number_retrieve = 0

    for item in ranked_list:
        number_retrieve += 1
        # rel(k): if the retrived image is flase => rel(k)=0, pass; if it's true => rel(k)=1
        if item not in pos_set: 
            continue
        
        relevant += 1
        total_precision += (relevant/number_retrieve)

    if relevant == 0:
        return 0
    return total_precision / relevant

def compute_mAP(feature_extractor, dataset, crop = False):

    if (dataset == 'oxford'):
        root_groundtruth = './dataset/groundtruth/oxford'
        root_evaluation = './dataset/evaluation/oxford'
    else:
        root_groundtruth = './dataset/groundtruth/paris'
        root_evaluation = './dataset/evaluation/paris'

    if (crop):
        path_evaluation =  root_evaluation + '/crop'
    else:
        path_evaluation = root_evaluation + '/original'

    path_evaluation += ('/' + feature_extractor)

    AP = 0.0
    number_query = 0.0

    for query in os.listdir(path_evaluation):
        with open(root_groundtruth + '/' + query[:-4] + '_good.txt', 'r') as file:
            good_set = file.read().split('\n')
        with open(root_groundtruth + '/' + query[:-4] + '_ok.txt', 'r') as file:
            ok_set = file.read().split('\n')
            
        # positive set of ground truth = ok_set + good_set
        pos_set = ok_set + good_set

        with open(path_evaluation + '/' + query) as file:
            ranked_list = file.read().split('\n')
        
        AP += compute_AP(pos_set, ranked_list)
        number_query += 1
    
    return AP / number_query