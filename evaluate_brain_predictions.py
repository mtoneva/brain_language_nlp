import argparse
import numpy as np
import pickle as pk
import time as tm

from utils.utils import binary_classify_neighborhoods, CV_ind


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--subject", default='')
    args = parser.parse_args()
    print(args)

    start_time = tm.time()

    loaded = np.load(args.input_path, allow_pickle=True)
    preds_t_per_feat = loaded.item()['preds_t']
    test_t_per_feat = loaded.item()['test_t']
    print(test_t_per_feat.shape)
    
    n_class = 20   # how many predictions to classify at the same time
    n_folds = 4
    
    neighborhoods = np.load('./data/voxel_neighborhoods/' + args.subject + '_ars_auto2.npy')
    n_words, n_voxels = test_t_per_feat.shape
    ind = CV_ind(n_words, n_folds=n_folds)

    accs = np.zeros([n_folds,n_voxels])
    acc_std = np.zeros([n_folds,n_voxels])

    for ind_num in range(n_folds):
        test_ind = ind==ind_num
        accs[ind_num,:],_,_,_ = binary_classify_neighborhoods(preds_t_per_feat[test_ind,:], test_t_per_feat[test_ind,:], n_class=20, nSample = 1000,pair_samples = [],neighborhoods=neighborhoods)


    fname = args.output_path
    if n_class < 20:
        fname = fname + '_{}v{}_'.format(n_class,n_class)

    with open(fname + '_accs.pkl','wb') as fout:
        pk.dump(accs,fout)

    print('saved: {}'.format(fname + '_accs.pkl'))


    
