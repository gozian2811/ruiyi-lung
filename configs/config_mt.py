import warnings
import os
class DefaultConfig(object):
    model_root = 'models_pytorch'
    model = 'DensecropNet_MultiTask'
    connection_mode = 'singleline'
    env = model + '_regression_expwise_easy_test_4'
    Devices_ID = '1'
    Num_Threads = '1'
    
    dataset = 'LIDC_Pathology_Relation_Dataset'
    label_mode = 'regression'
    root_dir = '../data_samples/lidc_cubes_64_overbound'
    filelist_shuffle = False
    num_cross_folds = 5
    val_fold = 1
    test_fold = -1
    remove_uncertain = True
    #filelist_train = model_root + '/DensecropNet_Iterative_test_3/DensecropNet_Iterative_test_3_step159.log'
    #filelist_val = model_root + '/DensecropNet_Iterative_test/filelist_val_fold0.log'
    filelist_train = '/home/fyl/programs/lung_project/pytorch_project/experiments_cl/DensecropNet_MultiTask_regression_expwise_fold1_epoch27_train/DensecropNet_MultiTask_regression_expwise_fold1_epoch27_easys.log'
    filelist_val = '/home/fyl/programs/lung_project/pytorch_project/experiments_cl/DensecropNet_MultiTask_regression_expwise_fold1_epoch27_val/DensecropNet_MultiTask_regression_expwise_fold1_epoch27_easys.log'
    filelist_test = model_root + '/DensecropNet_MultiTask_regression_expwise_fold1/filelist_train_fold[1, 2, 3, 4].log'
    
    load_model_path = None
    #load_model_path = model_root + '/DensecropNet_Iterative_gr14_agp_fold3/DensecropNet_Iterative_gr14_agp_fold3_epoch165'
    #load_model_path = "models_pytorch/DensecropNet_2_trand_fold1/DensecropNet_2_trand_fold1_epoch13"
    #load_model_path = "models_pytorch/DensecropNet_regression_2_fold1/DensecropNet_regression_2_fold1_epoch141"
    #load_model_path = "models_pytorch/DensecropNet_MultiTask_regression_expwise_fold1/DensecropNet_MultiTask_regression_expwise_fold1_epoch27"
    #load_model_path = "models_pytorch/DensecropNet_MultiTask_regression_fold1/DensecropNet_MultiTask_regression_fold1_epoch27"
    #load_model_path = "models_pytorch/DenseNet_Iterative_trand_fold5/DenseNet_Iterative_trand_fold5_epoch84"
    #load_model_path = "/home/lyj/zyplanet/DensecropNet/checkpoints/<class 'models.models.DensecropNet_Ini'>_0325_01:05:26.pkl"
    input_size = 64
    batch_size = 3
    use_gpu = True
    num_workers = 0
    print_freq = 1
    save_freq = -1
    
    translation_num = 1
    translation_range = (-3, 3)
    rotation_num = 1
    flip_num = 1
    noise_range = (-76, 76)

    max_epoch = 500

    lr = 0.005
    lr_decay_freq_ini = 32
    lr_decay_freq_fin = 45
    lr_decay_ini = 1.0
    lr_decay_fin = 1.0
    
    loss_weightlist = [0.3, 0.3, 0.4]
    loss_exp = 0
    balancing = 0.6
    drop_rate = 0.15
    weight_decay = 7.4e-4

    num_blocks = 3
    growth_rate = 24
    num_final_growth = 1
    average_pool = False
    compression_factor = 0.5
    def parse(self,kwargs):

        for k,v in kwargs.iteritems():
            if not hasattr(self,k):
                warnings.warn('Warning: opt has not attribut %s'%k)
            setattr(self,k,v)

        print('user config:')
        for k,v in self.__class__.__dict__.iteritems():
            if not k.startswith('__'):
                print(k,getattr(self,k))
