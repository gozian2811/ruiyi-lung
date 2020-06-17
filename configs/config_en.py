import warnings
import os
from .config import BasicConfig
class DefaultConfig(BasicConfig):
    model_root = '/data/fyl/models_pytorch'
    model = 'EnsembleNet_Paramwise'
    model_setup = {'pretrained': True}
    connection_mode = 'singleline'
    env = model + '_oa_lidc3s_classification_pretrained_adam_rfold1|10'
    Devices_ID = '6'
    
    dataset = 'Pathology_Slicewise_Dataset'
    label_mode = 'binclass'
    patch_mode = 'oa'
    root_dir = '/data/fyl/data_samples/lidc_3slices'
    #root_dir = model_root + "/BasicNet_lidc_classification_9/filelist_train_fold[0, 1, 2, 3, 4].log"
    patientlist = '/data/fyl/models_pytorch/BaseweightSharedBNNet_classification_pretrained_adam_rfold8|10/patientlist.log'
    filelist_shuffle = False
    num_cross_folds = 10
    val_fold = 1
    test_fold = -1
    remove_uncertain = True
    filelists = dict(
        #train = model_root + '/BasicNet_classification_lidc_test_rfold1/filelist_train_fold[1, 2, 3, 4].log',
        #val = model_root + '/BasicNet_classification_lidc_test_rfold1/filelist_val_fold0.log',
        test = model_root + '/EnsembleNet_Paramwise_oa_lidc3s_classification_pretrained_adam_rfold1|10/filelist_val_fold0.log'
    )
    
    #load_model_path = None
    #load_model_path = model_root + "/ParamResNet50_lidc_classification_pretrained_rfold1|10/ParamResNet50_lidc_classification_pretrained_rfold1|10_epoch22"
    #load_model_path = model_root + '/BasicNet_lidc_classification_9/BasicNet_lidc_classification_9_epoch500'
    #load_model_path = model_root + '/BasicNet_classification_lidc_transfer_rfold5/BasicNet_classification_lidc_transfer_rfold5_epoch1330'
    #load_model_path = model_root + '/EnsembleNet_Paramwise_lidc_classification_adam_rfold1|10/EnsembleNet_Paramwise_lidc_classification_adam_rfold1|10_epoch10'
    #load_model_path = model_root + "/DenseNet_Iterative_trand_fold1/DenseNet_Iterative_trand_fold1_epoch101"
    #load_model_path = model_root + "/ParamDenseCropNet_classification_fold4|5/ParamDenseCropNet_classification_fold4|5_epoch146"
    inference_setup = {'model_name': 'EnsembleNet_Paramwise_oa_lidc3s_classification_pretrained_adam_rfold1|10', 'epochs': [2, 3, 18, 19, 27, 35, 46]}
    use_gpu = True

    input_size = 200
    batch_size = 10
    print_freq = 50
    save_freq = 1
    
    data_preprocess = dict(
        translation_num = 1,
        translation_range = (-10, 10),
        rotation_num = 3,
        flip_num = 2,
	shear_num = 0,
        noise_range = (0, 0)
    )

    #drop_rate = 0
    weight_decay = 0
    loss_exp = 0
    balancing = (0.5, 0.5)

    max_epoch = 50
    lr = 5e-6

'''
further improvements:
1) randomly augment differently among 3 input channels;
2) add drop out, weight decay, or balancing factor;
3) further read implementation details from the original paper.
'''
