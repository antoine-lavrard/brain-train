import os
import torch
from time import sleep
os.system('ps aux | grep test_subdomains.py')
N=50   # number of subdomains 
info = 'results saved in a dictionnary'
torch.save({'info' : info}, 'results_dic.pt')
list_test_sets = [ 'metadataset_vgg_flower_test', 'metadataset_cub_test', 'metadataset_aircraft_test', 'metadataset_fungi_test', 'metadataset_dtd_test', 'metadataset_omniglot_test', 'metadataset_mscoco_test', 'metadataset_traffic_signs_test','metadataset_quickdraw_test']
list_test_sets = [ 'metadataset_quickdraw_test']
#list_test_sets = [ 'metadataset_vgg_flower_test']


for test_set in list_test_sets:
    for i in range(N):
        #try:
        print("CLUSTER NUMBER  :", i )
        os.system('python main.py  --epochs 1 --dataset-path "/users2/libre/datasets/"  --load-backbone /users2/libre/clusters/clusters'+str(i)+' --test-dataset ' + test_set + ' --backbone resnet12 --few-shot-queries 0 --few-shot-way 0 --few-shot-shots 0 --few-shot --few-shot-runs 600 --batch-size 128 --save-features-prefix /users2/local/r21lafar/features_test/'+test_set+str(i))
        sleep(3)  # time to exit process
        #except:
        #    print('#########################"" \n did not work for'+test_set+str(i)+'\n ###################')