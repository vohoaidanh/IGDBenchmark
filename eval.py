import os
import csv
import comet_ml
import torch

from validate import validate
from networks.resnet import resnet50
from options.test_options import TestOptions
from eval_config import *

from datetime import datetime
dt = datetime.now().strftime("%Y%m%d%H%M%S")

# Running tests
opt = TestOptions().parse(print_options=False)
######################################################################
comet_params = {
    'CropSize': opt.cropSize,
    'batch_size':opt.batch_size,
    'detect_method':'CNNDetection',
    'noise_type': 'None',
    'model_path': opt.model_path,
    'jpg_qual': opt.jpg_qual,
    'name': 'Run test set with RGB CNNDetection on RealFakeDB512s '
    }

comet_ml.init(api_key='MS89D8M6skI3vIQQvamYwDgEc')
experiment = comet_ml.Experiment(
        project_name="ai-generated-image-detection"
    )
experiment.log_parameter('Cross_test params', comet_params)
######################################################################

model_name = os.path.basename(model_path).replace('.pth', '')
rows = [["{} model testing on...".format(model_name)],
        ['testset', 'accuracy', 'avg precision']]

print("{} model testing on...".format(model_name))
for v_id, val in enumerate(vals):
    opt.dataroot = '{}/{}'.format(dataroot, val)
    opt.classes = ['']#os.listdir(opt.dataroot) if multiclass[v_id] else ['']
    opt.no_resize = True    # testing without resizing by default

    model = resnet50(num_classes=1)
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict['model'])
    model.cuda()
    model.eval()

    acc, ap, conf_mat = validate(model, opt)[:3]
    
    TP = conf_mat[1, 1]
    TN = conf_mat[0, 0]
    FP = conf_mat[0, 1]
    FN = conf_mat[1, 0]
    
    TPR = TP / (TP + FN)
    TNR = TN / (TN + FP)
    
    rows.append([val, acc, TPR, TNR])
    print("({}) acc: {}; TPR: {}, TNR: {}".format(val, acc, TPR, TNR))

    experiment.log_metric('corsstest/acc', acc)
    file_name = "corss_{}_{}.json".format(val, dt)
    experiment.log_confusion_matrix(matrix = conf_mat, file_name=file_name)

experiment.end()

csv_name = results_dir + '/{}.csv'.format(model_name)
with open(csv_name, 'w') as f:
    csv_writer = csv.writer(f, delimiter=',')
    csv_writer.writerows(rows)
