import torch
import numpy as np
from networks.resnet import resnet50
from sklearn.metrics import average_precision_score, accuracy_score, confusion_matrix
from tqdm import tqdm
from options.test_options import TestOptions
from data import create_dataloader


def validate(model, opt):
    if opt.detect_method.lower() in ['cnndetection', 'dire']:
        return validate_cnndetection(model, opt)
    
    if opt.detect_method.lower() in ['shading']:
        return validate_shading(model, opt)
    
    if opt.detect_method.lower() in ['fredect']:
        return validate_freedect(model, opt)

def validate_cnndetection(model, opt):
    data_loader = create_dataloader(opt)

    with torch.no_grad():
        y_true, y_pred = [], []
        for img, label in data_loader:
            in_tens = img.cuda()
            y_pred.extend(model(in_tens).sigmoid().flatten().tolist())
            y_true.extend(label.flatten().tolist())

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    r_acc = accuracy_score(y_true[y_true==0], y_pred[y_true==0] > 0.5)
    f_acc = accuracy_score(y_true[y_true==1], y_pred[y_true==1] > 0.5)
    acc = accuracy_score(y_true, y_pred > 0.5)
    ap = average_precision_score(y_true, y_pred)
    conf_mat = confusion_matrix(y_true, y_pred > 0.5)
    
    return acc, ap, conf_mat, r_acc, f_acc, y_true, y_pred

def validate_shading(model, opt):
    data_loader = create_dataloader(opt, opt.val_split)

    with torch.no_grad():
        y_true, y_pred = [], []
        for img, shading, label in tqdm(data_loader):
            in_tens = img.cuda()
            in_tens_shading = shading.cuda()
            y_pred.extend(model(in_tens,in_tens_shading).sigmoid().flatten().tolist())
            y_true.extend(label.flatten().tolist())

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    r_acc = accuracy_score(y_true[y_true==0], y_pred[y_true==0] > 0.5)
    f_acc = accuracy_score(y_true[y_true==1], y_pred[y_true==1] > 0.5)
    acc = accuracy_score(y_true, y_pred > 0.5)
    ap = average_precision_score(y_true, y_pred)
    conf_mat = confusion_matrix(y_true, y_pred > 0.5)
    
    return acc, ap, conf_mat, r_acc, f_acc, y_true, y_pred

def validate_freedect(model, opt):
    data_loader = create_dataloader(opt, opt.val_split)

    with torch.no_grad():
        y_true, y_pred = [], []
        for img, label in data_loader:
            in_tens = img.cuda()
            y_pred.extend(model(in_tens).sigmoid().flatten().tolist())
            y_true.extend(label.flatten().tolist())

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    r_acc = accuracy_score(y_true[y_true==0], y_pred[y_true==0] > 0.5)
    f_acc = accuracy_score(y_true[y_true==1], y_pred[y_true==1] > 0.5)
    acc = accuracy_score(y_true, y_pred > 0.5)
    ap = average_precision_score(y_true, y_pred)
    conf_mat = confusion_matrix(y_true, y_pred > 0.5)
    
    return acc, ap, conf_mat, r_acc, f_acc, y_true, y_pred

if __name__ == '__main__':
    opt = TestOptions().parse(print_options=False)

    model = resnet50(num_classes=1)
    state_dict = torch.load(opt.model_path, map_location='cpu')
    model.load_state_dict(state_dict['model'])
    model.cuda()
    model.eval()

    acc, avg_precision, r_acc, f_acc, y_true, y_pred = validate(model, opt)

    print("accuracy:", acc)
    print("average precision:", avg_precision)

    print("accuracy of real images:", r_acc)
    print("accuracy of fake images:", f_acc)
