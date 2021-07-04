import os
import torch
import numpy as np 
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score, confusion_matrix
from logger import *
from data_raw import TestDataset
from model import densenet3d

os.environ["CUDA_VISIBLE_DEVICES"]="0, 1"

def results(labels, preds, avg_type, class_type = "other"):
    precision = precision_score(labels, preds, average = avg_type)
    recall = recall_score(labels, preds, average = avg_type)
    f1score = f1_score(labels, preds,average = avg_type)
    print("class {} precision:{} recall:{} f1score:{} ".format(avg_type, precision,recall,f1score))
    log.info("class {} precision:{} recall:{} f1score:{} ".format(avg_type, precision,recall,f1score))
    report = classification_report(labels, preds)
    conf_matrix=confusion_matrix(labels,preds)
    print(f'{report}\n{conf_matrix}\n')
    log.info(f'{report}\n{conf_matrix}\n')
    if class_type == "two":
        return recall

def softmax(x):
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x)
    return softmax_x 

class Prediction:
    def __init__(self, outputs, labels, path_name, patient_id):
        self.outputs = outputs 
        self.labels = labels
        self.path_name = path_name
        self.patient_id = patient_id

    def __eq__(self, other):
        if self.patient_id == other.pateint_id: #and self.age == other.age:
            return True
        else:
            return False

    def __gt__(self, other):
        if self.patient_id != other.patient_id:
            return self.patient_id > other.patient_id
        else:
            return self.path_name > other.path_name


def test(test_data_loader, model, patient_id_list):
    print(len(test_data_loader))
    predictions = []

    for index, (inputs, labels, patient_name, patient_ids) in enumerate(test_data_loader):
        model.eval()
        inputs = inputs.cuda()
        labels = labels.cuda()
        inputs = inputs.unsqueeze(dim=1).float()
        inputs = F.interpolate(inputs, size=[16, 128, 128], mode = "trilinear", align_corners = False)
        outputs = model(inputs)
        labels_array = labels.cpu().numpy()
        outputs_array = outputs.detach().cpu().numpy()
        for index, patient_id in enumerate(patient_ids):
            patient_id = str(patient_id.cpu().numpy().item())
            if patient_id in patient_id_list:
                prediction = Prediction(outputs_array[index], labels_array[index], patient_name[index], patient_id)
                predictions.append(prediction)
    return predictions


def gen_dict(pred): 
    pred_sorted = sorted(pred)
    pred_lists = [[pred_sorted[0]]]
    for i in range(1, len(pred_sorted)):
        cur_info = pred_sorted[i]
        pre_info = pred_sorted[i - 1]
        if cur_info.patient_id != pre_info.patient_id:
            pred_lists.append([cur_info])
        else:
            pred_lists[-1].append(cur_info)
    return pred_lists 


def gen_ids(detail_csv):
    patient_ids = []
    with open(detail_csv, 'r') as fin:
        for line in fin:  
            patient_id, patient_name, gender, age = line.strip().split(',')
            patient_ids.append(patient_id)
    patient_ids.append("119040108765")

    return patient_ids
        

def gen_two_class(preds, labels):
    label_two, pred_two = [], []

    for label in labels:
        if label == 1 or label == 4 or label == 5:
            label_two.append(1)
        else:
            label_two.append(0)

    for pred in preds:
        if pred == 1 or pred == 4 or pred == 5:
            pred_two.append(1)
        else:
            pred_two.append(0)

    return pred_two, label_two


def gen_four_class(preds, labels):
    label_four, pred_four = [], []

    for label in labels:
        if label == 1 or label == 4 or label == 5:
            label_four.append(1)
        else:
            label_four.append(label)

    for pred in preds:
        if pred == 1 or pred == 4 or pred == 5:
            pred_four.append(1)
        else:
            pred_four.append(pred)

    return pred_four, label_four


def test_case(test_data_loader, model):
    patient_ids = gen_ids("./utils/patients_id_test.csv")
    preds = test(test_data_loader, model, patient_ids)
    pred_lists = gen_dict(preds)
    case_preds, case_labels, case_ids, case_path = [], [], [], []
    for case_pred in pred_lists:
        seq_preds = []
        for seq_pred in case_pred:
            seq_preds.append(seq_pred.outputs)
            label = seq_pred.labels
            patient_id = seq_pred.patient_id
            path_name = seq_pred.path_name
       
        mean_pred = np.mean(seq_preds,0)
        type_pred = np.argmax(mean_pred)
        case_preds.append(type_pred)
        case_ids.append(patient_id)
        case_path.append(path_name)
        case_labels.append(label)

    pred_two, label_two = gen_two_class(case_preds, case_labels)
    pred_four, label_four = gen_four_class(case_preds, case_labels)
    pred_six, label_six = case_preds, case_labels
    box_train = zip(case_path, case_ids, label_six, pred_six, label_four, pred_four, label_two, pred_two)
    
    results(label_six, pred_six, avg_type = "macro", class_type = "six")
    results(label_six, pred_six, avg_type = "micro", class_type = "six")
    results(label_four, pred_four, avg_type = "macro", class_type = "four")
    results(label_four, pred_four, avg_type = "micro", class_type = "four")
    recall = results(label_two, pred_two, avg_type = "macro", class_type = "two")
    return recall


if __name__ == "__main__":
    data_test = TestDataset()
    test_data_loader = DataLoader(dataset = data_test, batch_size = 4, shuffle = False, num_workers = 16)
    logfile = "./test.log"
    sys.stdout = Logger(logfile)
    patient_ids = gen_ids("./utils/patients_id_test.csv")
    
    for epoch in [5, 10, 15,20,25,30,35,40,45,55,60,65,70,75,80,85,90,95,100]:
        print("epoch:{}".format(epoch))
        PATH = "xxxx{}.pth".format(epoch) 
        checkpoint = torch.load(PATH)
        model = densenet3d().cuda()
        model = torch.nn.DataParallel(model)
        model.load_state_dict(checkpoint)
        preds = test(test_data_loader, model, patient_ids)
        pred_lists = gen_dict(preds)
        case_preds = []
        case_labels = []
        case_ids = []
        case_path = []
        pred_probs = []
        for case_pred in pred_lists:
            seq_preds = []
            for seq_pred in case_pred:
                seq_preds.append(seq_pred.outputs)
                label = seq_pred.labels
                patient_id = seq_pred.patient_id
                path_name = seq_pred.path_name
            mean_pred = np.mean(seq_preds,0)
            pred_prob = softmax(mean_pred)
            pred_probs.append(pred_prob.tolist())
            type_pred = np.argmax(mean_pred)
            case_preds.append(type_pred)
            case_ids.append(patient_id)
            case_path.append(path_name)
            case_labels.append(label)

        pred_two, label_two = gen_two_class(case_preds, case_labels)
        pred_four, label_four = gen_four_class(case_preds, case_labels)
        pred_six, label_six = case_preds, case_labels
        box_train = zip(case_path, case_ids, label_six, pred_six, label_four, pred_four, label_two, pred_two)

        results(label_six, pred_six, avg_type="macro")
        results(label_six, pred_six, avg_type="micro")
        results(label_four, pred_four, avg_type="macro")
        results(label_four, pred_four, avg_type="micro")
        results(label_two, pred_two, avg_type="macro", class_type = "two")


    


    







    


    






