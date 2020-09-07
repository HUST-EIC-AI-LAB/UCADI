import torch
import numpy as np
# import data
from data_raw import TestDataset
from model import densenet3d
import csv
import torch.nn.functional as F
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score, confusion_matrix
from logger import *
import os
from scipy.special import softmax
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"


def calculate_accuracy(outputs, targets):
    batch_size = targets.size(0)

    _, pred = outputs.topk(1, 1, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1))
    n_correct_elems = correct.float().sum().item()

    return n_correct_elems / batch_size

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
    print('multi test \n')
    predict_value = []
    names_box_test = []

    patient_id_test=np.asarray(patient_id_list)
    print('patient_id shape',patient_id_test.shape)
    pred_two, label_two, pred_four, label_four = [], [], [], []
    ## add five models
    predictions_allmodels=[]
    all_output_value=[]

    for model_index in range(5):
        predictions = []
        epoch_output_value=[]
        print(len(test_data_loader))
        for index, (inputs, labels, patient_name, patient_ids) in enumerate(test_data_loader):
            model.eval()
            sequence_inputs=inputs
            with torch.no_grad():
                inputs = inputs.cuda()
                labels = labels.cuda()
                # forward
                inputs = inputs.unsqueeze(dim=1).float()
                inputs = F.interpolate(inputs, size=[16, 128, 128], mode = "trilinear", align_corners = False)
                outputs = model(inputs)
                acc=calculate_accuracy(outputs,labels)
                # pred_six = torch.max(outputs, 1)[1].cpu().numpy().squeeze()
                labels_array = labels.cpu().numpy()
                outputs_array = outputs.detach().cpu().numpy()
                outputs_array=softmax(outputs_array,axis=1)
                for oo in outputs_array:
                    epoch_output_value.append(oo)
                acc=calculate_accuracy(outputs,labels)
                print('batch: {} acc: {}'.format(index+1,acc))
                log.info('batch: {} acc: {}'.format(index+1,acc))
                # import ipdb; ipdb.set_trace()
            for index, patient_id in enumerate(patient_ids):
                patient_id = str(patient_id.cpu().numpy().item())
                if patient_id in patient_id_list:
                    prediction = Prediction(outputs_array[index], labels_array[index], patient_name[index], patient_id)
                    predictions.append(prediction)
        predictions_allmodels.append(predictions)
        all_output_value.append(epoch_output_value)

    all_output_value=np.asarray(all_output_value).mean(0)
    print('all output value shape',all_output_value.shape)
    with open('benbu_output.csv','w') as file:
        csvwriter=csv.writer(file)
        csvwriter.writerows(all_output_value)
    #import ipdb;ipdb.set_trace()
    npa_predictions=np.asarray(predictions_allmodels)
    predictions_final=[]
    print('npa_predictions shape',npa_predictions.shape) # 5x819
    all_output=[]
    #import ipdb; ipdb.set_trace()
    for index in range(npa_predictions.shape[1]):
        npa_predict=npa_predictions[:,index]
        outputs_value=[]
        for predict_value in npa_predict:
            outputs_value.append(predict_value.outputs)
        outputs_value=np.asarray(outputs_value)
        outputs_value=outputs_value.mean(0)
        all_output.append(outputs_value)


        prediction=Prediction(outputs_value,npa_predict[0].labels,npa_predict[0].path_name,npa_predict[0].patient_id)
        predictions_final.append(prediction)

    return predictions_final # 819



    #return predictions


def gen_dict(pred):
    # import ipdb; ipdb.set_trace()
    pred_sorted = sorted(pred)
    pred_lists = [[pred_sorted[0]]]
    for i in range(1, len(pred_sorted)):
        cur_info = pred_sorted[i]
        pre_info = pred_sorted[i - 1]
        # import ipdb;ipdb.set_trace()
        if cur_info.patient_id != pre_info.patient_id:
            pred_lists.append([cur_info])
        else:
            pred_lists[-1].append(cur_info)

    # import ipdb;ipdb.set_trace()
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
    # import ipdb; ipdb.set_trace()

    return pred_four, label_four


def test_case(test_data_loader, model):
    # patient_ids = gen_ids("./utils/patients_id_test.csv")
    patient_ids = []
    patient_names = []
    with open("./utils/test_norm.csv", "r") as fin:
        for line in fin:
            # path_name, four_label, six_label, patient_id, patient_name, gender, age = line.strip().split(',')
            name, four_label, six_label, patient_id, patient_name, gender, age, score, most_score = line.strip().split(',')
            if patient_name not in patient_names:
                patient_names.append(patient_name)
                patient_ids.append(patient_id)

    preds = test(test_data_loader, model, patient_ids) # 819
    pred_lists = gen_dict(preds)
    case_preds, case_labels, case_ids, case_path = [], [], [], []
    for case_pred in pred_lists:
        seq_preds = []
        for seq_pred in case_pred:
            seq_preds.append(seq_pred.outputs)
            label = seq_pred.labels
            patient_id = seq_pred.patient_id
            path_name = seq_pred.path_name
            data_shape=seq_pred.shape

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

    # with open('case_test_six_{}.csv'.format(epoch),'w') as result_file:
    #     wr = csv.writer(result_file)
    #     for row in box_train:
    #         wr.writerow(row)
    results(label_six, pred_six, avg_type = "macro", class_type = "six")
    results(label_six, pred_six, avg_type = "micro", class_type = "six")
    results(label_four, pred_four, avg_type = "macro", class_type = "four")
    results(label_four, pred_four, avg_type = "micro", class_type = "four")
    recall = results(label_two, pred_two, avg_type = "macro", class_type = "two")
    return recall


if __name__ == "__main__":
    data_test = TestDataset()

    test_data_loader = DataLoader(dataset = data_test, batch_size = 100, shuffle = False, num_workers = 48)
    #test_data_loader = DataLoader(dataset = data_test, batch_size = 80, shuffle = False, num_workers = 48)
    logfile = "./test_all.log"
    sys.stdout = Logger(logfile)
    patient_ids = gen_ids("./utils/patients_id_test.csv")
    for epoch in [5,10,15,20,25,30,35,40,45,50,55]:#[10,15,20,25,30,35,40,45]:#[30, 35, 40]:
        print("epoch:{}".format(epoch))
        PATH = "/home/fzw/COVID-19-Fedrated-Learning-Framework/client_cambridge/model/Cambridge_current_{}.pth".format(epoch)
        #PATH='./checkpoint_32/weight_v{}.pth'.format(epoch)
        # sets = parse_opts()
        # sets.phase = "test"
        checkpoint = torch.load(PATH)
        model = densenet3d().cuda()
        model=DataParallel(model)
        model.load_state_dict(checkpoint)
        preds = test(test_data_loader, model, patient_ids)
        pred_lists = gen_dict(preds)
        case_preds = []
        case_labels = []
        case_ids = []
        case_path = []
        output_sequence=[]
        output_mean=[]
        for case_pred in pred_lists:
            seq_preds = []
            for seq_pred in case_pred:
                seq_preds.append(seq_pred.outputs)
                sequence_output=np.argmax(seq_pred.outputs)
                label = seq_pred.labels
                patient_id = seq_pred.patient_id
                path_name = seq_pred.path_name
                output_sequence.append([seq_pred.outputs,sequence_output,label,patient_id,path_name])
                print('output_sequence',output_sequence)

            # import ipdb;ipdb.set_trace()

            mean_pred = np.mean(seq_preds,0)
            type_pred = np.argmax(mean_pred)
            #import ipdb;ipdb.set_trace()
            case_preds.append(type_pred)
            case_ids.append(patient_id)
            case_path.append(path_name)
            case_labels.append(label)
            output_mean.append([mean_pred,type_pred,label,patient_id,path_name])
            print('output mean',output_mean)


        with open('output_sequence5_softmax.csv','w')as file:
            spamWriter=csv.writer(file)
            spamWriter.writerows(output_sequence)
        with open('output_mean4_softmax.csv','w') as file:
            spamWriter=csv.writer(file)
            spamWriter.writerows(output_mean)
        pred_two, label_two = gen_two_class(case_preds, case_labels)
        pred_four, label_four = gen_four_class(case_preds, case_labels)
        pred_six, label_six = case_preds, case_labels
        box_train = zip(case_path, case_ids, label_six, pred_six, label_four, pred_four, label_two, pred_two)

        with open('case_test_four_norm{}.csv'.format(epoch),'w') as result_file:
            wr = csv.writer(result_file)
            for row in box_train:
                wr.writerow(row)
        results(label_six, pred_six, avg_type="macro")
        results(label_six, pred_six, avg_type="micro")
        results(label_four, pred_four, avg_type="macro")
        results(label_four, pred_four, avg_type="micro")
        results(label_two, pred_two, avg_type="macro", class_type = "two")












