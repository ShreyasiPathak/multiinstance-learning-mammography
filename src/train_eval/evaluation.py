import itertools
import numpy as np
import openpyxl as op
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from utilities import utils

def results_store_excel(train_res, val_res, test_res, per_model_metrics, correct_train, total_images_train, train_loss, correct_test, total_images_test, test_loss, epoch, conf_mat_train, conf_mat_test, lr, auc_val, path_to_results, path_to_results_text):
    lines = [epoch+1, lr]
    if train_res:
        avg_train_loss=train_loss/total_images_train
        accuracy_train=correct_train / total_images_train
        speci_train=conf_mat_train[0,0]/sum(conf_mat_train[0,:])
        recall_train=conf_mat_train[1,1]/sum(conf_mat_train[1,:])
        prec_train=conf_mat_train[1,1]/sum(conf_mat_train[:,1])
        f1_train=2*recall_train*prec_train/(recall_train+prec_train)
        prec_train_neg=conf_mat_train[0,0]/sum(conf_mat_train[:,0])
        recall_train_neg=conf_mat_train[0,0]/sum(conf_mat_train[0,:])
        f1_train_neg=2*recall_train_neg*prec_train_neg/(recall_train_neg+prec_train_neg)
        f1macro_train=(f1_train+f1_train_neg)/2
        lines.extend([avg_train_loss, accuracy_train, f1macro_train, recall_train, speci_train])

    if val_res:
        speci_test=conf_mat_test[0,0]/sum(conf_mat_test[0,:])
        avg_test_loss=test_loss/total_images_test
        recall_test=conf_mat_test[1,1]/sum(conf_mat_test[1,:])
        prec_test=conf_mat_test[1,1]/sum(conf_mat_test[:,1])
        f1_test=2*recall_test*prec_test/(recall_test+prec_test)
        accuracy_test=correct_test / total_images_test
        recall_test_neg=conf_mat_test[0,0]/sum(conf_mat_test[0,:])
        prec_test_neg=conf_mat_test[0,0]/sum(conf_mat_test[:,0])
        f1_test_neg=2*recall_test_neg*prec_test_neg/(recall_test_neg+prec_test_neg)
        f1macro_test=(f1_test+f1_test_neg)/2
        lines.extend([avg_test_loss, accuracy_test, f1macro_test, recall_test, speci_test, auc_val])

    if test_res:
        lines.extend(per_model_metrics)
    
    out = open(path_to_results_text,'a')
    out.write(str(lines)+'\n')
    out.close()
    write_results_xlsx(lines, path_to_results, 'train_val_results')

def results_plot(df, file_name):
    plt.plot(df['Epoch'],df['F1macro Train'],'-r',label='Train F1macro')
    plt.plot(df['Epoch'],df['F1macro Val'],'-b',label='Val F1macro')
    plt.plot(df['Epoch'],df['Avg Loss Train'],'-g',label='Train Loss')
    plt.plot(df['Epoch'],df['Avg Loss Val'],'-y',label='Val Loss')
    plt.legend(loc='upper left')
    plt.xticks(np.arange(1,df.iloc[-1]['Epoch']))
    plt.xlabel('Epochs')
    plt.ylabel('Loss/F1macro')
    plt.title('Learning curve')
    plt.savefig(file_name, bbox_inches='tight')

def conf_mat_create(predicted, true, correct, total_images, conf_mat, classes):
    total_images+=true.size()[0]
    correct += predicted.eq(true.view_as(predicted)).sum().item()
    conf_mat_batch=confusion_matrix(true.cpu().numpy(),predicted.cpu().numpy(),labels=classes)
    conf_mat=conf_mat+conf_mat_batch
    return correct, total_images, conf_mat, conf_mat_batch

def write_results_xlsx(results, path_to_results, sheetname):
    wb = op.load_workbook(path_to_results)
    sheet = wb[sheetname]
    sheet.append(results)
    wb.save(path_to_results)

def write_results_xlsx_confmat(results, path_to_results, sheetname):
    wb = op.load_workbook(path_to_results)
    sheet = wb[sheetname]
    sheet.append([0,1])
    for row in results.tolist():
        sheet.append(row)
    wb.save(path_to_results)

def save_viewwise_count(views_names, test_batch, conf_mat_batch, batch_test_no):
    if batch_test_no==0:
        count_dic_viewwise={}
        conf_mat_viewwise={}
        
    views_names_key="+".join(views_names)
    #loss_dic_viewwise[views_names_key]=loss_dic_viewwise.get(views_names_key,0)+test_labels.size()[0]*loss1
    count_dic_viewwise[views_names_key]=count_dic_viewwise.get(views_names_key,0)+test_batch.shape[0]
    conf_mat_viewwise[views_names_key]=conf_mat_viewwise.get(views_names_key,np.zeros((2,2)))+conf_mat_batch
    count_dic_viewwise[str(len(views_names))]=count_dic_viewwise.get(str(len(views_names)),0)+test_batch.shape[0]
    conf_mat_viewwise[str(len(views_names))]=conf_mat_viewwise.get(str(len(views_names)),np.zeros((2,2)))+conf_mat_batch
    return count_dic_viewwise, conf_mat_viewwise

def calc_viewwise_metric(count_dic_viewwise, conf_mat_viewwise):
    val_stats_viewwise={}
    for key in count_dic_viewwise.keys():
        if count_dic_viewwise[key]:
            count_key=count_dic_viewwise[key]
        else:
            count_key=0
        if sum(conf_mat_viewwise[key][1,:]):
            tpr=float(conf_mat_viewwise[key][1,1])/sum(conf_mat_viewwise[key][1,:])
        else:
            tpr=0.0
        if sum(conf_mat_viewwise[key][0,:]):
            tnr=float(conf_mat_viewwise[key][0,0])/sum(conf_mat_viewwise[key][0,:])
        else:
            tnr=0.0
        if np.sum(conf_mat_viewwise[key]):
            acc=float(conf_mat_viewwise[key][0,0]+conf_mat_viewwise[key][1,1])/np.sum(conf_mat_viewwise[key])
        else:
            acc=0.0
        val_stats_viewwise[key]=[count_key,tpr,tnr,acc]
        #print(key, 'benign', sum(conf_mat_viewwise[key][0,:]))
        #print(key, 'malignant', sum(conf_mat_viewwise[key][1,:]))
    return val_stats_viewwise

def results_viewwise(sheet4, val_stats_viewwise):
    header=[]
    if val_stats_viewwise!={}:
        for key in val_stats_viewwise.keys():
            header=header+[key,'','','']
        sheet4.append(header)
        sheet4.append(['Count','Recall','Specificity','Acc']*len(list(val_stats_viewwise.keys())))
    row_sheet4=list(itertools.chain(*val_stats_viewwise.values()))
    sheet4.append(row_sheet4)
    return sheet4

def case_label_from_SIL(df_test, test_labels_all, test_pred_all, sheet3):
    dic_true={}
    dic_pred={}
    idx=0
    
    test_pred_all = test_pred_all.reshape(-1)
    print(test_labels_all)
    print(test_pred_all)
    for idx in df_test.index:
        dic_key = '_'.join(df_test.loc[idx,'ImageName'].split('_')[:3])
        dic_true[dic_key] = max(dic_true.get(dic_key,0),test_labels_all[idx])
        dic_pred[dic_key] = max(dic_pred.get(dic_key,0),test_pred_all[idx])
    case_labels_true = np.array(list(dic_true.values()))
    case_labels_pred = np.array(list(dic_pred.values()))

    print(case_labels_true)
    print(case_labels_pred)
    metrics_case_labels = utils.performance_metrics(None, case_labels_true, case_labels_pred, None)
    print(metrics_case_labels)
    sheet3.append(['Precision','Recall','Specificity','F1','F1macro','F1wtmacro','Acc','Bal_Acc','Cohens Kappa','AUC'])
    sheet3.append(metrics_case_labels)
    return sheet3
