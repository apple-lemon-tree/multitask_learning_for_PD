import numpy as np


def calculate_per_dim_accuracy(labels,predictions):
    labels_np = np.array(labels)
    predictions_np = np.array(predictions)

    correct = (labels_np == predictions_np).sum(axis=0)
    total = labels_np.shape[0]
    per_dim_accuracy = (correct / total).tolist()


    ext_acc, neu_acc,agr_acc,con_acc,ope_acc = per_dim_accuracy[0], per_dim_accuracy[1], per_dim_accuracy[2], per_dim_accuracy[3], per_dim_accuracy[4]

    return ext_acc, neu_acc,agr_acc,con_acc,ope_acc

def calculate_mbti_accuracy(labels,predictions):
    labels_np = np.array(labels)
    predictions_np = np.array(predictions)

    correct = (labels_np == predictions_np).sum(axis=0)
    total = labels_np.shape[0]
    per_dim_accuracy = (correct / total).tolist()


    I_E,N_S,T_F,P_J = per_dim_accuracy[0], per_dim_accuracy[1], per_dim_accuracy[2], per_dim_accuracy[3]

    return I_E,N_S,T_F,P_J