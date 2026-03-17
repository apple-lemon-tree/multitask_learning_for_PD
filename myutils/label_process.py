from collections import Counter

def multilabel_process(p_labels):

    count1 = Counter(tuple(label) for label in p_labels)

    count2 = sorted(count1, key=lambda x:count1[x], reverse=True)

    label_to_ids = {
        key:index
        for index,key in enumerate(count2)
    }

    new_labels = [label_to_ids[tuple(index)] for index in p_labels]

    return new_labels