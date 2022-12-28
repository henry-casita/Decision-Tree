import numpy as np
import copy
from math import trunc
import matplotlib.pyplot as plt

def entropy(dataset): #Given a dataset, compute the entropy of the labels
    unused_unique_labels, counts = np.unique(dataset[:,-1], return_counts=True) #returns an array of unique labels, and an array of the counts of each label
    probabilities = counts/np.sum(counts) #using the counts of each lable and the total count, we can compute the probabilities of each label
    informations = -np.log2(probabilities) #computes I(x)
    return np.sum(probabilities*informations) #computes the entropy Σ pk * I(pk)

def remainder(left_set, right_set): #Given 2 sets, compute the remainder according to the lecture formula
    left_samples = len(left_set)
    right_samples = len(right_set)
    total_samples = left_samples+right_samples
    return left_samples*entropy(left_set)/total_samples + right_samples*entropy(right_set)/total_samples

def information_gain(full_set, left_set, right_set):
    return entropy(full_set) - remainder(left_set, right_set)

def find_split(dataset):
    max_info_gain = 0
    split_value = 0
    split_attribute = -1
    for i, instance in enumerate(dataset[:,:-1].transpose()):
        sorted_data = dataset[np.argsort(instance.transpose())]
        unused, indices = np.unique(sorted_data[:,i], return_index=True)
        for j in indices:
                info_gain = information_gain(sorted_data, sorted_data[:j], sorted_data[j:])
                if info_gain > max_info_gain:
                    max_info_gain = info_gain
                    split_value = (sorted_data[j-1,i]+sorted_data[j, i])/2
                    split_attribute = i
    return split_value, split_attribute

def decision_tree_learning(training_dataset, depth):
    assert(len(training_dataset>0))
    labels = training_dataset[:, -1]
    if len(np.unique(labels))==1:
        return {"attribute": np.unique(labels)[0], "value":len(training_dataset), "left":None, "right":None,"leaf":True}, depth
    else:
        split, attribute = find_split(training_dataset)
        node_left, l_depth = decision_tree_learning(training_dataset[training_dataset[:, attribute]>split], depth+1)
        node_right, r_depth = decision_tree_learning(training_dataset[training_dataset[:, attribute]<=split], depth+1)
        node = {"attribute": attribute, "value":split, "left":node_left, "right":node_right,"leaf":False}
        return node, max(l_depth, r_depth)

def split_dataset(x, y, k, random_generator=np.random.default_rng()):
    #x = attributes
    #y = labels
    up_confusion_matrix = [] #up stands for unpruned
    up_accuracy = []
    up_precision = []
    up_recall = []
    up_f1 = []
    up_depth = []
    testing_number = len(x)/k #divides the total length of the dataset by k folds.
    shuffled_indices = random_generator.permutation(len(x)) #randomly shuffle the matrix
    for num in range (0,k): 
        if num == 0: #This accounts for the first iteration of the fold cross validation
            x_train = x[shuffled_indices[:int(len(x)-testing_number)]]
            y_train = y[shuffled_indices[:int(len(x)-testing_number)]]
            x_test = x[shuffled_indices[int(len(x)-testing_number):]]
            y_test = y[shuffled_indices[int(len(x)-testing_number):]]       

        else: #This accounts for every other iteration of the fold cross validation
            train1_end = int(len(x) - testing_number*(num+1))
            train2_start = int(2000 - num*testing_number)
            x_train = np.row_stack((x[shuffled_indices[:train1_end]] ,x[shuffled_indices[train2_start:int(len(x))]]))
            y_train = np.concatenate((y[shuffled_indices[:train1_end]], y[shuffled_indices[train2_start:int(len(x))]]))
            x_test = x[shuffled_indices[train1_end:int(train1_end + testing_number)]] 
            y_test = y[shuffled_indices[train1_end:int(train1_end + testing_number)]]

        data_train = np.column_stack((x_train, y_train)) #This combines the training attributes and labels into one array 
        data_test = np.column_stack((x_test, y_test)) #This combines the testing attributes and labels into one array     
        train_tree, depth = decision_tree_learning(data_train, 0)
        accuracy_split, confusion_matrix_split, precision_split, recall_split, f1_split = evaluate(data_test, train_tree)
        up_confusion_matrix.append(confusion_matrix_split) #Add the results of the confusion matrix into one list
        up_accuracy.append(accuracy_split)
        up_precision.append(precision_split)
        up_recall.append(recall_split)
        up_f1.append(f1_split)
        up_depth.append(depth)
    avg_up_confusion_matrix = np.array(up_confusion_matrix).mean(axis=0) #Finds the average of all the confusion matrices
    avg_up_accuracy = np.mean(up_accuracy)
    avg_up_precision = np.mean(up_precision, axis = 0)
    avg_up_recall = np.mean(up_recall, axis = 0)
    avg_up_f1 = np.mean(up_f1, axis = 0)
    avg_up_depth = np.mean(up_depth)

    return(avg_up_confusion_matrix, avg_up_accuracy, avg_up_precision, avg_up_recall, avg_up_f1, avg_up_depth)

def evaluate(data_test, trained_tree):
    confusion_matrix = np.zeros((4,4)) # Creates a 4x4 array filled with 0. 
    predicted_labels = [] 
    test_attribute = data_test[:,:-1] #Splits the dataset into attribute
    test_label = data_test[:,-1] #Splits the dataset into labels
    for i in range (len(test_attribute)): #Go through all the rows in the training dataset
        labels = predict(trained_tree, test_attribute[i])
        predicted_labels.append(labels) #Append the predicted labels into an array
    accuracy = sum(test_label == predicted_labels)/len(test_attribute) #Compute the accuracy 

    for j in range(len(test_attribute)): #Go through all the rows in the training dataset
        test_index = test_label[j] - 1 #Index is one less than the room number
        predicted_index = predicted_labels[j] - 1

        if test_label[j] == predicted_labels[j]:
            confusion_matrix[int(test_index), int(test_index)] = confusion_matrix[int(test_index), int(test_index)] + 1 #Increment by 1 to the respective value on the diagonal
        else: 
            confusion_matrix[int(test_index), int(predicted_index)] = confusion_matrix[int(test_index), int(predicted_index)] + 1 #Increment by 1 to the actual and predicted values of the confusion matrix
    
    diagonal = np.diag(confusion_matrix) #Gets the diagonal of the confusion matrix
    sum_col = np.sum(confusion_matrix, axis = 0) #Sum the classes by columns
    sum_row = np.sum(confusion_matrix, axis = 1) #Sum the classes by rows
    precision = diagonal/sum_col #Lecture formula
    recall = diagonal/sum_row #Lecture formula
    f1 = (2 * precision * recall)/(precision + recall) #Lecture formula
    return accuracy, confusion_matrix, precision, recall, f1

def predict(node, row):
    if node['left'] != None: #This is a decision node
        if row[node['attribute']] < node['value']: #Check the value of the row's attribute is less than the decision node attribute's value
            return predict(node['right'], row) #Recursively go into the right tree
        else:
            return predict(node['left'], row) #Recursively go into the left tree
    else:
        value = node['attribute'] #This is the predicted label based off the tree
        return (value) 


def prune_tree(tree,data): #prunes the trees leaves in a bottom up manner, if in doing so the accuracy (metric chosen in accordance with Machine Learning : Tom Mitchell book), improves or stays the same

        if tree["left"]!=None and tree["right"]!=None: #check if subtree exist
            #want to recur in a "bottom up" manner so that after pruning leaves, the former parent now leaf is also evaluated to see if it should be pruned
            if not(tree["left"]["leaf"]==True and tree["right"]["leaf"]==False) or not(tree["left"]["leaf"]==False and tree["right"]["leaf"]==True) or not(tree["left"]["leaf"]==True and tree["right"]["leaf"]==True):
                tree['left']=prune_tree(tree['left'],data)
                tree['right']=prune_tree(tree['right'],data)

            if tree["left"]["leaf"]==True and tree["right"]["leaf"]==False: # cant prune if left is leaf and right isnt, so recur right
                tree['right']=prune_tree(tree['right'],data)
            
            if tree["left"]["leaf"]==False and tree["right"]["leaf"]==True: # cant prune if right is a leaf and left isnt so recur left
                tree['left']=prune_tree(tree['left'],data)

            if tree["left"]["leaf"]==True and tree["right"]["leaf"]==True: #can prune if left and right are leaves
                unpruned_tree=copy.deepcopy(tree) # create a deep copy of the tree before potential pruning
                unpruned_accuracy,unpruned_confusion_matrix,unpruned_precision,unpruned_recall,unpruned_f1=evaluate(data,tree) #evaluate current unpruned trees perfornance

                if tree["right"]['attribute'] == tree["left"]['attribute']: #check if left and right leaves point to the same room, if so can prune w/o evaluating
                    tree['value'] = tree["left"]['value'] + tree["right"]['value']
                    tree['attribute'] = tree["left"]['attribute']
                    tree["left"]=None
                    tree["right"]=None
                    tree["leaf"]=True

                else: # if leaves point to different rooms, turn the node into a leaf with the value of the majority set
                    tree['value'] = tree["left"]['value'] + tree["right"]['value']
                    if tree["left"]['value'] > tree["right"]['value']:
                        tree['attribute'] = tree["left"]['attribute']
                    else:
                        tree['attribute'] = tree["right"]['attribute']
                    tree["left"]=None
                    tree["right"]=None
                    tree["leaf"]=True
                pruned_accuracy,pruned_confusion_matrix,pruned_precision,pruned_recall,pruned_f1=evaluate(data,tree) #evaluate to find pruned tree's metrics
                if pruned_accuracy >= unpruned_accuracy: #prune if the accuracy after pruning is equal to or better than the unpruned version, otherwise dont prune
                    return tree 
                else:
                    return unpruned_tree
        return tree

def nested_cross_validation(n_folds, n_instances, random_generator=np.random.default_rng()):
    test_indices = []
    trainval_indices = []
    val_indices = []
    train_indices = []
    fold_length = n_instances/n_folds
    shuffled_indices = random_generator.permutation(n_instances)
    for k in range(10):
        test_indices.append(shuffled_indices[int(k*fold_length):int((k+1)*fold_length)])
        trainval_indices = np.hstack((shuffled_indices[0:int(k*fold_length)], shuffled_indices[int((k+1)*fold_length):n_instances]))
        for i in range(n_folds-1):
            val_indices.append(trainval_indices[int(i*fold_length):int((i+1)*fold_length)])
            train_indices.append(np.hstack((trainval_indices[0:int(i*fold_length)], trainval_indices[int((i+1)*fold_length):len(trainval_indices)])))
    return(test_indices, val_indices, train_indices)

def treeplot(tree,x=0, y=5,h=0):
    if tree["left"]!=None and tree["right"]!=None: #check if subtree exist
        print("can recur further") #debug
        decisionlabel=str("["+"X" + str(tree['attribute']) + " < " + str(tree['value'])+"]")
        maxdecisions=2**h #maximum number of decisions per line is 2^n therefore split horizantly by this amount
        if h==5:
            xleft=x-1/(maxdecisions-3) #next x value to the left should slightly to left of parent
            xright=x+1/(maxdecisions-3)
        if h==6:
            xleft=x-1/(maxdecisions-4) #next x value to the left should slightly to left of parent
            xright=x+1/(maxdecisions-4)
            print("---------------h6 spacing----------------------------")
        if h==7:
            xleft=x-1/(maxdecisions-6) #next x value to the left should slightly to left of parent
            xright=x+1/(maxdecisions-6)
        else:
            xleft=x-1/(maxdecisions) #next x value to the left should slightly to left of parent
            xright=x+1/(maxdecisions)
        plt.text(x, y, decisionlabel,fontsize = 12, bbox = dict(facecolor="yellow",edgecolor='black', boxstyle='round',alpha = 1)) #plot decision node box
        print("plotting decision node at:","(",x,",",y,")") #debug
        xvals=[xleft,x,xright]#to plot lines between nodes
        yvals=[y-1/4,y,y-1/4]
        plt.plot(xleft,y-1/4) #plot point on left child (redundant)
        plt.plot(x,y) #point on parent
        plt.plot(xright,y-1/4) #plot point on right child
        plt.plot(xvals,yvals)
        print("recurring left ....."+"from", tree["attribute"],tree["value"])
        treeplot(tree['left'],xleft, y-1/4,h+1)
        print("recurring right....."+"from", tree["attribute"],tree["value"])
        treeplot(tree['right'],xright, y-1/4,h+1)
        return
    if tree["leaf"]==True:
            print("at leaf",tree["attribute"],tree["value"])
            leaflabel= str("["+"Room:"+str(tree["attribute"])+"]")
            plt.text(x, y, leaflabel,fontsize = 10, bbox = dict(facecolor="lightgreen",edgecolor='lightgreen', boxstyle='round',alpha = 1))
            print("plotting leaf node at:","(",x,",",y,")")
            return

def plot_tree(tree) :
    plt.figure(figsize=(2*5,2*5))
    treeplot(tree)
    plt.savefig("Cleandatasetunprunedtree")
    plt.close()
    return

data = np.loadtxt(fname = "./intro2ML-coursework1/wifi_db/clean_dataset.txt")
np.set_printoptions(threshold=3000)
np.seterr(invalid='ignore')
depth = 0 #tracks depth
seed = 60000
rg = np.random.default_rng(seed)
p_depth = []
p_confusion_matrix = []
p_accuracy = []
p_precision = []
p_recall = []
p_f1 = []

avg_up_confusion_matrix, avg_up_accuracy, avg_up_precision, avg_up_recall, avg_up_f1, avg_up_depth = split_dataset(data[:,:-1], data[:,7], k = 10, random_generator = rg) #This splits the data set to training and testing. Same as Lab 2
print("avg unpruned confusion matrix: ")
print(avg_up_confusion_matrix)
print("avg unpruned accuracy: ",avg_up_accuracy)
print("avg unpruned precision: ",avg_up_precision)
print("avg unpruned recall: ",avg_up_recall)
print("avg unpruned f1: ",avg_up_f1)
print("avg unpruned depth: ",avg_up_depth)

test_indices, val_indices, train_indices = nested_cross_validation(10, len(data), rg)
for i in range(0, 90):
    test_index = trunc(i/10)
    train_data = data[train_indices[i]]
    val_data = data[val_indices[i]]
    test_data = data[val_indices[test_index]]
    new_tree, depth = decision_tree_learning(train_data, 0)
    pruned = prune_tree(new_tree, val_data)
    temp_acc, temp_cm, temp_prec, temp_rec, temp_f1 = evaluate(test_data, pruned)
    p_confusion_matrix.append(temp_cm) #Add the results of the confusion matrix into one list
    p_accuracy.append(temp_acc)
    p_precision.append(temp_prec)
    p_recall.append(temp_rec)
    p_f1.append(temp_f1)
    p_depth.append(depth)

avg_p_confusion_matrix = np.array(p_confusion_matrix).mean(axis=0) #Finds the average of all the confusion matrices
avg_p_accuracy = np.mean(p_accuracy)
avg_p_precision = np.mean(p_precision, axis = 0)
avg_p_recall = np.mean(p_recall, axis = 0)
avg_p_f1 = np.mean(p_f1, axis = 0)
avg_p_depth = np.mean(p_depth)

print("avg pruned confusion matrix: ")
print(avg_p_confusion_matrix)
print("avg pruned accuracy: ",avg_p_accuracy)
print("avg pruned precision: ",avg_p_precision)
print("avg pruned recall: ",avg_p_recall)
print("avg pruned f1: ",avg_p_f1)
print("avg pruned depth: ", avg_p_depth)