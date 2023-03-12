import numpy as np
import random
import copy

class Validator:
    def __init__(self,feature_subset,classifier,dataset):
        self.feature_subset = feature_subset
        self.classifier = classifier
        self.dataset = dataset
        self.labels = dataset.iloc[:,0]
    
    def Accuracy(self):
        accuracy = 0
        # self.feature_subset.remove(0)
        df = self.dataset.iloc[:,1:]
        # print(df)
        df = df.iloc[:,self.feature_subset]
        # print(df)
        for i in range(0,len(df.iloc[:,0])):
            #get slice from 0->index and index+1->end
            n = df.index.isin([i])
            train = df[~n]
            test = df.iloc[i,:]
            self.classifier.Train(train)
            label_output = self.classifier.Test(test)
            if self.labels[i] == label_output:
                accuracy += 1
        accuracy = accuracy / (1.0*len(df.iloc[:,0]))
        return accuracy

class Classifier:
    def __init__(self,labels):
        self.labels = labels
        # self.train_instances = []
        pass

    def Train(self,train_instances):
        self.train_instances = train_instances
        pass

    def Test(self,test_instance):
        small = 99999999999
        # print(test_instance)
        # print(self.train_instances)
        for instance in self.train_instances.iterrows():
            dist = self.Euclidean_Distance(instance,test_instance)
            if dist < small:
                small = dist
                label = self.labels[instance[0]]
        return label

    def Euclidean_Distance(self,one,two):
        #for each x y do (x-y) squared added then sqrt
        sum = 0
        # print(list(one[1]))
        # print(list(two))
        for (x,y) in zip(list(one[1]),list(two)):
            sum += (x-y)**2 
        return (sum**.5)

class Node:
    def __init__(self, features,dataset):
        self.features = features #list of features in a node
        self.features.append(0)
        self.dataset = dataset
        self.labels = dataset.iloc[:,0]
        pass

    def append(self,feature):
        self.features.append(feature)

    def remove(self,feature):
        placeholder = [i for i in self.features if i != feature]
        self.features = placeholder
    def len(self):
        return len(self.features)

    def Evaluate(self,validator):
        self.value = validator.Accuracy()
        return self.value
    
    def __repr__(self):
        return f"{self.features}"
    
    def __str__(self):
        return f"{self.features}"

def ForwardSelection(features,df):
    features.remove(0)
    features_len = features.len() # Number of features
    #use feature size to start from 1, select 1 features, then next
    #to select that many features, permutate by
    labels = df.iloc[:,0] #preserve labels
    df = df #ive made a terrible mistake
    greedy_features = Node([],features.dataset)
    greedy_features.remove(0)
    remaining_features = copy.deepcopy(features)
    curr_best = 0 
    for i in range(1,features_len+2):
        #current best is greedy features evaluation
        #i need to remove the 0 that got in there somehow but like keep the labels
        start_len = greedy_features.len()
        #use same feature set for i+1 features
        loop_features = copy.deepcopy(greedy_features)
        for feature in remaining_features.features:
            #reset test features
            test_features = []
            test_features = copy.deepcopy(loop_features)
            #add each one feature at a time
            test_features.append(feature)
            #if this feature set is better than our previous best, then its the new best
            v = Validator(test_features.features,Classifier(labels),df)
            acc = test_features.Evaluate(v)
            print(f"Using feature(s) {test_features} accuracy is {acc}")
            if acc > curr_best:
                #remove new feature from set if it is the latest added one
                new_feature = feature
                greedy_features = copy.deepcopy(test_features)
                curr_best = acc
        #remove latest feature from list if theres a new one
        if greedy_features.len() > start_len:
            print(f'Feature set {greedy_features} was best, accuracy is {greedy_features.value}\n')
            remaining_features.remove(new_feature)
        else:
            print('Accuracy decreased or no more nodes to explore, ending feature search')
            #if adding more features is bad then stop
            break
    print(f'Best feature set was {greedy_features}, accuracy is {greedy_features.value}\n')
    return greedy_features

def BackwardElimination(features,df):
    print(df)
    features.remove(0)

    features_len = features.len()
    greedy_features = features
    labels = df.iloc[:,0]
    for i in range(1,features_len+2):
        v = Validator(greedy_features.features,Classifier(labels),df)
        curr_best = greedy_features.Evaluate(v)
        start_len = greedy_features.len()
        loop_features = copy.deepcopy(greedy_features)
        for feature in loop_features.features:
            test_features = copy.deepcopy(loop_features)
            test_features.remove(feature)

            v = Validator(test_features.features,Classifier(labels),df)
            acc = test_features.Evaluate(v)
            print(f"Using feature(s) {test_features} accuracy is {acc}")
            if acc > curr_best:
                greedy_features = copy.deepcopy(test_features)
                curr_best = acc
        if greedy_features.len() < start_len:
            print(f'Feature set {greedy_features} was best, accuracy is {greedy_features.value}')
        else:
            print('Accuracy decreased, ending feature search')
            #if removing more features is bad then stop
            break
    print(f'Finished search. Best feature subset is {greedy_features.features} with accuracy of {greedy_features.value}')
    return greedy_features

def main(feature_set,dataset):
    print('Welcome to Gwendolyn Kiler\'s Feature Selection Algorithm (SID:862208140)')
    # feature_num = int(input('Please enter total number of features: '))
    algorithm = int(input('Type the number of algorithm you want to run\n\n1. Forward Selection\n\n2. Backward Elimination\n\n'))

    # feature_list = []
    # for i in range(1,feature_num+1):
    #     feature_list.append(Feature(1,i))
    feature_set = Node(feature_set,dataset)    
    if algorithm == 1:
        ForwardSelection(feature_set,dataset)
    elif algorithm == 2:
        BackwardElimination(feature_set,dataset)

def datainput():
    # dataset_path = 'CS170_Spring_2022_Small_data__39.txt' 
    dataset_path = 'CS170_Spring_2022_Large_data__39.txt'
    data = []
    with open(dataset_path,'r') as f:
        for line in f:
            str_vals = line.strip().split()
            # label = int(float(str_vals[0]))
            vals = []
            for val in str_vals:
                vals.append(float(val))
            # temp = [label,vals]
            data.append(vals)
    import pandas as pd
    df = pd.DataFrame(data)
    df = df.rename(columns={df.columns[0]:'Class Label'})
    
    def znorm(column):
        return (column - np.mean(column))/np.std(column)
    
    #normalize the stuff
    df.iloc[:,1:] = df.iloc[:,1:].apply(znorm)
    # classifier = Classifier()
    feature_set = [] #should be set to 0 when doing self select
    print("Bear in mind that indices are offset, so input 0 for feature 1, 1 for feature 2 etc")
    num_features_want = int(input('Input desired number of features to select: '))
    
    for i in range(0,num_features_want):
        x = int(input("\nInput next desired feature subset: "))
        feature_set.append(x)
    # validator = Validator(feature_set,classifier,df)#feature_set
    # print(f'Using features {feature_set}')
    main(feature_set,df)
    # print(validator.Accuracy())

if __name__ == "__main__":
    random.seed()
    datainput()