import logging
import argparse
import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

def main():
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)

    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers= [stream_handler]
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("goodware_folder", type=str, help="The folder with goodware files")
    parser.add_argument("malware_folder", type=str, help="The folder with malware files")
    parser.add_argument("goodware_opcodes_folder", type=str, help="The folder where goodware opcodes files are stored")
    parser.add_argument("malware_opcodes_folder", type=str, help="The folder where goodware opcodes files are stored")

    args = parser.parse_args()

    goodware_folder = args.goodware_folder    
    malware_folder = args.malware_folder
    goodware_opcodes_folder = args.goodware_opcodes_folder    
    malware_opcodes_folder = args.malware_opcodes_folder

    if not os.path.isdir(goodware_folder):
        print(f"Error: Path '{goodware_folder}' is not a valid directory.")
        return
    
    if not os.path.isdir(malware_folder):
        print(f"Error: Path '{malware_folder}' is not a valid directory.")
        return
    
    if not os.path.isdir(goodware_opcodes_folder):
        print(f"Error: Path '{goodware_opcodes_folder}' is not a valid directory.")
        return
    
    if not os.path.isdir(malware_opcodes_folder):
        print(f"Error: Path '{malware_opcodes_folder}' is not a valid directory.")
        return
    
    all_opcodes = [
        # Unconditional Branches
        'b', 'bl', 'bx', 'blx',
        
        # Conditional Branches
        'b.eq', 'b.ne', 'b.lt', 'b.gt', 'b.le', 'b.ge', 'b.hi', 'b.lo', 'b.pl', 
        'b.mi', 'b.vs', 'b.vc', 'b.cs', 'b.cc', 'b.al', 'b.nv',
        
        # Compare and Branch
        'cbz', 'cbnz',
        
        # Test and Branch
        'tbz', 'tbnz',
        
        # Return Instructions
        'ret', 'eret',
        
        # Exception Generation
        'svc', 'hvc', 'smc', 'brk', 'hlt',
        
        # Indirect Branching
        'br', 'blr', 'braa', 'brab', 'retab',
        
        # Hints
        'nop', 'yield', 'wfe', 'wfi', 'sev', 'sevl', 'isb', 'dmb', 'dsb'
    ]

    training_data = []
    
    goodware_opcodes_dir = os.listdir(goodware_opcodes_folder)
    goodware_opcodes_files = sorted(goodware_opcodes_dir, key=lambda x: os.path.getsize(os.path.join(goodware_opcodes_folder, x)))
    for file in goodware_opcodes_files[:150]:
        logging.info(f'Processing file: {file}')
        training_row = {}
        for opcode in all_opcodes:
            training_row[opcode] = 0

        application, extension = os.path.splitext(os.path.basename(file))
        logging.info(f'Application name: {application}')

        nr_lines = get_number_lines_file(application)
        logging.info(f'Number of lines: {nr_lines}')

        with open(f'{goodware_opcodes_folder}/{file}', 'r') as file:
            lines = file.readlines()
            for line in lines:
                opcode = line.strip()
                if opcode in all_opcodes:
                    training_row[opcode] += 1

        for key in training_row.keys():
            training_row[key] = round(training_row[key] / nr_lines, 3)

        training_row['label'] = 0

        logging.info(training_row)

        training_data.append(training_row)

    malware_opcodes_dir = os.listdir(malware_opcodes_folder)
    malware_opcodes_files = sorted(malware_opcodes_dir, key=lambda x: os.path.getsize(os.path.join(malware_opcodes_folder, x)))
    for file in malware_opcodes_files[:50]:
        logging.info(f'Processing file: {file}')
        training_row = {}
        for opcode in all_opcodes:
            training_row[opcode] = 0

        application, extension = os.path.splitext(os.path.basename(file))
        logging.info(f'Application name: {application}')

        nr_lines = get_number_lines_file(application)
        logging.info(f'Number of lines: {nr_lines}')

        with open(f'{malware_opcodes_folder}/{file}', 'r') as file:
            lines = file.readlines()
            for line in lines:
                opcode = line.strip()
                if opcode in all_opcodes:
                    training_row[opcode] += 1

        for key in training_row.keys():
            training_row[key] = round(training_row[key] / nr_lines, 3)

        training_row['label'] = 1

        logging.info(training_row)

        training_data.append(training_row)

    data = pd.DataFrame(training_data)

    # Features (X) and labels (y)
    X = data.drop('label', axis=1)  # Features
    y = data['label']   # Labels/Target

    logging.info(f"X: {X}")
    logging.info(f"y: {y}")

    # Assuming X and y are pandas DataFrames/Series
    kf = KFold(n_splits=5, shuffle=True, random_state=41)

    # To store the accuracy results for each algorithm
    rf_acc_list = []
    knn_acc_list = []
    nb_acc_list = []
    svm_acc_list = []
    dt_acc_list = []
    dnn_acc_list = []

    # 5-Fold Cross-validation
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # 1. Random Forest
        rf = RandomForestClassifier()
        rf.fit(X_train, y_train)
        rf_pred = rf.predict(X_test)
        rf_acc = accuracy_score(y_test, rf_pred)
        rf_acc_list.append(rf_acc)

        # 2. K-Nearest Neighbor
        knn = KNeighborsClassifier()
        knn.fit(X_train, y_train)
        knn_pred = knn.predict(X_test)
        knn_acc = accuracy_score(y_test, knn_pred)
        knn_acc_list.append(knn_acc)

        # 3. Naive Bayes
        nb = GaussianNB()
        nb.fit(X_train, y_train)
        nb_pred = nb.predict(X_test)
        nb_acc = accuracy_score(y_test, nb_pred)
        nb_acc_list.append(nb_acc)

        # 4. Support Vector Machine
        svm = SVC()
        svm.fit(X_train, y_train)
        svm_pred = svm.predict(X_test)
        svm_acc = accuracy_score(y_test, svm_pred)
        svm_acc_list.append(svm_acc)

        # 5. Decision Tree
        dt = DecisionTreeClassifier()
        dt.fit(X_train, y_train)
        dt_pred = dt.predict(X_test)
        dt_acc = accuracy_score(y_test, dt_pred)
        dt_acc_list.append(dt_acc)

        # 6. Deep Neural Network
        dnn = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=41)
        dnn.fit(X_train, y_train)
        dnn_pred = dnn.predict(X_test)
        dnn_acc = accuracy_score(y_test, dnn_pred)
        dnn_acc_list.append(dnn_acc)

    # Average accuracy across the 5 folds for each algorithm
    rf_avg_acc = np.mean(rf_acc_list)
    knn_avg_acc = np.mean(knn_acc_list)
    nb_avg_acc = np.mean(nb_acc_list)
    svm_avg_acc = np.mean(svm_acc_list)
    dt_avg_acc = np.mean(dt_acc_list)
    dnn_avg_acc = np.mean(dnn_acc_list)

    # Log the average accuracies
    logging.info(f"Random Forest Average Accuracy: {rf_avg_acc}")
    logging.info(f"KNN Average Accuracy: {knn_avg_acc}")
    logging.info(f"Naive Bayes Average Accuracy: {nb_avg_acc}")
    logging.info(f"SVM Average Accuracy: {svm_avg_acc}")
    logging.info(f"Decision Tree Average Accuracy: {dt_avg_acc}")
    logging.info(f"Deep Neural Network Average Accuracy: {dnn_avg_acc}")

def get_number_lines_file(application):
    with open('file_lines.json', 'r') as file_stats:
        stats = json.load(file_stats)
        return stats[application]['lines']
    
if __name__ == "__main__":
    main()