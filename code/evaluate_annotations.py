import pandas as pd
import glob
import os.path
import random
from itertools import combinations
from sklearn.metrics import cohen_kappa_score, confusion_matrix


terms = ["Gesetz", "Schwangerschaft", "Vergewaltigung", "wet", "verkrachting", "zwangerschap", "liġi", "stupru", "tqala"]
categories = ["C", "U", "O", "X"]

data_path = input("provide path to annotation sheets folder: ")

for term in terms:
    print(term)
    annotations = {}
    # Read in the data
    for sheet in glob.glob(data_path+"/**/annotationsheet_" + term +"*.tsv", recursive=True):
        filename, extension = os.path.basename(sheet).split(".")
        prefix, term, annotator = filename.split("_")

        # Read in annotations
        annotation_data = pd.read_csv(sheet, sep="\t", header=0, keep_default_na=False)
        annotations[annotator] = annotation_data["Annotation"]

    annotators = annotations.keys()


    for annotator_a, annotator_b in combinations(annotators, 2):
        agreement = [anno1 == anno2 for anno1, anno2 in  zip(annotations[annotator_a], annotations[annotator_b])]
        percentage = sum(agreement)/len(agreement)
        print(annotator_a, annotator_b)
        print("Percentage Agreement: %.2f" %percentage)
        kappa = cohen_kappa_score(annotations[annotator_a], annotations[annotator_b], labels=categories)
        print("Cohen's Kappa: %.2f" %kappa)
        confusions = confusion_matrix(annotations[annotator_a], annotations[annotator_b], labels=categories)
        print(confusions)
        print()

