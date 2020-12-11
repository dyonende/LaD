Authors: Dyon van der Ende, Etienne Galea, Lois Rink
Research question: Are Maltese news articles less objective than Dutch and German news articles on the topic of abortion?
Link to blog post: https://languageasdata.wordpress.com/2020/12/03/cultural-acceptance-of-abortion

Before continuing, please make sure all requirements from the requirements.txt are fulfilled.


This readme will walk you through all the steps to obtain the data and run the experiments that were used for this assignment.


CODE
All scripts can be found in the ‘code’ folder.


1. get_all_documents.py
Run this script to obtain 100 news articles for Dutch, German and Maltese.
When executed, it will ask for a folder to store the data. Choose a convenient name, for example “./data/”. This will create a new folder on this location if it does not already exist. Then in that folder, three subfolders will be created for each language, where the data files will be stored.
The output is 100 json files for both Dutch and German and 1 csv file for Maltese.
The output will not be the same as the data that was used for the assignment, as the content of the website constantly changes. In the data that is provided, urls to the articles are stored to obtain the original articles.


2. json_to_tsv.py
This script is added as a convenience, to convert the 100 json files to 1 tsv file, it is not necessary to do this step, as the rest of the steps will use the json files.
It asks for the folder that contains the json files and will try to convert all json files in that folder into 1 file, so make sure that only the right data is in that folder.


3. evaluate_annotations.py
This script will also ask for the folder with the annotation sheets. The annotations sheets are provided in a folder containing 1 subfolder for each language. This script searches recursively through the provided folder, so it is not necessary to run it on each subfolder.
The output are confusion matrices and basic statics for each annotated term, printed to the terminal.


4. run_all_analyses.py
This is a large script that can take a long time to run. It is important that all python libraries and language models are installed. The language models should be located in the same folder. The data that is analysed is the data from the first step, with the same folder structure. The script will ask for the locations of the models and the data.
The output is all the results that are described in the blog post and are printed to the terminal.
To obtain the result from the blog post, run this script with the data set that was provided.


DATA
The data can be found here: https://drive.google.com/drive/folders/1VpXCXaRFJPaZZzoSol9V8eV9bdVAcvOz?usp=sharing
This folder contains two tsv files and three subfolders. The two tsv files are the same data as the json files in the folders for Dutch and German, but in a different format. In the ‘mt’ folder, the data is already in tsv format.


Annotations
All the annotated sheets can be found in the folder ‘annotations’.
The annotations are stored in a subfolder per language. Each subfolder contains six files: 3 terms annotated by two annotators.
