Decision Tree Coursework
========================

The aim of this project is to implement a decision tree algorithm and use it to determine one of the indoor
locations based on WIFI signal strengths collected from a mobile phone. There are 7 different WIFI signals and 4 different rooms the phone could be in. Documentation for this project can be found in *Decision Tree Coursework.pdf*.
Two sample datasets for the problem scenario, [*clean_dataset.txt*](data/clean_dataset.txt) and [*noisy_dataset.txt*](data/noisy_dataset.txt) are provided in the [*data*](data) folder.

To load in a custom dataset, put your dataset into the 'data' folder and change the path of the file (line 233):
data = np.loadtxt(fname = [./data/<your filename>.txt])
Insert the name of your file instead of <your filename>.

To use the tree plotting function:
treeplot(Dictionary of tree to be plotted)

To run the file:
python decision_tree_coursework.py
