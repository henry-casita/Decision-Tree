Decision Tree Coursework
========================

The aim of this project is to implement a decision tree algorithm and use it to determine one of the indoor
locations based on WIFI signal strengths collected from a mobile phone. There are 7 different WIFI signals and 4 different rooms the phone could be in. Two sample datasets for this scenario are provided in the data folder.

To load in a custom dataset, put your dataset into the 'data' folder and change the path of the file (line 233):
data = np.loadtxt(fname = [./data/<your filename>.txt])
Insert the name of your file instead of <your filename>.

To use the tree plotting function:
treeplot(Dictionary of tree to be plotted)

To run the file:
python decision_tree_coursework.py
