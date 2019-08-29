import SimpSOM as sps
import pandas as pd

df = pd.read_csv('../data_test/page_aug_fixed_out.csv', delimiter=',', encoding='iso-8859-1', nrows=100)
raw_data = df.values
import pdb; pdb.set_trace()


#Build a network 20x20 with a weights format taken from the raw_data and activate Periodic Boundary Conditions.
net = sps.somNet(20, 20, raw_data, PBC=True)

#Train the network for 10000 epochs and with initial learning rate of 0.01.
net.train(0.01, 10000)

#Save the weights to file
net.save('test_weights')

#Print a map of the network nodes and colour them according to the first feature (column number 0) of the dataset
#and then according to the distance between each node and its neighbours.
net.nodes_graph(colnum=0)
net.diff_graph()

#Project the datapoints on the new 2D network map.
net.project(raw_data, labels=labels)

#Cluster the datapoints according to the Quality Threshold algorithm.
net.cluster(raw_data, type='qthresh')
