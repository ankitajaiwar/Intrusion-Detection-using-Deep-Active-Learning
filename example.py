from __future__ import print_function, division

import numpy

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from example_utils import prune_dataset, binarize_score_array

from PAL import GenericActiveLearner
from PAL.point_density import *
from PAL.confusion import *
from PAL.disagreement import *
from PAL.class_imbalance import *

import matplotlib.pyplot as plt



# Load train-unlabeled, and test data.
x_train_unlabeled = numpy.load("example_data/x_all_train_unlabeled.npy")
y_train_unlabeled = numpy.load("example_data/y_all_train_unlabeled.npy", allow_pickle=True)

x_valid = numpy.load("example_data/x_all_valid.npy")
y_valid = numpy.load("example_data/y_all_valid.npy", allow_pickle=True)

print(x_train_unlabeled.shape, " ", y_train_unlabeled.shape)
print(x_valid.shape, " ", y_valid.shape)



# Prune data to make sure we don't have none values, too small 
# score_arrays, and score_arrays that does not work with 
# Mel's Volcano Metric.
print("Pruning Training Set: ")
print("\n")
x_train_unlabeled, y_train_unlabeled = prune_dataset(x_train_unlabeled, y_train_unlabeled)
print("\n\n\n")
print("Pruning Validation Set: ")
print("\n")
x_valid, y_valid = prune_dataset(x_valid, y_valid)
print("\n\n\n")

# Randomly permute the data.
indices = numpy.random.permutation(x_train_unlabeled.shape[0])
x_train_unlabeled = x_train_unlabeled[indices]
y_train_unlabeled = y_train_unlabeled[indices]


# Split train-unlabeled into train data and unlabeled data. 
# (change the fractions to suit your needs).
x_train = x_train_unlabeled[:int(0.4*x_train_unlabeled.shape[0])]
y_train = y_train_unlabeled[:int(0.4*y_train_unlabeled.shape[0])]

x_unlabeled = x_train_unlabeled[int(0.4*x_train_unlabeled.shape[0]):] 
y_unlabeled = y_train_unlabeled[int(0.4*y_train_unlabeled.shape[0]):]


# Convert score arrays into binary data 
# (0 -> no intent to cross, 1 -> intent to cross).
print("Binarizing Training Set: ")
print("\n")
y_train_binary = binarize_score_array(y_train)
print("Binarizing Unlabeled Set: ")
print("\n")
y_unlabeled_binary = binarize_score_array(y_unlabeled)
print("Binarizing Validation Set: ")
print("\n")
y_valid_binary = binarize_score_array(y_valid)


# Get shapes of train, unlabeled, and test. Should print out (XXX, 520), (XXX,).
print("Training Set Shapes: ", x_train.shape, " ", y_train_binary.shape)
print("Unlabeled Set Shapes: ", x_unlabeled.shape, " ", y_unlabeled_binary.shape)
print("Testing Set Shapes: ", x_valid.shape, " ", y_valid_binary.shape)

# Make a committee of models.
models = [ RandomForestClassifier(n_jobs=-1, n_estimators=100), RandomForestClassifier(n_jobs=-1, n_estimators=200), RandomForestClassifier(n_jobs=-1, n_estimators=50) ]


# Start AL of various methods on this data.
print(" RANDOM SAMPLING : ")
ral = GenericActiveLearner(models, x_train[:,:512], y_train_binary)
random_f1 = ral.iterative_simulation(x_unlabeled[:,:512], y_unlabeled_binary, x_valid[:,:512], y_valid_binary, 50, 50, 'f1')
print("\n\n\n")
#print(" UNLABELED LARGE CLUSTERS : ")
#ullcal = UnlabeledLargeClusters(models, x_train[:,:512], y_train_binary)
#unlbl_lg_clust_f1 = ullcal.iterative_simulation(x_unlabeled[:,:512], y_unlabeled_binary, x_valid[:,:512], y_valid_binary, 5, 5, 'f1')
#print("\n\n\n")
#print(" UNEXPLORED LARGE CLUSTERS : ")
#uxlcal = UnexploredLargeClusters(models, x_train[:,:512], y_train_binary)
#unxp_lg_clust_f1 = uxlcal.iterative_simulation(x_unlabeled[:,:512], y_unlabeled_binary, x_valid[:,:512], y_valid_binary, 5, 5, 'f1')
#print("\n\n\n")
#print(" VOTE ENTROPY : ")
#veal = VoteEntropy(models, x_train[:,:512], y_train_binary)
#ve_f1 = veal.iterative_simulation(x_unlabeled[:,:512], y_unlabeled_binary, x_valid[:,:512], y_valid_binary, 5, 5, 'f1')
#print("\n\n\n")
#print(" KL DIVERGENCE : ")
#klal = KLDivergence(models, x_train[:,:512], y_train_binary)
#kl_f1 = klal.iterative_simulation(x_unlabeled[:,:512], y_unlabeled_binary, x_valid[:,:512], y_valid_binary, 5, 5, 'f1')
#print("\n\n\n")
print(" CONSENSUS ENTROPY : ")
ceal = ConsensusEntropy(models, x_train[:,:512], y_train_binary)
ce_f1 = ceal.iterative_simulation(x_unlabeled[:,:512], y_unlabeled_binary, x_valid[:,:512], y_valid_binary, 50, 50, 'f1')
print("\n\n\n")
#print(" CONSENSUS MARGIN : ")
#cmal = ConsensusMargin(models, x_train[:,:512], y_train_binary)
#cm_f1 = cmal.iterative_simulation(x_unlabeled[:,:512], y_unlabeled_binary, x_valid[:,:512], y_valid_binary, 5, 5, 'f1')
#print("\n\n\n")
print(" LEAST CONFIDENCE BOUND : ")
lcbal = LeastConfidenceBias(models, x_train[:,:512], y_train_binary)
lcb_f1 = lcbal.iterative_simulation(x_unlabeled[:,:512], y_unlabeled_binary, x_valid[:,:512], y_valid_binary, 50, 50, 'f1')
print("\n\n\n")
print(" MY WAY : ")
mwal = MyWay(models, x_train[:,:512], y_train_binary)
mw_f1 = mwal.iterative_simulation(x_unlabeled[:,:512], y_unlabeled_binary, x_valid[:,:512], y_valid_binary, 50, 50, 'f1')


# Make a nice looking plot for the results.
#try:
fig = plt.figure(figsize=(25.8,17.82))
ax = fig.add_subplot(111)

ax.plot(random_f1, linestyle='-', color='k', label='Random Sampling')

#ax.plot(unlbl_lg_clust_f1, linestyle='-', color='r', label='Unlabeled Large Clusters')
#ax.plot(unxp_lg_clust_f1, linestyle='--', color='r', label='Unexplored Large Clusters')

ax.plot(ve_f1, linestyle='-', color='#034769', label='Vote Entropy')
ax.plot(kl_f1, linestyle='--', color='#034769', label='KL Divergence')

ax.plot(ce_f1, linestyle='-', color='#A65900', label='Consensus Entropy')
ax.plot(cm_f1, linestyle='--', color='#A65900', label='Consensus Margin')

ax.plot(lcb_f1, linestyle='-', color='#6E0069', label='Least Confidence Bias')
ax.plot(mw_f1, linestyle='--', color='#6E0069', label='My Method')

plt.xticks(fontsize=15)
plt.legend(fontsize=15)
plt.yticks(fontsize=15)

plt.xlabel("Iterations", fontsize= 20, fontweight='bold')
plt.ylabel("F1 Score", fontsize= 20, fontweight='bold')
plt.title("F1 Score per Iterations for all the Methods implemented in PAL", fontsize= 30)
plt.savefig('Results.pdf', dpi = 300)
plt.show()
'''
except DisplayError:
	print("Non GUI based system detected. Printing F1 scores instead.")
	print("\n\n")
	print("- Random Sampling: ")
	print(random_f1)
	print("\n\n")
	print("- Unlabeled Large Clusters: ")
	print(unlbl_lg_clust_f1)
	print("\n\n")
	print("- Unexplored Large Clusters: ")
	print(unxp_lg_clust_f1)
	print("\n\n")
	print("- Vote Entropy: ")
	print(ve_f1)
	print("\n\n")
	print("- KL Divergence: ")
	print(kl_f1)
	print("\n\n")
	print("- Consensus Entropy: ")
	print(ce_f1)
	print("\n\n")
	print("- Consensus Margin: ")
	print(cm_f1)
	print("- Least Confidence Bias: ")
	print(lcb_f1)
	print("- My Way: ")
	print(mw_f1)
'''