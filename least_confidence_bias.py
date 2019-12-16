from __future__ import print_function, division
import numpy

from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from scipy.spatial.distance import cdist

from generic_active_learner import GenericActiveLearner

from joblib import Parallel, delayed
import timeit

def compute_q(prob_x_in_class_1, P_max, q, i): 
	if prob_x_in_class_1 < P_max:
		q[i] = float(prob_x_in_class_1)/P_max
	else:
		q[i] =  float(1 - prob_x_in_class_1)/(1-P_max)

class LeastConfidenceBias( GenericActiveLearner ):
	"""Least Confidence Bias.

	Given unlabeled points, uses existing class imbalance in training
	set to bias picking samples with larger likelihood of belonging to
	minority class. In case of perfect class balance, becomes same as
	consensus entropy based sampling.

    Notes
    ---------------------
    Can only be used on binary classification. For now.
    README DOC: Topic - Least Confidence Bias
    """
	def query_for_points(self, x_unlabeled, k):
		# Get current class imbalance in training set
		percent_positive = numpy.sum(self.y_train)/self.y_train.shape[0]

 
		# This array (cpp) stores the probability that an unlabeled 
		# point is in class '1'.
		cpp = numpy.zeros(x_unlabeled.shape[0])
		for i in range(0, len(self.committee)):
			cpp += self.committee[i].predict_proba(x_unlabeled)[:,1]
		cpp /= len(self.committee)
		
		# P_max initialized as mean between 0.5 (ideal class ratio) 
		# and 1 - current fraction of positive labels(minority classes). 
		# We want to query labels whose probability of unlabeled point 
		# belonging to minority class is closer to this. 
		P_max = (0.5 + (1-percent_positive))/2
		
		# Initialize an empty list q. 
		# q will hold a value between [0,1] where higher value 
		# indicates that the P(y=1|x) of this unlabeled point 
		# is closer to P_max, and therefore should be queried.
		q = [0 for i in range(x_unlabeled.shape[0])]
		start_time = timeit.default_timer()
		Parallel(n_jobs=-1, require='sharedmem', prefer="threads")(
			delayed(compute_q)(cpp[i], P_max, q, i) for i in range(x_unlabeled.shape[0]))
		
		#q = []
		
		#for j in range(0, x_unlabeled.shape[0]):
		#	if cpp[j] < P_max:
		#		q.append(float(cpp[j])/P_max )
		#	else:
		#		q.append(float(1-cpp[j])/float(1-P_max) )

		#elapsed = timeit.default_timer() - start_time
		#print ("TIME : ", elapsed)
		# Query those points which have the highest values of q.
		return numpy.argsort( numpy.asarray(q) )[::-1][:k]