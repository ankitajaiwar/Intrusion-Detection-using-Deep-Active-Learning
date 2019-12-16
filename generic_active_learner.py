from __future__ import print_function, division
import numpy
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
#from .utils import *
import keras as K

class GenericActiveLearner:

	"""This is the Generic Active Learner Class.

	Base class for all active learners. Attributes consist of training set 
	and the committee (a list of models). Query method of this class is 
	random sampling.

    Parameters/Attributes
    ---------------------
    committee : { sklearn model,  list of sklearn_models }
        Used to form the committee (collection of models) on which 
        active Learning is performed.

    x_train : numpy.ndarray, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y_train : numpy.ndarray, shape (n_samples, )
        Training labels relative to x_train

    """
	def __init__(self, committee, x_train, y_train, x_test, y_test): 

		if x_train.shape[0] != y_train.shape[0]:
			raise DataError(message="x_train and y_train should have same number of rows.")

		self.x_train = x_train
		self.y_train = y_train

		try:	 
			len(committee)
		except TypeError:
			committee = [committee] #If only 1 model is passed, make a 1 element list to preserve homogeinity.
		finally:
			self.committee = committee

		for model in self.committee:
			model.fit(self.x_train, K.utils.to_categorical(self.y_train, 2), batch_size = 128,epochs=10,verbose=0, validation_data = (x_test, K.utils.to_categorical(y_test, 2)))



	def get_performance_score(self, x_test, y_test, performance_metric='f1'):
		"""Evaluate active learner performance according to testing set.

        Parameters
        ----------
        x_test : numpy.ndarray, shape (n_samples, n_features)
            Test samples.

        y_test : numpy.ndarray, shape (n_samples,)
            True labels of x_test.

        performance_metric : {'f1', 'accuracy'}, optional (default='accuracy')
            Performance metric on which to test. 'f1' implies F1 score
            will be computed. 'accuracy' tests simple accuracy.

        Returns
        -------
        score : performance score based on the selected metric.
        
        """
		#if x_test.shape[0] != y_test.shape[0]:
		#	raise DataError(message="x_test and y_test should have same number of rows.")
		#if x_test.shape[1:] != self.x_train.shape[1:]: 
		#	raise DataError(message="x_test and x_train should have same number of columns.")
		
		num_classes = numpy.unique(y_test).shape[0]
		y_predict_proba = numpy.zeros([ y_test.shape[0], num_classes ])

		# In case of a committee based model, add all P(y_i|x) 
		# for a particular i from all models.
		for i in range(0, len(self.committee)):
			y_predict_proba += self.committee[i].predict_proba(x_test)
		y_predict_proba /= len(self.committee)

		# Pick the y_i with largest P(y_i|x) as label for x. 
		y_pred = numpy.zeros(y_test.shape)
		for i in range(0, y_predict_proba.shape[0]):
			y_pred[i] = numpy.argmax(y_predict_proba[i])

		print("Confusion : ", confusion_matrix(y_test, y_pred))
		# Evaluate performance based on metric.
		if performance_metric == 'f1':
			return f1_score(y_pred, y_test)
		else:
			if performance_metric == 'accuracy':
				return accuracy_score(y_pred, y_test)
			else:
				# Implement your own metric here.
				raise NotImplementedError

		
	def query_for_points(self, x_unlabeled, k):
		"""Active learner query handler.

		All derived classes of active_learner class should ideally only
		change this function.

        Parameters
        ----------
        x_unlabeled : numpy.ndarray, shape (n_samples, n_features)
            Unlabeled set.

        k : int
            Batch size.

        Returns
        -------
        indices : numpy.ndarray, shape (k, )
        	Indices of x_unlabeled that needs to be queried.
        
        """
		if x_unlabeled.shape[1:] != self.x_train.shape[1:]: 
			raise DataError

		# Random sampling is implemented here.
		return numpy.random.permutation(x_unlabeled.shape[0])[:k]


	def iterative_simulation(self, x_unlabeled, y_unlabeled, x_test, y_test, k, iterations, performance_metric = 'f1'):
		"""Active learner iterative simulator.

		Simulates how an active learning method will impact performance of model.
		Given unlabeled set, test set, k, iterations and performance metric, 
		returns the peformance metric per iteration of model on the test set

        Parameters
        ----------
        x_unlabeled : numpy.ndarray, shape (n_samples, n_features)
            "Unlabeled" set samples. In case of iterative simulation, 
            this will be labeled, but we pretend that it is unlabeled.

        y_unlabeled : numpy.ndarray, shape (n_samples,)
            True labels of the "Unlabeled" set.

        x_test : numpy.ndarray, shape (n_samples, n_features)
            Test samples.

        y_test : numpy.ndarray, shape (n_samples,)
            True labels of x_test.

        k : int
            Batch size.

        iterations : int
            Number of times to run the iterative simulation. Make sure that
            n_samples of x_unlabeled is >= k*iterations.

        performance_metric : {'f1', 'accuracy'}, optional (default='f1')
            Performance metric on which to test. 'f1' implies
            F1 score will be computed. 'accuracy' tests simple accuracy.

        Returns
        -------
        indices : numpy.ndarray, shape (k, )
        	Indices of x_unlabeled that needs to be queried.
        
        """
		if x_unlabeled.shape[0] != y_unlabeled.shape[0]:
			raise DataError
		if x_unlabeled.shape[1:] != self.x_train.shape[1:]: 
			raise DataError

		performance_score = []

		print("------- ENTERING MAIN ACTIVE LEARNING LOOP -------")
		#print_progress_bar(0, iterations, prefix = 'Progress:', suffix = 'Complete', length = 100)

		for iteration in range(0, iterations):
			print("Iteration: ", iteration)
			performance_score.append( self.get_performance_score(x_test, y_test, performance_metric=performance_metric) )


			query_point_indices = self.query_for_points(x_unlabeled, k)

			count = 0
			for j in range(y_unlabeled.shape[0]-1, -1, -1):
				if j in query_point_indices:
					self.x_train = numpy.append( self.x_train, [x_unlabeled[j]], 0 )
					self.y_train = numpy.append( self.y_train, y_unlabeled[j] )

					x_unlabeled = numpy.delete( x_unlabeled, j, 0 ) 
					y_unlabeled = numpy.delete( y_unlabeled, j )
					count += 1

			for model in self.committee:
				model.fit(self.x_train, K.utils.to_categorical(self.y_train, 2), batch_size = 128,epochs=10,verbose=0, validation_data = (x_test, K.utils.to_categorical(y_test, 2)))


			#print_progress_bar(iteration+1, iterations, prefix = 'Progress:', suffix = 'Complete', length = 100)
			if x_unlabeled.shape[0] < k:
				break

		return performance_score