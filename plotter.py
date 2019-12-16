import numpy
import matplotlib.pyplot as plt

#f1_scores3 = numpy.load('F1Scores4.npy')
#f1_scores2 = numpy.load('F1Scores3.npy')
#f1_scores1 = numpy.load('F1Scores2.npy')
f1_scores0 = numpy.load('F1Scores_151.npy')
f1_scores1 = numpy.load('F1Scores_152.npy')
f1_scores2 = numpy.load('F1Scores_153.npy')
f1_scores3 = numpy.load('F1Scores_154.npy')

f1_scores = (f1_scores0 + f1_scores1 + f1_scores2 + f1_scores3)/4

plt.xlabel("Iterations")
plt.ylabel("F1 Score")
plt.title("Average F1 Score per Iteration for Intrusion Detection")

plt.plot(f1_scores)
plt.show()