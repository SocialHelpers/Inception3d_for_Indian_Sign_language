"""
This script has utilities for predicting a output label with a neural network
"""

def probability2label(arProbas, oClasses, nTop = 3):
    """
    identifies top 3 probabilities and their class label.
    :param arProbas: array of probabilities of each class.
    :param oClasses: names of output classes.
    :param nTop: number of probabilities to calculate.
    :return: Label and probabilities of identified classes.
    """

    arTopLabels = arProbas.argsort()[-nTop:][::-1]
    arTopProbas = arProbas[arTopLabels]

    for i in range(nTop):
        sClass = oClasses.dfClass.sClass[arTopLabels[i]] + " " + oClasses.dfClass.sDetail[arTopLabels[i]]
        print("Top %d: [%3d] %s (confidence %.1f%%)" % (i+1, arTopLabels[i], sClass, arTopProbas[i]*100.))
        
    return arTopLabels[0], oClasses.dfClass.sDetail[arTopLabels[0]], arTopProbas[0]