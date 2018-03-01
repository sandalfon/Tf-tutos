# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 10:45:43 2018

@author: Sandalfon
"""
import matplotlib.pyplot as plt

class StatPlot(object):
    
    def plot(self, train_loss,test_accuracy , options):
        generations = options.getRunOptions('generations')
        eval_every = options.getRunOptions('eval_every')
        output_every = options.getRunOptions('output_every')
        eval_indices = range(0, generations, eval_every)
        output_indices = range(0, generations, output_every)
        
        # Plot loss over time
        plt.plot(output_indices, train_loss, 'k-')
        plt.title('Softmax Loss per Generation')
        plt.xlabel('Generation')
        plt.ylabel('Softmax Loss')
        plt.show()
        
        # Plot accuracy over time
        plt.plot(eval_indices, test_accuracy, 'k-')
        plt.title('Test Accuracy')
        plt.xlabel('Generation')
        plt.ylabel('Accuracy')
        plt.show()