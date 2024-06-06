#IMPORTS
#built-in
from collections import OrderedDict

#third party
import torch.nn as nn


class BasicClassifier(nn.Module):
    """
        Head for classification task. 
        Initialize classification head with input sizes
        Source: https://colab.research.google.com/github/m3hrdadfi/soxan/blob/main/notebooks/Eating_Sound_Collection_using_Wav2Vec2.ipynb#scrollTo=Fv62ShDsH5DZ
    """

    def __init__(self, input_size=768, bottleneck=150, output_size=2, activation='relu',final_dropout=0.2, layernorm=False ):
        """
        Create a classification head with a dense layer, relu activation, a dropout layer, and final classification layer
        :param input_size: size of input to classification head 
        :param bottleneck: size to reduce to in intial dense layer (if you don't want to reduce size, set bottleneck=input size)
        :param output_size: number of categories for classification output
        :param num_labels: specify number of categories to classify
        :param activation: activation function for classification head
        :param final_dropout: amount of dropout to use in classification head
        :param layernorm: include layer normalization in classification head
        """
        super().__init__()
        self.input_size = input_size
        self.bottleneck= bottleneck
        self.output_size = output_size
        self.activation = activation
        self.layernorm = layernorm
        self.final_dropout = final_dropout

        classifier = []
        key = []
        classifier.append(nn.Linear(self.input_size, self.bottleneck))
        key.append('dense')
        if self.layernorm:
            classifier.append(nn.LayerNorm(self.bottleneck))
            key.append('norm')
        if self.activation == 'relu':
            classifier.append(nn.ReLU())
            key.append('relu')
        classifier.append(nn.Dropout(self.final_dropout))
        key.append('dropout')
        classifier.append(nn.Linear(self.bottleneck, self.output_size))
        key.append('outproj')

        self.classifier=classifier
        self.key=key

        seq = []
        for i in range(len(classifier)):
            seq.append((key[i],classifier[i]))
        
        self.head = nn.Sequential(OrderedDict(seq))

    def forward(self, x, **kwargs):
        """
        Run input (features) through the classifier
        :param features: input 
        :return x: classifier output
        """
        return self.head(x)