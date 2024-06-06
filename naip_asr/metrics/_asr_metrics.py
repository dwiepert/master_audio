"""
Metrics for scoring ASR transcriptions

Last Modified: 04/15/2024
Author(s): Daniela Wiepert
"""
# REQUIRED MODULES
#built-in
import string

#third-party
import numpy as np 


def _editdistance(r, h, cost_sub = 1, cost_ins = 1, cost_del = 1):
    """
    Get edit distance between a reference object and hypothesis object

    Inputs: 
    param r: (str list) reference (target) transcription, split into words/characters/phonemes/features 
    param h: (str list) hypothesis (predicted) transcription, split into words/characters/phonemes/features 
    param cost_sub: (int/float) cost associated with a substitution
    param cost_ins: (int/float) cost associated with an insertion
    param cost_del: (int/float) cost associated with a deletion

    Outputs:
    param score: (float) distance score
    param backtrace: (int matrix) matrix containing the lowest cost operation at a given index, which can be traced back to determine order of operations
    """

    #initialize cost matrix for Levenshtein distance
    m = len(r)+1
    n = len(h)+1 
    D = np.zeros((m,n))
    D[0,:] = np.arange(n) # First row represents the case where we achieve the hypothesis by inserting all hypothesis words into a zero-length reference.
    D[:,0] = np.arange(m) # First column represents the case where we achieve zero hypothesis words by deleting all reference words.
    
    #initialize backtracing
    backtrace = np.zeros((m,n)) #ok_ind = 0, sub_ind = 1, ins_ind = 2, del_ind = 3
    
    if h == []:
        score = D[-1,0]/len(r)
        backtrace = np.full((m,1), fill_value=3)

    else:
        #compute 
        for i in range(1,m):
            for j in range(1,n):
                if r[i-1] == h[j-1]:
                    D[i,j] = D[i-1,j-1]
                    backtrace[i,j] = 0
                else:
                    subc= D[i-1,j-1] + 1
                    insc = D[i,j-1] + 1
                    delc = D[i-1,j] + 1
                    D[i,j] = min(subc, insc, delc)

                    if D[i,j] == subc:
                        backtrace[i,j] = 1
                    elif D[i,j] == insc:
                        backtrace[i,j] = 2
                    else:
                        backtrace[i,j] = 3

        score = D[m-1,n-1] / len(r)

    return score, backtrace


def _backtracing_op(backtrace):
    """
    Function to backtrace through Levenshtein distance operations to generate list of steps (S/I/D)

    Input: 
    param backtrace: (np matrix) matrix containing all operations used in distance calculation

    Output:
    param steps: (str list) list containing all operations done to convert reference to hypothesis with minimum distance
    param counts: (dict) dictionary containing number of correct/substituted/inserted/deleted words/characters/phonemes/features
    """
    numSub = 0
    numDel = 0
    numIns = 0
    numCor = 0
    steps = []
    i = backtrace.shape[0]-1
    #print('Backtracing')
    #print(i)
    j = backtrace.shape[1]-1

    if j == 0:
        numDel = i 
        steps = ['d' for x in range(i)]
    
    else:
        while i > 0 and j > 0:
            if backtrace[i,j] == 0:
                numCor += 1
                i-=1
                j-=1
                steps.append('e')
            elif backtrace[i,j] == 1:
                numSub +=1
                i-=1
                j-=1
                steps.append('s')
            elif backtrace[i,j] == 2:
                numIns += 1
                j-=1
                steps.append('i')
            elif backtrace[i,j] == 3:
                numDel += 1
                i-=1
                steps.append('d')
        steps.reverse()
    
    counts = {'cor':[numCor], 'sub':[numSub], 'ins': [numIns], 'del':[numDel]}
    if counts == {}:
        print('pause')
    return steps, counts

def _base_rate(r, h, print):
    """
    Inputs: 
    param r: (str list) reference (target) transcription, split into words/characters/phonemes/features 
    param h: (str list) hypothesis (predicted) transcription, split into words/characters/phonemes/features 

    Outputs:
    param score: (float) distance score
    param steps: (str list) list containing all operations done to convert reference to hypothesis with minimum distance
    param counts: (dict) dictionary containing number of correct/substituted/inserted/deleted words/characters/phonemes/features
    """
    #get error rate
    score, backtrace = _editdistance(r,h)
    steps, counts= _backtracing_op(backtrace)
    
    if print:
        _alignedPrint(steps, r, h, score)

    return score, steps, counts


def _alignedPrint(list, r, h, result):
    '''
    This funcition is to print the result of comparing reference and hypothesis sentences in an aligned way.
    https://github.com/zszyellow/WER-in-python/blob/master/wer.py
    Attributes:
        list   -> the list of steps.
        r      -> the list of words produced by splitting reference sentence.
        h      -> the list of words produced by splitting hypothesis sentence.
        result -> the rate calculated based on edit distance.
    '''
    print("\nREF:", end=" ")
    for i in range(len(list)):
        if list[i] == "i":
            count = 0
            for j in range(i):
                if list[j] == "d":
                    count += 1
            index = i - count
            print(" "*(len(h[index])), end=" ")
        elif list[i] == "s":
            count1 = 0
            for j in range(i):
                if list[j] == "i":
                    count1 += 1
            index1 = i - count1
            count2 = 0
            for j in range(i):
                if list[j] == "d":
                    count2 += 1
            index2 = i - count2
            if len(r[index1]) < len(h[index2]):
                print(r[index1] + " " * (len(h[index2])-len(r[index1])), end=" ")
            else:
                print(r[index1], end=" "),
        else:
            count = 0
            for j in range(i):
                if list[j] == "i":
                    count += 1
            index = i - count
            print(r[index], end=" "),
    print("\nHYP:", end=" ")
    for i in range(len(list)):
        if list[i] == "d":
            count = 0
            for j in range(i):
                if list[j] == "i":
                    count += 1
            index = i - count
            print(" " * (len(r[index])), end=" ")
        elif list[i] == "s":
            count1 = 0
            for j in range(i):
                if list[j] == "i":
                    count1 += 1
            index1 = i - count1
            count2 = 0
            for j in range(i):
                if list[j] == "d":
                    count2 += 1
            index2 = i - count2
            if len(r[index1]) > len(h[index2]):
                print(h[index2] + " " * (len(r[index1])-len(h[index2])), end=" ")
            else:
                print(h[index2], end=" ")
        else:
            count = 0
            for j in range(i):
                if list[j] == "d":
                    count += 1
            index = i - count
            print(h[index], end=" ")
    print("\nEVA:", end=" ")
    for i in range(len(list)):
        if list[i] == "d":
            count = 0
            for j in range(i):
                if list[j] == "i":
                    count += 1
            index = i - count
            print("D" + " " * (len(r[index])-1), end=" ")
        elif list[i] == "i":
            count = 0
            for j in range(i):
                if list[j] == "d":
                    count += 1
            index = i - count
            print("I" + " " * (len(h[index])-1), end=" ")
        elif list[i] == "s":
            count1 = 0
            for j in range(i):
                if list[j] == "i":
                    count1 += 1
            index1 = i - count1
            count2 = 0
            for j in range(i):
                if list[j] == "d":
                    count2 += 1
            index2 = i - count2
            if len(r[index1]) > len(h[index2]):
                print("S" + " " * (len(r[index1])-1), end=" ")
            else:
                print("S" + " " * (len(h[index2])-1), end=" ")
        else:
            count = 0
            for j in range(i):
                if list[j] == "i":
                    count += 1
            index = i - count
            print(" " * (len(r[index])), end=" ")
    print("\nWER: " + str(result))


# MAIN SCORING IMPLEMENTATIONS
def wer(reference, hypothesis, print=False):
    """
    Calculate Levenshtein (edit) distance for words

    Input:
    param reference: (str) target transcription
    param hypothesis:(str) predicted transcription
    param print: (boolean) boolean indicating whether to print aligned transcription to console (default = False)

    Output:
    param score: (float) distance score
    param steps: (str list) list containing all operations done to convert reference to hypothesis with minimum distance
    param counts: (dict) dictionary containing number of correct/substituted/inserted/deleted words/characters/phonemes/features
    """
    #split for WER
    #lower, remove punctuation, then split string into words
    r = reference.lower()
    r = r.translate(str.maketrans('', '', string.punctuation))
    r = r.split() # target
    
    h = hypothesis.lower()
    h = h.translate(str.maketrans('', '', string.punctuation))
    h = h.split()# transcription

    return _base_rate(r, h, print)

def cer(reference, hypothesis, print):
    """
    Calculate Levenshtein (edit) distance for phonemes

    Input:
    param reference: (str) target transcription
    param hypothesis:(str) predicted transcription
    param print: (boolean) boolean indicating whether to print aligned transcription to console (default = False)

    Output:
    param score: (float) distance score
    param steps: (str list) list containing all operations done to convert reference to hypothesis with minimum distance
    param counts: (dict) dictionary containing number of correct/substituted/inserted/deleted words/characters/phonemes/features
    """
    r = reference.lower()
    r = r.translate(str.maketrans('', '', string.punctuation))
    r = r.replace(' ','')
    r = [x for x in r] # target
    h = hypothesis.lower()
    h = h.translate(str.maketrans('', '', string.punctuation))
    h = h.replace(' ','')
    h = [x for x in h]# transcription

    return _base_rate(r, h, print)
