# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 20:20:20 2020

@author: gjsiv

Name: Janardhan Siva Kumar Gunakala

Student ID: R00183561

Project Title: Digitization of Handwritten Text

"""

# Importing the libraries
import os
import random
import sys
import cv2
import editdistance
import re
import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")


# ----------------------------- update the below local path based on the system saved location. ----------------------

# accesing the directories
imageFiles = os.listdir('/Users/gjsiv/Downloads/Assignments/Project/Final Code/Data/iam-handwriting-top50/data_subset/data_subset/')

#print(imageFiles)

data_folder = "/Users/gjsiv/Downloads/Assignments/Project/Final Code/Data/"

output_folder = "/Users/gjsiv/Downloads/Assignments/Project/Final Code/Output/"

charData = data_folder + 'charData.txt'
LabelData = data_folder + 'Labelled_Data.txt'
wordChar = data_folder + 'wordCharData.txt'

# ------------------------------------- Declaring the Class

class Batch:
    
    # batch containig images and ground truth texts
    def __init__(self, Texts, images):
        
        self.images = np.stack(images, axis= 0)
        self.Texts = Texts
        
class Decorder_Type:
    
    BestPath = 0
    
# ------------------------------------------------- Word Segmentation --------------------------------------------------------
    
def prepareImage(image, height):
    
    # convert the given image to grayscale image and resize to desired height
    assert image.ndim in (2, 3)
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    A = image.shape[0]
    factor = height / A
    return cv2.resize(image, dsize=None, fx=factor, fy=factor)

def createKernel(kernelSize, sigma, theta):
    
    assert kernelSize % 2
    halfSize = kernelSize // 2
    
    #print("Kernel Size: ", kernelSize)
    #print("Half Size: ", halfSize)
    
    kernel = np.zeros([kernelSize, kernelSize])
    sigmaX = sigma
    sigmaY = sigma * theta
    
    for i in range(kernelSize):
        
        for j in range(kernelSize):
            
            x = i - halfSize
            y = j - halfSize
            
            expTerm = np.exp(-x**2 / (2 * sigmaX) - y**2 / (2 * sigmaY))
            xTerm = (x**2 - sigmaX**2) / (2 * math.pi * sigmaX**5 * sigmaY)
            yTerm = (y**2 - sigmaY**2) / (2 * math.pi * sigmaY**5 * sigmaX)
            
            kernel[i, j] = (xTerm + yTerm) * expTerm
    
    kernel = kernel / np.sum(kernel)
    
    return kernel

def wordSegment(image, kernelSize=25, sigma=11, theta=7, minArea=0):
    
    # apply filter kernel
    kernel = createKernel(kernelSize, sigma, theta)
    imageFiltered = cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_REPLICATE).astype(np.uint8)
    (_, imageThres) = cv2.threshold(imageFiltered, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    imageThres = 255 - imageThres
    
    (components, _) = cv2.findContours(imageThres, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    # append components to result
    result = []
    for comp in components:
        
        # skip small word candidates
        if cv2.contourArea(comp) < minArea:
            continue
        
        # appending bounding box and image of word to result list
        curBox = cv2.boundingRect(comp)
        (x, y, w, h) = curBox
        curImage = image[y:y+h, x:x+w]
        result.append((curBox, curImage))
        
    # returning the list of words, sorted by x-coordinate
    return sorted(result, key= lambda entry:entry[0][0])

# ------------------------------------------------------- Model -----------------------------------------------------------------

class Model:
    
    # intializing the variables
    batchSize = 50
    imageSize = (128, 32)
    maxTextlength = 32
    Count = 0
    
    # class Initializing
    def __init__(self, charList, decorderType = Decorder_Type.BestPath, mustRestore=False):
        
        # inital model - adding CNN, RNN and CTC layers
        # Initialize TF model
        
        self.charList = charList
        self.decorderType = decorderType
        self.mustRestore = mustRestore
        self.snapID = 0
        Model.Count += 1
        
        # New placeholder - Whether to use normalization over a batch or a population
        self.is_train = tf.compat.v1.placeholder(tf.bool, name='is_train')
        
        # calling the CNN layer
        self.inputImages = tf.compat.v1.placeholder(tf.float32, shape=(Model.batchSize, Model.imageSize[0], Model.imageSize[1]))
        
        # passing the input image into the CNN Layer
        cnnOutImage = self.CNN_Layer(self.inputImages)        
        
        # RNN Layer - the output of cnn layer is passed input to the RNN
        rnnOutImage = self.RNN_Layer(cnnOutImage)
        
        # CTC Layer
        (self.loss, self.decoder) = self.CTC_Layer(rnnOutImage)
        
        # optimizer for NN parameters
        #self.batchesTrained = 0
        self.learningRate = tf.compat.v1.placeholder(tf.float32, shape= [])
        self.optimizer = tf.compat.v1.train.RMSPropOptimizer(self.learningRate).minimize(self.loss)
        
        # Initialize TF 
        (self.session, self.saver) = self.TF()       
        
    
    def CNN_Layer(self, cnnInImg):
        
        # create the CNN layers and return output of these layers
        cnnInLayer = tf.expand_dims(input= cnnInImg, axis=3)
        
        # list of parameters for the Layers
        kernelVals = [5, 5, 3, 3, 3]
        featureVals = [1, 32, 64, 128, 128, 256]
        strideVals = poolVals = [(2,2), (2,2), (1,2), (1,2), (1,2)]
        numLayers = len(strideVals)
        
        # create Layers - input to first CNN layer
        pool = cnnInLayer
        for i in range(numLayers):
            
            kernel = tf.Variable(tf.random.truncated_normal([kernelVals[i], kernelVals[i], featureVals[i], featureVals[i + 1]], stddev=0.1))
            
            # using the tensorflow convolution 2d
            con = tf.nn.conv2d(pool, kernel, padding='SAME', strides=(1,1,1,1))
            
            # using the Rectified Linear Unit - relu
            relu = tf.nn.relu(con)
            
            # using the tensorflow max pooling
            pool = tf.nn.max_pool2d(relu, (1, poolVals[i][0], poolVals[i][1], 1), (1, strideVals[i][0], strideVals[i][1], 1), 'VALID')
        
        return pool
    
    def RNN_Layer(self, rnnInImg):
        
        # create the RNN Layers and return output of these Layers
        rnnInLayer = tf.squeeze(rnnInImg, axis=[2])
        
        # basic cells which is used to build RNN - 2 layers
        numHidden = 256
        cells_data = [tf.contrib.rnn.LSTMCell(num_units= numHidden, state_is_tuple= True) for _ in range(2)]
        
        # stack basic cells
        stack_basic = tf.contrib.rnn.MultiRNNCell(cells_data, state_is_tuple= True)
        
        # bidirectional RNN - AxBxC -> AxBx2C
        ((fw, bw), _) = tf.nn.bidirectional_dynamic_rnn(cell_fw= stack_basic, cell_bw= stack_basic, inputs= rnnInLayer, dtype= rnnInLayer.dtype)
        
        # AxBxC + AxBxC -> AxBx2C -> AxBx1x2C
        concat = tf.expand_dims(tf.concat([fw, bw], 2), 2)
        
        # project output to chars including blank : AxBx1x2C -> AxBx1xC -> AxBxC
        kernel = tf.Variable(tf.random.truncated_normal([1, 1, numHidden * 2, len(self.charList) + 1], stddev=0.1))
        
        return tf.squeeze(tf.nn.atrous_conv2d(value= concat, filters= kernel, rate= 1, padding= 'SAME'), axis=[2])
    
    def CTC_Layer(self, ctcInImg):
        
        # create CTC Loss and decoder and return them AxBxC -> BxAxC
        self.ctcInBAC = tf.transpose(ctcInImg, [1, 0, 2])
        
        # ground truth text as sparse tensor
        self.Texts = tf.SparseTensor(tf.compat.v1.placeholder(tf.int64, shape= [None, 2]), tf.compat.v1.placeholder(tf.int32, [None]), tf.compat.v1.placeholder(tf.int64, [2]))
        
        # calculate the Loss for batch
        self.seqLen = tf.compat.v1.placeholder(tf.int32, [None])
        loss = tf.compat.v1.nn.ctc_loss(labels= self.Texts, inputs= self.ctcInBAC, sequence_length= self.seqLen, ctc_merge_repeated= True)
        
        # calculating Loss for each element to compute label probability
        self.savedCtcInput = tf.compat.v1.placeholder(tf.float32, shape=[Model.maxTextlength, None, len(self.charList) + 1])
        self.lossPerElement = tf.compat.v1.nn.ctc_loss(labels= self.Texts, inputs= self.savedCtcInput, sequence_length= self.seqLen, ctc_merge_repeated= True)
        
        # decorer: best path decoding
        if self.decorderType == Decorder_Type.BestPath:
            decoder = tf.nn.ctc_greedy_decoder(inputs= self.ctcInBAC, sequence_length= self.seqLen)
        
        # returning a CTC operation to compute the loss and a CTC operation to decode the RNN output
        return (tf.reduce_mean(loss), decoder)
        
    
    def TF(self):
        
        # initializing TF
        print("Python: "+sys.version)
        print("Tensorflow: "+tf.__version__)
        
        # Tensorflow session
        session = tf.compat.v1.Session()
        
        #saver saves the model to file
        saver = tf.compat.v1.train.Saver(max_to_keep= 1)
        #dataDir = '/Users/gjsiv/Downloads/Assignments/Project/Final Code/Data/'
        
        # we are checking, is their any saved model.
        LatestSnapshot = tf.train.latest_checkpoint(data_folder)
        
        # if model must be restored (for inference), there must be a snapshot
        if self.mustRestore and not LatestSnapshot:
            raise Exception('No saved model found in: ' + data_folder)
        
        # Load saved model if available
        if LatestSnapshot:
            print("Init with stored values from " + LatestSnapshot)
            saver.restore(session, LatestSnapshot)
        else:
            print('Init with new values')
            session.run(tf.global_variables_initializer())
            
        return (session, saver)
    
    def toSparse(self, texts):
        
        # put ground truth texts into sparse tensor for ctc_loss
        indices = []
        values = []
        # last entry must be max(labelList[i])
        shape = [len(texts), 0]
        
        # go over all texts
        for (batchElement, text) in enumerate(texts):
            
            # convert to string of label (i.e. class-ids)
            labelStr = [self.charList.index(c) for c in text]
            
            # sparse tensor must have size of max. label-string
            if len(labelStr) > shape[1]:
                shape[1] = len(labelStr)
            
            # put each label into sparse tensor
            for (i, label) in enumerate(labelStr):
                
                indices.append([batchElement, i])
                values.append(label)
        
        return (indices, values, shape)
    
    
    def decodeOutputToText(self, ctcOutput, batchSize):
        
        # extracts texts from output of CTC decoder
        
        # contains string of labels for each batch element
        encodeLabelStrs = [[] for i in range(Model.batchSize)]
        
        # TF decoders: label strings are contained in sparse tensor
        if self.decorderType == Decorder_Type.BestPath:
            
            # ctc returns tuple, first element is SparseTensor
            decoded = ctcOutput[0][0]
            
            # go over all indices and save mapping: batch -> values
            idxDict = { b : [] for b in range(Model.batchSize) }
            for (idx, idx2d) in enumerate(decoded.indices):
                label = decoded.values[idx]
                
                # index according to [b, t]
                batchElement = idx2d[0]
                encodeLabelStrs[batchElement].append(label)
         
        # map labels to chars for all batch elements
        return [str().join([self.charList[c] for c in labelStr]) for labelStr in encodeLabelStrs]

    
    def inferBatch(self, batch, calcProbability=False, probabilityOfGT= False):
        
        # feed a batch into the NN to recognize the texts - deocde, optionally save RNN output
        
        #print(len(batch.images))
        numBatchEle = len(batch.images)
        evalRNNOutput = calcProbability
        evalList = [self.decoder] + ([self.ctcInBAC] if evalRNNOutput else [])
        feedDict = {self.inputImages : batch.images, self.seqLen : [Model.maxTextlength] * numBatchEle, self.is_train: False}
        evalRes = self.session.run(evalList, feedDict)
        decoded = evalRes[0]
        text = self.decodeOutputToText(decoded, numBatchEle)
        
        # feed RNN output and recognized text into CTC loss to compute labeling probability
        probs = None
        if calcProbability:
            sparse = self.toSparse(batch.Texts) if probabilityOfGT else self.toSparse(text)
            ctcInput = evalRes[1]
            evalList = self.lossPerElement
            feedDict = {self.savedCtcInput : ctcInput, self.Texts : sparse, self.seqLen : [Model.maxTextlength] * numBatchEle, self.is_train: False}
            lossVals = self.session.run(evalList, feedDict)
            probs = np.exp(-lossVals)
        
        return (text, probs)

# -------------------------------------------- calculating the CER and WER -----------------------------------------------
        
class CER_WER:
    
    # CER and WER
    
    def __init__(self, wordData=r'\w'):
        
        self.numWords = 0
        self.numChars = 0
        
        self.edWords = 0
        self.edChars = 0
        
        self.Pattern = '[' + wordData + ']'
        
    def WordIDString(self, w1, w2):
        
        # get words in string 1 and string 2
        words1 = re.findall(self.Pattern, w1)
        words2 = re.findall(self.Pattern, w2)
        
        # find unique words
        allWords = list(set(words1 + words2))
        
        # list of words id's for string 1
        Str1 = []
        for i in words1:
            
            Str1.append(allWords.index(i))
        
        # list of words id's for string 2
        Str2 = []
        for j in words2:
            
            Str2.append(allWords.index(i))
        
        return (Str1, Str2)
    
    def Sample(self, outPut, label):
        
        # insert result and ground truth for next sample
        
        # charaters
        self.edChars += editdistance.eval(outPut, label)
        self.numChars += len(outPut)
        
        # words
        (StroutPut, StrLabel) = self.WordIDString(outPut, label)
        self.edWords += editdistance.eval(StroutPut, StrLabel)
        self.numWords += len(StroutPut)
    
    def CER(self):
        
        # get the Character Error Rate
        return self.edChars / self.numChars
    
    def WER(self):
        
        # get the Word Error Rate
        return self.edWords / self.numWords

# -------------------------------------------- Preprocess the Image -------------------------------------------------------

def preprocessImage(image, imageSize, dataAugmentation= False):
    
    # put image into target image of size imageSize, transpose for TF and normalize gray-values
    
    # there are damaged files in IAM dataset - just use black image instead
    if image is None:
        image = np.zeros([imageSize[1], imageSize[0]])
        
    # increase dataset size by applying random stretches to the images
    if dataAugmentation:
        
        stretch = (random.random() - 0.5)
        
        # random width, but atleast 1
        wStretched = max(int(image.shape[1] * (1 + stretch)), 1)
        
        # stretch horizontally by factor
        image = cv2.resize(image, (wStretched, image.shape[0]))
    
    # create target image and copy sample image into it
    (wt, ht) = imageSize
    (h, w) = image.shape
    fx = w / wt
    fy = h / ht
    f = max(fx, fy)
    
    # scale according to f - result at least 1 and at most wt or ht
    newSize = (max(min(wt, int(w / f)), 1), max(min(ht, int(h / f)), 1))
    image = cv2.resize(image, newSize)
    target = np.ones([ht, wt]) * 255
    target[0:newSize[1], 0:newSize[0]] = image
    
    # transpose for TF
    image = cv2.transpose(target)
    
    # normalize
    (m, s) = cv2.meanStdDev(image)
    m = m[0][0]
    s = s[0][0]
    image = image - m
    image = image / s if s>0 else image
    
    return image

# ------------------------------------------------------- Display the Text and Probaility ------------------------------

def infer(model, fnImage):
    
    # recognize text in image provided by file path
    image = preprocessImage(cv2.imread(fnImage, cv2.IMREAD_GRAYSCALE), Model.imageSize)
    
    # fill all batch elements with same input image
    batch = Batch(None, [image] * Model.batchSize)
    
    # recognize text
    recognized, probability = model.inferBatch(batch, True)
    
    # display the recognized text
    print("Recognized Text: ", '"' + recognized[0] + '"')
    
    # display the probability for the recognized text
    print("Probability of the Text Recognized: ", probability[0])
    
    return recognized[0]
    

# ------------------------------------- Declaring the Main Function -------------------------------------------

def main():
    
#    try:
    # calling the Decorder_Type class
    decorderType = Decorder_Type.BestPath
    
    # reading the charData text file
    element = open(charData).read()
    
    #print("Element: ", element)
    
    # calling the Model class
    model = Model(element, decorderType, mustRestore=True)
    
    # calling the CER_WER class
    metric = CER_WER()
    
    # reading the label_data text file
    label_data = open(LabelData).read()
    
    # splits the data in label_data using '\n'
    data_label = label_data.split('\n')
    
    # creating the dictonary
    dict_allLabel = {}
    
    # assigning key - image name and value - labelled data
    for i in range(0, len(data_label)):
        
        ele = data_label[i].split("===")
        dict_allLabel[ele[0].strip()] = ele[1].strip()
        
    #print(dict_allLabel)
    
    not_processed = []
    
    for (i, im) in enumerate(imageFiles):
        
        #print('Segmenting the words in Image %s'%im)
    
        image_folder_name = im
    
        print("Created Image Folder Name: ", image_folder_name)
        
        # calling the prepareImage function
        image = prepareImage(cv2.imread('/Users/gjsiv/Downloads/Assignments/Project/Final Code/Data/iam-handwriting-top50/data_subset/data_subset/%s'%im), 50)
        
        # the image is splitted using the wordSegment function
        wordSplit = wordSegment(image, kernelSize= 25, sigma= 11, theta= 7, minArea= 100)
        
#        plt.imshow(image, cmap='gray', interpolation= 'bicubic')
#        
#        plt.show()
        
        
        # creating the directory if the directory is not exist.
        if not os.path.exists('/Users/gjsiv/Downloads/Assignments/Project/Final Code/Output/Splitted Images/%s'%im):
            
            os.mkdir('/Users/gjsiv/Downloads/Assignments/Project/Final Code/Output/Splitted Images/%s'%im)
        
        #print('The Image is Segmented into %d words'%len(wordSplit))
        
        image_len_words = len(wordSplit)
        
        image_collection = []
        image_name = []
        
        for (k, L) in enumerate(wordSplit):
            
            (wordBox, wordImg) = L
            
            # getting the x, y, w, and h values from the wordBox
            (x, y, w, h) = wordBox
            
            # saving the image into the directory
            cv2.imwrite('/Users/gjsiv/Downloads/Assignments/Project/Final Code/Output/Splitted Images/%s/%d.png'%(im, k), wordImg)
            
#            plt.imshow(wordImg, cmap='gray', interpolation= 'bicubic')
#            
#            plt.show()
            
            image_collection.append(wordImg)
            image_name.append(k)
            
            cv2.rectangle(image, (x, y), (x+w, y+h), 0, 1)
        
        # saving the summary image to the directory
        cv2.imwrite('/Users/gjsiv/Downloads/Assignments/Project/Final Code/Output/Splitted Images/%s/summary.png'%im, image)
        
        # displaying the summary image
        summary_image = cv2.imread('/Users/gjsiv/Downloads/Assignments/Project/Final Code/Output/Splitted Images/%s/summary.png'%im)
        
#        plt.imshow(summary_image, cmap= 'gray', interpolation= 'bicubic')
#        
#        plt.show()
        
        # creating the folder in the location
        location = '/Users/gjsiv/Downloads/Assignments/Project/Final Code/Output/Splitted Images/' + image_folder_name + '/'
        
        image_file_location = os.listdir(location)
    
        #text_image = []
    
        result_text = ""
    
        for (i, im) in enumerate(image_file_location):
                
            #print("-----------------------------------------------------------------------------------")
            
            append_name = str(i) + ".png"
            
            #print("Image Name: ", append_name)
            
            for i in range(image_len_words):
                
                split_image_name = str(i) + '.png'
                
                # checking whether the image name is same name in the location
                if split_image_name == append_name:
                    
                    fnInfer = location + split_image_name
                    try: 
                        # calling the infer function
                        recordWord = infer(model, fnInfer)
                    
                        result_text = result_text + " " + recordWord
                        
                    except:
                        
                        print("Error Occured: Image Not able to Process")
                        result_text = ""
                if(result_text==""):
                    break
            if(result_text==""):
                break
        
        notepad_file = output_folder + '/Text Output Notepads/' + image_folder_name[: -4] + '.txt'
            
        # opening the notepad file
        file = open(notepad_file, 'w')
        
        if result_text != "":
            
            # displays the recognized text
            print("Text Recognized from the Image: ", result_text)
            
            final = "Text Recognized is : " + result_text
            
            # write the recognized text
            file.write(final)
        else:
            
            #print("Not able to Process this Image")
            final = "Not able to process this Image"
        
            # write the recognized text
            file.write(final)
        
            file.close()
            not_processed.append(image_folder_name)
            continue
        
        try: 
            # calling the sample function in CER_WER class
            metric.Sample(result_text, dict_allLabel[image_folder_name[: -4]])
            
            CER = str(float(metric.CER())*100)
            
            CER = "Character Error Rate: " + CER
            
            #print(CER)
            
            # gives the one line space
            file.write("\n")
            
            # writes the CER data into notepad
            file.write(CER)
            
            WER = float(metric.WER())*100
            
            WER = 100 - WER
            
            WER = "Word Error Rate: " + str(WER)
            
            #print(WER)
            
            # gives the one line space
            file.write("\n")
            
            # writes the WER data into notepad
            file.write(WER)
            
            file.close()
        
        except:
            # close the notepad file
            text_err = "Unable to Process CER and WER"
            
            file.write("\n")
            file.write(text_err)
            
            file.close()
    
    # dsiplays the images which is not processed
    print("List of Images which are unable to process: ", not_processed)
        
        
        
# ------------------------------ operating the Total file calling the main function which is declared below

main()

