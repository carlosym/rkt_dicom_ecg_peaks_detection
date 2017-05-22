import sys
import glob
import os
import getopt
import numpy as np
#import matplotlib.pyplot as plt
from pywt import wavedec
from scipy.interpolate import interp1d
import csv
import json

if len(sys.argv) == 3:
    ######################## EXPERIMENT 3: ECG PEAKS DETECTION #########################
    
    InputFolderPath = sys.argv[1]  # "./../Annotated images/Normal/"
    OutputFolderPath = sys.argv[2]  # "./../Annotated images/Normal/out/"

    print(InputFolderPath)
    print(OutputFolderPath)

    #InputFolderPath = "/home/"+InputFolderPath
    #OutputFolderPath = "/home/"+OutputFolderPath

    
    for imgpath in glob.glob(InputFolderPath+"*.csv"):
        ######################## 1. Reading csv file (segmented ECG) #######################

        inputpath = InputFolderPath + "/" + os.path.basename(imgpath)
        print("Input: ", inputpath)
        outputpath = OutputFolderPath + "/" + os.path.basename(imgpath)
        print("output:", outputpath)
        
        #inputpath = sys.argv[1]
        #print ("Input: ", inputpath)
        #print()

        xList = [] # x-coords of the ecg
        ecgList = [] # y-coords (heights) of the ecg

        f = open(inputpath, 'r') # csv file with normalized ECG obtained in experiment1
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            xList.append(int(row[0]))
            ecgList.append(int(row[1]))
        f.close()
        ####################################################################################
        
        ######################## 2. Finding ECG peaks ########################
        xArray = np.array(xList)
        ecgArray = np.array(ecgList)

        ####### BASED ON SERGIO'S CODE (MATLAB) #######
        # Decomposition of the ECG using different wavelet levels. We use Daubechies1

        # 1) Multilevel wavelet decomposition
        levels = 5
    
        C = wavedec(ecgList, 'db1', level=levels) # list of arrays

        # 2) Reading and resampling the 2 bottom levels
        filtered_1a = C[levels-1]
        filtered_2a = C[levels] # both are arrays

        #NOTE: MATLAB and Python function 'wavedec' express vector C differently! 
        # For 5 levels:
        # MATLAB: C = [cA5, cD5, cD4, cD3, cD2, cD1] (array)
        # Python: C = [[cA5], [cD5], [cD4], [cD3], [cD2], [cD1]] (list of arrays)
        
        # For that reason, if we want to do it LITERALLY like the original MATLAB code:
        # 1)
        #C2 = wavedec(ecgList, 'db1', level=levels) # list of arrays
        #C = np.concatenate(C2).ravel().tolist() # wavelet decomposition vector (list)
        #L = [len(x) for x in C2]
        #L.append(len(ecgList)) # bookkeeping vector (list)

        # 2) 
        #filtered_1a = C[(len(C) - L[levels] - L[levels-1]+1)- 1 : (len(C) - L[levels] + 1)- 1]
        #filtered_2a = C[(len(C) - L[levels]+1)- 1 :] # both are lists
        # NOTE: see http://stackoverflow.com/questions/12218796/python-slice-first-and-last-element-in-list
        # for slicing of python lists

        x_new = np.arange(len(ecgList)) # [0, len(ecgList)) == [0, len(ecgList)-1]
        
        time_1 = np.linspace(0,len(ecgList),len(filtered_1a), endpoint=True)
        filtered_1 = interp1d(time_1,filtered_1a)(x_new) # array

        time_2 = np.linspace(0,len(ecgList),len(filtered_2a), endpoint=True)
        filtered_2 = interp1d(time_2,filtered_2a)(x_new) # array

        # 3) We work with the sum of the two bottom levels
        filtered = filtered_1 + filtered_2

        # 4) Definition of the cutoff value
        filtered = filtered - np.mean(filtered)
        cutoff = max(filtered)/2.4

        # 5)  Searching for peaks on the filtered signal 
        p = np.arange(len(filtered))
        inf = np.array([np.inf])
        v = np.concatenate((inf,filtered))
        b = np.diff(v)!=0
        aa = filtered[b] # values of 'filtered' in which the derivative is not 0
        pp = p[b] # indexs of 'filtered' in which the derivative is not 0
        m1 = aa>np.concatenate((aa[1:], inf))
        m2 = aa>np.concatenate((inf,aa[:len(aa)-1]))
        mask = (m1) & (m2)
        QRS_peaks = pp[mask]
            
        # Just consider those peaks above the cutoff value defined above
        peaks = filtered[QRS_peaks] > cutoff
        QRS_peaks = QRS_peaks[peaks==True].tolist()

        # The found peaks may not exactly agree with the peak values of the ECG signal.
        # We will find the max value in the vecinity of the QRS_peak value  
        prev = 20
        post = 20
        QRS_peaks_2 = []

        for i in range(len(QRS_peaks)):
            
            # To prevent from exceeding the ECG array
            if QRS_peaks[i]+post <= len(ecgList) and QRS_peaks[i]-prev >= 1:
                interval = abs(ecgArray[QRS_peaks[i]-prev : QRS_peaks[i]+post + 1])         

            elif QRS_peaks[i]+post > len(ecgList):
                interval = abs(ecgArray[QRS_peaks[i]-prev :])

            elif QRS_peaks[i]-prev < 1:
                interval = abs(ecgArray[: QRS_peaks[i]+post + 1])
            
            ind_max = interval.argmax(0) # '(0) because we only want the first index that we obtain   
            QRS_peaks_2.append(QRS_peaks[i] - prev + ind_max)

        # If the peaks/cardiac cycles do not fulfill any of the following conditions, 
        # the user will be able to choose the peaks manually:

        # 1st condition: minimum distance of 20 samples between found peaks
        # 2nd condition: no more than 4 cardiac cycles in an ECG signal 

        '''detection = 0
        dist_peaks = np.array(QRS_peaks_2[1:])-np.array(QRS_peaks_2[:-1]) # array with the distances between peaks
        
        while(detection == 0):
            if np.any(dist_peaks < 20) or len(QRS_peaks_2)-1 >= 4:
                print('--> The found peaks are not correct. You will have to choose manually the peaks...')
                print('CLOSE THE PLOT to obtain the cardiac cycles')
                #detection = 2 # THE MANUAL DETECTION WILL BE DONE VIA WEB
                detection = 1
                
            else: 
                print('--> There are less than 4 cardiac cycles, so the results are correct:')
                print('CLOSE THE PLOT to obtain the cardiac cycles')
                #detection = 1
                detection = 1
        '''
        ####################################################################################

        ##################### EXPERIMENT 4: CARDIAC CYCLE DETECTION #####################
        detection = 1

        if detection == 2:
            print('END of the code')

        elif detection == 1:
            num_row = len(QRS_peaks_2) - 1 # '-1' because the last peak does not form any cardiac cycle
            num_col = 3 

            # INITIALIZATION OF THE CSV AND JSON DATA

            # cardiac_cycles_csv --> [[1, onset1, offset1],[2, onset2, offset2],..., [num_row-1, final_onset, final_offset]]
            cardiac_cycles_csv = [[0 for i in range(num_col)] for j in range(num_row)]
            
            # cardiac_cycles_json --> {"1":"onset1, offset1","2":"onset2, offset2", ..., "num_row-1":"final_onset, final_offset"}
            cardiac_cycles_json = {key: [] for key in range(num_row)} # Initialization of dictionaries -> 
            # -> See: http://stackoverflow.com/questions/11509721/how-do-i-initialize-a-dictionary-of-empty-lists-in-python
            
            # THEN, WE FILL 'cardiac_cycles_csv' AND 'cardiac_cycles.json'
            for i in range(num_row):

                onset = xArray[QRS_peaks_2[i]] # onset of the cardiac cycle "i" (x-coord)
                offset = xArray[QRS_peaks_2[i+1]] # offset of the cardiac cycle "i" (x-coord)

                # CSV
                cardiac_cycles_csv[i][0] = i+1 # "cardiac cycle number 'i'"
                cardiac_cycles_csv[i][1] = onset 
                cardiac_cycles_csv[i][2] = offset 

                # JSON
                cardiac_cycles_json[i] = str([onset,offset]).strip('[]') # This transforms the list [onset,offset] into the string 'onset,offset'

            #print('CSV: The coordinates of the cardiac cycles are:', cardiac_cycles_csv)
            #print('JSON: The coordinates of the cardiac cycles are:', cardiac_cycles_json)
        #################################################################################
        
            filename  = outputpath.split('_')[0]

        ######################## CSV FILE CREATION ############################
            
            wr = csv.writer(open('%s_exp3_4_cardiac_cycles.csv' % filename,'w'), delimiter='\t')
            wr.writerows(cardiac_cycles_csv)
            #print('Output: %s_cardiac_cycles.csv' % filename)
        ########################################################################

        ######################## JSON FILE CREATION ###########################
        
            jsonfile = open('%s_exp3_4_cardiac_cycles.json' % filename,'w')
            json.dump(cardiac_cycles_json,jsonfile)

            print('%s_exp3_4_cardiac_cycles.json' % filename)
        #########################################################################
        ###############################################################################################################################

else:
    print ('Execution : python exp3_4.py inputfilepath')