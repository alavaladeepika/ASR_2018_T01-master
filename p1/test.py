#!/usr/bin/env python

'''
    NAME    : LDC TIMIT Dataset
    URL     : https://catalog.ldc.upenn.edu/ldc93s1
    HOURS   : 5
    TYPE    : Read - English
    AUTHORS : Garofolo, John, et al.
    TYPE    : LDC Membership
    LICENCE : LDC User Agreement
'''

import errno
import os
from os import path
import sys
import tarfile
import fnmatch
import pandas as pd
import subprocess
import argparse
from mapping import phone_maps
import python_speech_features as psf
import scipy.io.wavfile as wav
import numpy as np
import warnings
from sklearn.externals import joblib
from sklearn.mixture import GaussianMixture 

warnings.filterwarnings('ignore',category=pd.io.pytables.PerformanceWarning)
timit_phone_map = phone_maps(mapping_file="kaldi_60_48_39.map")

def clean(word):
    # LC ALL & strip punctuation which are not required
    new = word.lower().replace('.', '')
    new = new.replace(',', '')
    new = new.replace(';', '')
    new = new.replace('"', '')
    new = new.replace('!', '')
    new = new.replace('?', '')
    new = new.replace(':', '')
    new = new.replace('-', '')
    return new

def compute_mfcc(wav_file, n_delta=0):
    mfcc_feat = psf.mfcc(wav_file)
    if(n_delta == 0):
        return(mfcc_feat)
    elif(n_delta == 1):
        return(np.hstack((mfcc_feat, psf.delta(mfcc_feat,1))))
    elif(n_delta == 2):
        return(np.hstack((mfcc_feat, psf.delta(mfcc_feat,1), psf.delta(mfcc_feat, 2))))
    else:
        return 0

def read_transcript(full_wav):
    trans_file = full_wav[:-8] + ".PHN"
    with open(trans_file, "r") as file:
        trans = file.readlines()
    durations = [ele.strip().split(" ")[:-1] for ele in trans]
    durations_int = []
    for duration in durations:
        durations_int.append([int(duration[0]), int(duration[1])])
    trans = [ele.strip().split(" ")[-1] for ele in trans]
    trans = [timit_phone_map.map_symbol_reduced(symbol=phoneme) for phoneme in trans]
    # trans = " ".join(trans)
    return trans, durations_int

def data_conv(args):
    # Assume data is downloaded from LDC - https://catalog.ldc.upenn.edu/ldc93s1
    # We convert the .WAV (NIST sphere format) into MSOFT .wav
    # creates _rif.wav as the new .wav file
    target = args.timit
    full_wavs = []
    for root, dirnames, filenames in os.walk(target):
        for filename in fnmatch.filter(filenames, "*.WAV"):
            sph_file = os.path.join(root, filename)
            wav_file = os.path.join(root, filename)[:-4] + "_rif.wav"
            full_wavs.append(wav_file)
            subprocess.check_call(["sox", sph_file, wav_file])

    print("Preprocessing Complete")
    print("Building features")
    return full_wavs

def testing_data(args,path, full_wavs):
    n_delta = 0
    if path == 'mfcc_delta':
        n_delta = 1
    elif path == 'mfcc_delta_delta':
        n_delta = 2
        
    mfcc_features = []
    mfcc_labels = []

    for full_wav in full_wavs:
        trans, durations = read_transcript(full_wav = full_wav)
        labels = []

        (sample_rate,wav_file) = wav.read(full_wav)
        mfcc_feats = compute_mfcc(wav_file[durations[0][0]:durations[0][1]], n_delta=n_delta)

        for i in range(len(mfcc_feats)):
            labels.append(trans[0])
        for index, chunk in enumerate(durations[1:]):
            mfcc_feat = compute_mfcc(wav_file[chunk[0]:chunk[1]], n_delta=n_delta)
            mfcc_feats = np.vstack((mfcc_feats, mfcc_feat))
            for i in range(len(mfcc_feat)):
                labels.append(trans[index])
        mfcc_features.extend(mfcc_feats)
        mfcc_labels.extend(labels)
    
    #Testing 
    mfcc_labels = np.array(mfcc_labels)
    phonemes = ['sil', 'sh', 'ih', 'hh', 'eh', 'jh', 'd', 'ah', 'k', 's', 'uw', '', 'n', 'g', 'r', 'w', 'aa', 'dx', 'er', 'l', 'y', 'uh', 'ae', 'm', 'oy', 'dh', 'iy', 'v', 'f', 't', 'ow', 'ch', 'b', 'ng', 'ay', 'th', 'ey', 'p', 'aw', 'z']
    mfcc_features = np.array(mfcc_features)
    l = ['i','ii'] 
    
    f = open("results/results.txt", "a")
    for ele in l:
        dir_list = os.listdir(path='./models/'+path+"/"+ele)
        dir_list.sort()
        for dir_path in dir_list:
            f1 = open("results/per/ref/ref_"+path+"_"+ele+"_"+dir_path+".txt",'a')
            f2 = open("results/per/hyp/hyp_"+path+"_"+ele+"_"+dir_path+".txt",'a')
            matches = 0
            gmms = []
            for phoneme in phonemes:
                gmm = joblib.load("./models/"+path+"/"+ele+"/"+dir_path+"/phn_"+phoneme+".pkl")
                gmms.append(gmm)
                
            # Score prediction
            # for mfcc_feature, mfcc_label in zip(mfcc_features,mfcc_labels):
            prob = []
            max_index = 0
            for gmm in gmms:
                prob.append(gmm.score_samples(mfcc_features))
            prob = np.array(prob)
            pred = np.argmax(prob, axis=0)
            for i in range(len(pred)):
                if phonemes[pred[i]] == mfcc_labels[i]:
                    matches+=1
                f1.write(mfcc_labels[i]+"\n")
                f2.write(phonemes[pred[i]]+'\n')
            
            #max_index = prob.index(max(prob))
            #pred_label = phonemes[max_index]
            # print ("Predicted label: "+pred_label)
            #if mfcc_label == pred_label:
            #    matches+=1
            f.write("/n")
            print ("No. of samples matched: "+str(matches))
            f.write("No. of samples matched: "+str(matches)+"\n")
            accuracy = (matches/mfcc_features.shape[0]) * 100
            print ("Accuracy using " + path + " features(" + ele + ") with " + dir_path + ": "+str(accuracy))
            f.write("Accuracy using " + path + " features(" + ele + ") with " + dir_path + ": "+str(accuracy)+"\n")
            f1.close()
            f2.close()
            
    f.close()
        
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--timit', type=str, default="./TIMIT/TEST",
                       help='TIMIT root directory')

    args = parser.parse_args()
    #print(args)
    #print("TIMIT path is: ", args.timit)
    full_wavs = data_conv(args)
    paths = ['mfcc','mfcc_delta','mfcc_delta_delta']
    for path in paths:
        testing_data(args, path, full_wavs)
    print("Completed "+path)