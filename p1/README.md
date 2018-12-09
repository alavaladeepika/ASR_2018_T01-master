# ASR_2018_T01
## Data pre processing
Run ```python import_timit.py --timit=./TIMIT --preprocessed=False```
to compute the features and store them in a folder.
This script also converts the [NIST "SPHERE" file format](https://www.isip.piconepress.com/projects/speech/software/tutorials/production/fundamentals/v1.0/section_02/s02_01_p04.html) to [WAVE PCM format](http://soundfile.sapp.org/doc/WaveFormat/).
If you have already converted the files, set ```--preprocessed=True``` to skip the conversion process.

## References:
- Mel Frequency Cepstral Coefficient (MFCC) tutorial :
    - [http://practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/](http://practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/)
- TIMIT related documents: 
    - [https://nvlpubs.nist.gov/nistpubs/Legacy/IR/nistir4930.pdf](https://nvlpubs.nist.gov/nistpubs/Legacy/IR/nistir4930.pdf) 
    - [https://github.com/philipperemy/timit](https://github.com/philipperemy/timit)
- Implementation references:
    - [http://scikit-learn.org/stable/index.html](http://scikit-learn.org/stable/index.html)
    - [http://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html](http://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html)
    - [http://www.pitt.edu/~naraehan/python2/pickling.html](http://www.pitt.edu/~naraehan/python2/pickling.html)
    - [https://github.com/belambert/asr-evaluation](https://github.com/belambert/asr-evaluation)
## This repo contains
- [x] Code to read files and compute MFCC features
- [x] Computing MFCC for time slices given in .PHN files
- [x] Dumping computed features to a folder
- [x] Dumping phone-wise features to a folder
- [x] GMM training
- [x] GMM model dumping
- [x] GMM evaluation
- [x] PER computation

## Steps to execute:
- 'python3 import_timit.py --n_delta=n' where n = 0, 1, 2 to generate mfcc, mfcc delta, mfcc delta delta phone-wise features respectively.
- 'train.ipynb' : run using jupyter notebook to GMMs with and without energy co-efficients
- 'python3 test.py' to obtain predicted labels as 'hyp_<model_name>', truth labels as 'ref_<model_name>', accuracy for each model in 'results.txt' in the 'results' folder.
- Once 'asr-evaluation' is installed(as in requirements.txt), run 'wer ref_<model_name> hyp_<model_name> > wer_<model_name>'. From this we obtain the wer for each model(stored in 'results/per/wer' folder). 
