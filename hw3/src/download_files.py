import os
import sys
import errno
import tarfile
import nltk

if sys.version_info >= (3,):
    from urllib.request import urlretrieve
else:
    from urllib import urlretrieve

DATA_DIR = 'Data'

# http://stackoverflow.com/questions/273192/how-to-check-if-a-directory-exists-and-create-it-if-necessary
def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

# adapted from http://stackoverflow.com/questions/51212/how-to-write-a-download-progress-indicator-in-python
def dl_progress_hook(count, blockSize, totalSize):
    percent = int(count * blockSize * 100 / totalSize)
    sys.stdout.write("\r" + "...%d%%" % percent)
    sys.stdout.flush()

print('=============== Downloading Data and Models ===============')

########################################################
print('== Skipthoughts models ==')
SKIPTHOUGHTS_DIR = os.path.join(DATA_DIR, 'skipthoughts')
SKIPTHOUGHTS_BASE_URL = 'http://www.cs.toronto.edu/~rkiros/models/'
make_sure_path_exists(SKIPTHOUGHTS_DIR)
skipthoughts_files = [
    'dictionary.txt', 'utable.npy', 'btable.npy', 'uni_skip.npz', 'uni_skip.npz.pkl', 'bi_skip.npz',
    'bi_skip.npz.pkl',
]
for filename in skipthoughts_files:
    src_url = SKIPTHOUGHTS_BASE_URL + filename
    print('Downloading ' + src_url)
    urlretrieve(src_url, os.path.join(SKIPTHOUGHTS_DIR, filename),
                reporthook=dl_progress_hook)

########################################################
print('== NLTK pre-trained Punkt tokenizer for English ==')
nltk.download('punkt')

########################################################
print('== Pretrained model ==')
MODEL_DIR = os.path.join(DATA_DIR, 'Models')
make_sure_path_exists(MODEL_DIR)
pretrained_model_filename = 'latest_faces_model.ckpt.data-00000-of-00001'
src_url = 'https://www.space.ntu.edu.tw/webrelay/directdownload/9zwuzrhn1xl7cc/?dis=10014&fi=48158036' #'https://www.space.ntu.edu.tw/navigate/s/7F19A05FFC1B4D8C92D904D6F2311803QQY'
# src_url = 'https://bitbucket.org/paarth_neekhara/texttomimagemodel/raw/74a4bbaeee26fe31e148a54c4f495694680e2c31/' + pretrained_model_filename
print('Downloading ' + src_url)
urlretrieve(
    src_url,
    os.path.join(MODEL_DIR, pretrained_model_filename),
    reporthook=dl_progress_hook,
)
pretrained_model_filename = 'latest_faces_model.ckpt.index'
src_url = 'https://www.space.ntu.edu.tw/webrelay/directdownload/au3i45hrl9mr2na/?dis=10014&fi=48158795' #'https://www.space.ntu.edu.tw/navigate/s/680731D7911A4DC89FA8BE4551592733QQY'
# src_url = 'https://bitbucket.org/paarth_neekhara/texttomimagemodel/raw/74a4bbaeee26fe31e148a54c4f495694680e2c31/' + pretrained_model_filename
print('Downloading ' + src_url)
urlretrieve(
    src_url,
    os.path.join(MODEL_DIR, pretrained_model_filename),
    reporthook=dl_progress_hook,
)
pretrained_model_filename = 'latest_faces_model.ckpt.meta'
src_url = 'https://www.space.ntu.edu.tw/webrelay/directdownload/au3i45hrl9mr2na/?dis=10014&fi=48158796' #'https://www.space.ntu.edu.tw/navigate/s/50EDF0BB7EE54F9C8ABD2958580EC74DQQY'
# src_url = 'https://bitbucket.org/paarth_neekhara/texttomimagemodel/raw/74a4bbaeee26fe31e148a54c4f495694680e2c31/' + pretrained_model_filename
print('Downloading ' + src_url)
urlretrieve(
    src_url,
    os.path.join(MODEL_DIR, pretrained_model_filename),
    reporthook=dl_progress_hook,
)

print('=============== Finished Downloading Data and Models ===============\n')