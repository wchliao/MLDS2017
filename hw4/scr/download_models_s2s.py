import os
import sys
import errno
import tarfile

if sys.version_info >= (3,):
    from urllib.request import urlretrieve
else:
    from urllib import urlretrieve

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

print('=============== Downloading Vocabulary ===============')
MODEL_DIR = 'data/works/open_subtitles/data/'
make_sure_path_exists(MODEL_DIR)
pretrained_model_filename = 'chat.ids80000.in'
src_url = 'https://www.space.ntu.edu.tw/webrelay/directdownload/x85k19nnb1jnbu/?dis=10014&fi=48620058'
fname = MODEL_DIR + pretrained_model_filename
if os.path.isfile(fname):
    print('Already exists: '+ pretrained_model_filename)
else:
    print('Downloading: ' + pretrained_model_filename)
    urlretrieve(
        src_url,
        os.path.join(MODEL_DIR, pretrained_model_filename),
        reporthook=dl_progress_hook,
    )
pretrained_model_filename = 'chat.in'
src_url = 'https://www.space.ntu.edu.tw/webrelay/directdownload/abck8lnnabyqcm/?dis=10014&fi=48620075'
fname = MODEL_DIR + pretrained_model_filename
if os.path.isfile(fname):
    print('Already exists: '+ pretrained_model_filename)
else:
    print('Downloading: ' + pretrained_model_filename)
    urlretrieve(
        src_url,
        os.path.join(MODEL_DIR, pretrained_model_filename),
        reporthook=dl_progress_hook,
    )

print('=============== Downloading Models ===============')
MODEL_DIR = 'data/works/open_subtitles/nn_models/'
make_sure_path_exists(MODEL_DIR)
pretrained_model_filename = 'model.ckpt-405500.data-00000-of-00001'
src_url = 'https://www.space.ntu.edu.tw/webrelay/directdownload/vo76oqnn9o0kci/?dis=10014&fi=48620059'
fname = MODEL_DIR + pretrained_model_filename
if os.path.isfile(fname):
    print('Already exists: '+ pretrained_model_filename)
else:
    print('Downloading: ' + pretrained_model_filename)
    urlretrieve(
        src_url,
        os.path.join(MODEL_DIR, pretrained_model_filename),
        reporthook=dl_progress_hook,
    )
pretrained_model_filename = 'model.ckpt-405500.index'
src_url = 'https://www.space.ntu.edu.tw/webrelay/directdownload/h4gvwlnn95cqcb/?dis=10014&fi=48620060'
fname = MODEL_DIR + pretrained_model_filename
if os.path.isfile(fname):
    print('Already exists: '+ pretrained_model_filename)
else:
    print('Downloading: ' + pretrained_model_filename)
    urlretrieve(
        src_url,
        os.path.join(MODEL_DIR, pretrained_model_filename),
        reporthook=dl_progress_hook,
    )
pretrained_model_filename = 'model.ckpt-405500.meta'
src_url = 'https://www.space.ntu.edu.tw/webrelay/directdownload/pv6dkznn8re4bu/?dis=10014&fi=48620061'
fname = MODEL_DIR + pretrained_model_filename
if os.path.isfile(fname):
    print('Already exists: '+ pretrained_model_filename)
else:
    print('Downloading: ' + pretrained_model_filename)
    urlretrieve(
        src_url,
        os.path.join(MODEL_DIR, pretrained_model_filename),
        reporthook=dl_progress_hook,
    )