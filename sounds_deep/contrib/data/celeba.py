import gzip
import hashlib
import logging
import os
import sys
import tarfile
import urllib
import urllib.request
import zipfile
from subprocess import Popen
from contextlib import contextmanager

import numpy as np
from imageio import imread

log = logging.getLogger(__name__)

_ALIGNED_IMGS_URL = (
    'https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AADIKlz8PR9zr6Y20qbkunrba/Img/img_align_celeba.zip?dl=1',
    'b7e1990e1f046969bd4e49c6d804b93cd9be1646')

_PARTITIONS_URL = (
    'https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AADxLE5t6HqyD8sQCmzWJRcHa/Eval/list_eval_partition.txt?dl=1',
    'fb3d89825c49a2d389601eacb10d73815fd3c52d')

_ATTRIBUTES_URL = (
    'https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AAC7-uCaJkmPmvLX2_P5qy0ga/Anno/list_attr_celeba.txt?dl=1',
    'da6959c54754838f1a12cbb80ed9baba5618eddd')


class CelebA(object):
    '''
    Large-scale CelebFaces Attributes (CelebA) Dataset [1].
    http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

    References:
    [1]: Ziwei Liu, Ping Luo, Xiaogang Wang and Xiaoou Tang.
         Deep Learning Face Attributes in the Wild. Proceedings of
         International Conference on Computer Vision (ICCV), December, 2015.
    '''
    def __init__(self, data_dir):
        self.name = 'celeba'
        self.n_imgs = 202599
        self.data_dir = os.path.join(data_dir, self.name)
        self._npz_path = os.path.join(self.data_dir, self.name + '.npz')
        self.img_dir = os.path.join(self.data_dir, 'img_align_celeba')
        self._install()
        (self.train_idxs, self.val_idxs, self.test_idxs, self.attribute_names,
         self.attributes) = self._load()

    def _download(self, url, sha1):
        log.info('Downloading %s', url)
        filepath = download(url, self.data_dir)
        return filepath

    def __getitem__(self, idx):
        return self.img(idx)

    def img(self, idx):
        img_path = os.path.join(self.img_dir, '%.6d.jpg' % (idx + 1))
        return imread(img_path)

    def imgs(self):
        for i in range(self.n_imgs):
            yield self.img(i)

    def _install(self):
        checkpoint_file = os.path.join(self.data_dir, '__install_check')
        with checkpoint(checkpoint_file) as exists:
            if exists:
                return
            url, md5 = _ALIGNED_IMGS_URL
            filepath = self._download(url, md5)
            log.info('Unpacking %s', filepath)
            archive_extract(filepath, self.data_dir)

            url, md5 = _PARTITIONS_URL
            filepath = self._download(url, md5)
            partitions = [[], [], []]
            with open(filepath, 'r') as f:
                for i, line in enumerate(f):
                    img_name, partition = line.strip().split(' ')
                    if int(img_name[:6]) != i + 1:
                        raise ValueError('Parse error.')
                    partition = int(partition)
                    partitions[partition].append(i)
            train_idxs, val_idxs, test_idxs = map(np.array, partitions)

            url, md5 = _ATTRIBUTES_URL
            filepath = self._download(url, md5)
            attributes = []
            with open(filepath, 'r') as f:
                f.readline()
                attribute_names = f.readline().strip().split(' ')
                for i, line in enumerate(f):
                    fields = line.strip().replace('  ', ' ').split(' ')
                    img_name = fields[0]
                    if int(img_name[:6]) != i + 1:
                        raise ValueError('Parse error.')
                    attr_vec = np.array(list(map(int, fields[1:])))
                    attributes.append(attr_vec)
            attributes = np.array(attributes)

            with open(self._npz_path, 'wb') as f:
                np.savez(f,
                         train_idxs=train_idxs,
                         val_idxs=val_idxs,
                         test_idxs=test_idxs,
                         attribute_names=attribute_names,
                         attributes=attributes)

    def _load(self):
        with open(self._npz_path, 'rb') as f:
            dic = np.load(f)
            return (dic['train_idxs'], dic['val_idxs'], dic['test_idxs'],
                    dic['attribute_names'][()], dic['attributes'])


def download(url, target_dir, filename=None):
    require_dir(target_dir)
    if filename is None:
        filename = url_filename(url)
    filepath = os.path.join(target_dir, filename)
    if sys.version_info[0] > 2:
        urllib.request.urlretrieve(url, filepath)
    else:
        urllib.urlretrieve(url, filepath)
    return filepath


@contextmanager
def checkpoint(filepath):
    try:
        yield os.path.exists(filepath)
    finally:
        pass
    touch(filepath)


def archive_extract(filepath, target_dir):
    target_dir = os.path.abspath(target_dir)
    if tarfile.is_tarfile(filepath):
        with tarfile.open(filepath, 'r') as tarf:
            # Check that no files get extracted outside target_dir
            for name in tarf.getnames():
                abs_path = os.path.abspath(os.path.join(target_dir, name))
                if not abs_path.startswith(target_dir):
                    raise RuntimeError('Archive tries to extract files '
                                       'outside target_dir.')
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(tarf, target_dir)
    elif zipfile.is_zipfile(filepath):
        with zipfile.ZipFile(filepath, 'r') as zipf:
            zipf.extractall(target_dir)
    elif filepath[-3:].lower() == '.gz':
        with gzip.open(filepath, 'rb') as gzipf:
            with open(filepath[:-3], 'wb') as outf:
                outf.write(gzipf.read())
    elif filepath[-2:].lower() == '.z':
        if os.name != 'posix':
            raise NotImplementedError('Only Linux and Mac OS X support .Z '
                                      'compression.')
        cmd = 'gzip -d %s' % filepath
        retval = Popen(cmd, shell=True).wait()
        if retval != 0:
            raise RuntimeError('Archive file extraction failed for %s.' %
                               filepath)
    else:
        raise ValueError('% is not a supported archive file.' % filepath)


def touch(filepath, times=None):
    with open(filepath, 'a'):
        os.utime(filepath, times)


def require_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def url_filename(url):
    return url.split('/')[-1].split('#')[0].split('?')[0]
