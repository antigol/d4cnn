# pylint: disable=E1101,R,C
import os
import csv
import torch
import torch.utils.data
from astropy.io import fits
import tarfile
import glob


class GG(torch.utils.data.Dataset):
    '''
    Gravitational Lens Finding Challenge
    '''

    url_train = 'http://metcalf1.difa.unibo.it/blf-portal/data/DataChallenge1.1train.tar.gz'
    url_label = 'http://metcalf1.difa.unibo.it/blf-portal/data/challenge2trainingKey.csv'
    url_test = 'http://metcalf1.difa.unibo.it/blf-portal/data/DataChallenge1.1.tar.gz'

    def __init__(self, root, train=True, download=False, transform=None, target_transform=None):
        self.root = os.path.expanduser(root)

        self.dir = os.path.join(self.root, "DataChallenge2trainDoOver.0" if train else "DataChallenge2DoOver.0")
        self.transform = transform
        self.target_transform = target_transform

        if not self._check_exists():
            if download:
                self.download(train)
            else:
                raise RuntimeError('Dataset not found.' +
                                   ' You can use download=True to download it')

        self.n = len(glob.glob(os.path.join(self.dir, "Public/Band1/imageEUC_VIS-*.fits")))

        if train:
            with open(os.path.join(self.root, "challenge2trainingKey.csv"), 'rt') as f:
                reader = csv.reader(filter(lambda row: not row.startswith('#'), f))
                header = next(reader)
                self.labels = {}
                for row in [x for x in reader]:
                    self.labels[row[0]] = {key: value for key, value in zip(header[1:], row[1:])}
        else:
            self.labels = None

    def __getitem__(self, index):
        with fits.open(os.path.join(self.dir, "Public/Band1/imageEUC_VIS-{}.fits".format(100000 + index))) as hdul:
            v = hdul[0].data
        with fits.open(os.path.join(self.dir, "Public/Band2/imageEUC_J-{}.fits".format(100000 + index))) as hdul:
            j = hdul[0].data
        with fits.open(os.path.join(self.dir, "Public/Band3/imageEUC_H-{}.fits".format(100000 + index))) as hdul:
            h = hdul[0].data

        img = (v, j, h)

        if self.transform is not None:
            img = self.transform(img)

        if self.labels is not None:
            target = self.labels[str(100000 + index)]

            if self.target_transform is not None:
                target = self.target_transform(target)

            return img, target
        else:
            return img

    def __len__(self):
        return self.n

    def _check_exists(self):
        vs = glob.glob(os.path.join(self.dir, "Public/Band1/imageEUC_VIS-*.fits"))
        js = glob.glob(os.path.join(self.dir, "Public/Band2/imageEUC_J-*.fits"))
        hs = glob.glob(os.path.join(self.dir, "Public/Band3/imageEUC_H-*.fits"))
        return len(vs) > 0 and len(vs) == len(js) == len(hs)

    def _download(self, url):
        import requests

        filename = url.split('/')[-1]
        file_path = os.path.join(self.root, filename)

        if os.path.exists(file_path):
            return file_path

        print('Downloading ' + url)

        r = requests.get(url, stream=True)
        with open(file_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=16 * 1024 ** 2):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)
                    f.flush()

        return file_path

    def download(self, train):

        if self._check_exists():
            return

        # download files
        try:
            os.makedirs(self.root)
        except OSError as e:
            if e.errno == os.errno.EEXIST:
                pass
            else:
                raise

        url = self.url_train if train else self.url_test
        filename = self._download(url)

        tar = tarfile.open(filename, "r:gz")
        tar.extractall(self.root)
        tar.close()

        os.unlink(filename)

        if train:
            self._download(self.url_label)

        print('Done!')
