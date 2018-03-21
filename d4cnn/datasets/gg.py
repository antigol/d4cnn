# pylint: disable=E1101,R,C
import os
import csv
import torch
import torch.utils.data
from astropy.io import fits
import tarfile


class GG(torch.utils.data.Dataset):
    '''
    Gravitational Lens Finding Challenge
    '''

    url_train = 'http://metcalf1.difa.unibo.it/blf-portal/data/DataChallenge1.1train.tar.gz'
    url_label = 'http://metcalf1.difa.unibo.it/blf-portal/data/challenge2trainingKey.csv'
    url_test = 'http://metcalf1.difa.unibo.it/blf-portal/data/DataChallenge1.1.tar.gz'

    def __init__(self, root, train=True, download=False, transform=None, target_transform=None):
        self.root = os.path.expanduser(root)

        self.file = os.path.join(self.root, "DataChallenge1.1train.tar.gz" if train else "DataChallenge1.1.tar.gz")
        self.transform = transform
        self.target_transform = target_transform

        if download:
            self.download(train)

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        self.tar = tarfile.open(self.file, "r:gz")

        if train:
            with open(os.path.join(self.root, "challenge2trainingKey.csv"), 'rt') as f:
                reader = csv.reader(filter(lambda row: not row.startswith('#'), f))
                header = next(reader)
                self.labels = {}
                for row in [x for x in reader]:
                    self.labels[row[0]] = { key: value for key, value in zip(header[1:], row[1:])}
        else:
            self.labels = None

    def __del__(self):
        self.tar.close()

    def __getitem__(self, index):
        [vis] = fits.open(self.tar.extractfile("DataChallenge2trainDoOver.0/Public/Band1/imageEUC_VIS-{}.fits".format(100000 + index)))
        [j] = fits.open(self.tar.extractfile("DataChallenge2trainDoOver.0/Public/Band2/imageEUC_J-{}.fits".format(100000 + index)))
        [h] = fits.open(self.tar.extractfile("DataChallenge2trainDoOver.0/Public/Band3/imageEUC_H-{}.fits".format(100000 + index)))
        img = (vis, j, h)

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
        return len(self.files)

    def _check_exists(self):
        try:
            tar = tarfile.open(self.file, "r:gz")
            tar.close()
            return True
        except:
            return False

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
        self._download(url)

        if train:
            self._download(self.url_label)

        print('Done!')
