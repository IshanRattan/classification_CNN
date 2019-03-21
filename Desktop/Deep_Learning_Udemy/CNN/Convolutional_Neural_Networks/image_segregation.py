import os
import shutil

class ImageSegregator:

    def __init__(self, input_dict):
        self.input_dict = input_dict
        self.train_test_split, self.path_src, self.path_target = self._input_optimizer
        self.index = ''

    @property
    def _input_optimizer(self):
        split_ratio = self.input_dict['train_test_split'] if self.input_dict['train_test_split'] < 1 else float(self.input_dict['train_test_split'])/100
        path_src = self.input_dict['path_src'] if self.input_dict['path_src'].endswith('/') else self.input_dict['path_src'] + '/'
        path_target = self.input_dict['path_target'] if self.input_dict['path_target'].endswith('/') else self.input_dict[
                                                                                                         'path_target'] + '/'
        return split_ratio, path_src, path_target

    def folder_path_collector(self):
        """This function will take the path of your source folder and will make a list all the sub folders in it."""
        paths = map(lambda x: f'{self.path_src}{x}', os.listdir(self.path_src))
        for folder_path in paths:
            if folder_path.split('/')[-1].startswith('.'): # to handle system files
                print(f'Skipped following folder: {folder_path}.')
                continue
            os.chdir(folder_path)
            self._image_collector(folder_path)

    def _image_collector(self, folder_path):
        """Creates a list of all the items/images present in a folder and passes it on to segmenter."""
        images = os.listdir(folder_path)
        for index, image in enumerate(images, start=1):
            self.index = index
            if image.endswith('.jpg') or image.endswith('.png') or image.endswith('.jpeg'):
                try:
                    if round(index/len(images),2) > self.train_test_split:
                        category = 'test_set'
                    else:
                        category = 'training_set'
                    self._data_segmenter(category, folder_path.split('/')[-1], f"{folder_path}/{image}")
                    os.chdir(folder_path)
                except:
                    print(f'Segmentation failed for: {folder_path}/{image} while copying it to: {category}.')

    def _data_segmenter(self, category, subCategory, file_path):
        """Segments the data. Copies data to training_set/test_set based on train_test_split ratio specified."""
        if not os.path.exists(f'{self.path_target}{category}/{subCategory}'):
            os.makedirs(f'{self.path_target}{category}/{subCategory}')

        image_name = file_path.split('/')[-1]
        shutil.copy(file_path, f'{self.path_target}{category}/{subCategory}/{image_name}')
        self._image_labeller(f'{self.path_target}{category}/{subCategory}/{image_name}', subCategory, image_name)

    def _image_labeller(self, path, subCategory, name):
        """Labels the image with subcategory name."""
        os.chdir(path.replace(name, ''))
        os.rename(path, f'{subCategory}.{self.index}.jpg')

def initiate(input_dict):
    ImageSegregator(input_dict).folder_path_collector()


# train_test_split : The ratio in which you would want to divide your data. Can take ratio or percent.
# path_src : source folder where your data is present. Folder -> SubFolders -> Images
# path_target : target path where you would like to store your data. Output will be 2 folders(training_set & test_set)
inputVars = {'train_test_split' : 80,
             'path_src' : '/Users/dataset/101_ObjectCategories/',
             'path_target' : '/Users/dataset/CNN/'}

initiate(inputVars)

