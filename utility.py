from typing import Union, Tuple
import os
from glob import glob
import torch
from torchvision.transforms import functional as FT
import numpy as np
import pandas as pd
import PIL
from PIL import Image

def get_max3channels_uint(img : Union[Image.Image, torch.Tensor]):
    '''
    Converts the images to three channels if it has also the alpha channel,
    otherwise it remains with 1 channel. The convention for the images is:
    1. if the dtype is uint then the image has pixels in [0,255],
    2. if the dtype is float then the image has pixels in [0,1].

    We report the images to have the uint type and maximum 3 channels
    
    Args
    ----
    :param Image.Image|torch.Tensor img: the input image to be converted to the common
        base in the range [0,255]
    :returns the image which is reported in [0,255] with max 3 channels
    :raises TypeError if the input image is not a PIL Image or a torch.Tensor
    '''
    
    try:
        if not isinstance(img, (Image.Image or torch.Tensor)):
            raise TypeError('The input image has to be a PIL image or a torch.Tensor')
    except TypeError as type_err:
        print(type_err)
    # the input image can have 1 channel, 3 or 4 channels
    # the channels can contain integer and float types
    if type(img) is Image.Image:
        pixel = img.getpixel((0,0))
        if isinstance(pixel, tuple) and len(pixel)==4:
            # remove the alpha channel
            img.convert('RGB')
        if isinstance(pixel, float) or (isinstance(pixel, tuple) and type(pixel[0]) is float):
            img = np.asarray(img)
            # x : img = 255 : 1
            img = img * 255
            # convert to int
            img = img.astype('uint8')
            # convert to PIL image
            img = Image.fromarray(img)
    else:
        # remove the alpha channel
        if img.shape[0]==4:
            img = img[:3,:]
        if img.dtype is torch.float:
            img = img * 255
            img.dtype = torch.uint8
    return img

def pad_image(
        img,
        out_width=1024,
        padding_mode='constant',
        fill=-1,
    ):
    '''
    Takes in input both RGB images and gray scale ones and pad it with the median
    of the image. If it is not necessary to pad, it leaves the images ase they are.

    Args
    ----
    :param img Image.Image|torch.Tensor: the input image to be padded
    :param out_width int: the out width we want
    :param float or tuple(float, float, float) fill: if -1 it fills the pad with the median of the color in the
            image, otherwise with the color passed
    :rises TyoeError if we pass a img which is not Image.Image or torch.Tensor and
        if the fill is not an integer or a Tuple[int, int, int]
    '''
    assert type(fill) == (int or Tuple[int, int, int]), print('Fill has to be an integer or a 3-dim tuple' +
                                        'of integers')
    try:
        if not isinstance(img, (Image.Image, torch.Tensor)):
            raise TypeError("Got inappropriate image arg")
        
        img = get_max3channels_uint(img)
        if isinstance(img, Image.Image):
            pad_width = out_width - img.width
        else:
            pad_width = out_width - img.shape[-1]
        if pad_width == 0:
            return img
        if pad_width < 0:
            # raise ValueError(f"Pad width results negative: {pad_width}...\n" \
            #     f"Choose out_width >= {out_width-pad_width}.") 
            return img
        else:
            pad_width_left_border = int(0.5*pad_width)
            pad_width_right_border = pad_width - pad_width_left_border
            padding = (pad_width_left_border, 0, pad_width_right_border, 0) # left, top, right, bottom
        
        if fill == -1:
            if isinstance(img, Image.Image):
                # if it is a PIL image, we convert it to numpy array to get the median
                tmp = np.array(img)
                fill = int(np.median(tmp).item())
            else:
                fill = int(img.median().item())
    except TypeError as type_err:
        print(type_err)
            
    return FT.pad(img, padding, fill, padding_mode)

def resize_and_pad_images(dataset_path : str='.', new_height : int=128, new_width : int=1024, save_img : bool=True):
    '''
    Resizes images that are listed in textual files (which can have .ln or .txt 
    as extension), saving these in a folder named @Resized@ where it creates 
    different folders with the same name of the name of the files it founds 
    (generally these files are named 'train.ln/txt', 'val.ln/txt', 'test.ln/txt').

    The structure of the dataset is supposed with the folder 'lines' at the same 
    location of the partition files ('train.ln/txt', 'val.ln/txt', 'test.ln/txt')

    Args
    ----
    :param dataset_path str : the path where to find the files with the splits
        in training, validation and test. It could be the case you pass files with the
        a different name, in this case a folder with the name of the file is created.
        Inside this folders two folders are created @aratio_kept@ and @aratio_mod@.
        @aratio_kept@ has the images with the aspect ratio that is kept, @aratio_mod@
        contains the images which are modified in the aspect ratio
    :param new_height int: the new height we want for the images
    :param new_width int: the new width we want for the images that is reached by
        padding if the resulting resized image is smaller than it or padding otherwise
    :raises ValueError: if there are not files to take the samples from
    '''
    try:
        split_files = []
        type_file = ('*.ln','*.txt')
        
        for t in type_file:
            path_to_split = os.path.join(dataset_path,t)
            split_files.extend(glob(path_to_split))
        
        if len(split_files)==0:
            raise ValueError('No files are found!\nPlease provide a path with files')

        for f in split_files:
            count_img_out_width = 0
            path_resized = os.path.join(dataset_path, 'Resized')
            # we remove the extension
            path_to_resized_by_split = path_resized+'\\'+os.path.splitext(f.split('\\')[-1])[0]
            # create the folders with the files with aspect ratio kept and the
            # one with the aspect ratio modified 
            path_resized_aratio_kept = os.path.join(path_to_resized_by_split,'aratio_kept')
            path_resized_aratio_mod = os.path.join(path_to_resized_by_split,'aratio_mod')
            if not os.path.exists(path_resized):
                os.mkdir(path_resized)
            if not os.path.exists(path_to_resized_by_split):
                os.mkdir(path_to_resized_by_split)
            if not os.path.exists(path_resized_aratio_kept):
                os.mkdir(path_resized_aratio_kept) 
            if not os.path.exists(path_resized_aratio_mod):
                os.mkdir(path_resized_aratio_mod)

            with open(f, 'r') as x:
                nameimgs = x.readlines()
                for img_name in nameimgs:
                    aspect_ratio_modified = False
                    img_name = img_name.strip() # we remove '\n' from the name
                    path_to_img = os.path.join(dataset_path,'lines',img_name)
                    # we convert the image to gray
                    img = Image.open(path_to_img).convert('L')
                    width, height = img.size

                    resized_width = int((new_height*width)/height)
                    newsize = (resized_width, new_height)
                    img = img.resize(newsize)
                    # the height is mandatory, thus is the dimension we want for sure
                    # to take into account
                    # if the resized width given by the new height is smaller than
                    # the desidered width, we pad it to the desidered width @new_width@
                    # otherwise we resize it to the desidered width @new_width@
                    if resized_width < new_width:
                        img = pad_image(img, new_width)
                    else:
                        img = img.resize((new_width, new_height))
                        count_img_out_width += count_img_out_width
                        aspect_ratio_modified = True
                    if save_img:
                        if aspect_ratio_modified:
                            img.save(os.path.join(path_resized_aratio_mod,img_name))
                        else:
                            img.save(os.path.join(path_resized_aratio_kept,img_name))
            print('Number of files for ' + f.split('\\')[-1] + \
                f'\nwith width bigger than {new_width}: {count_img_out_width}')
            print('Percentage of files for ' + f.split('\\')[-1] + \
                f'\nwith width bigger than {new_width}: {(count_img_out_width/len(nameimgs))*100}')
    except ValueError as val_err:
        print(val_err)

def get_dataset_statistics(path : str = './', \
                           save_plk : bool = False, \
                           name_plk :str = 'dataset_stat') -> pd.DataFrame:
    '''
    Computes the statistics over a dataset that is contained in a folder.
    The dataset has to be composed of both the images (which can be either .png
    or .jpg) of lines of text and the textual files 
    with the transcription. It is possible to save the pd.DataFrame
    as a plk file.

    Args
    ----
    :parameter str path: the path to the folder containing the images of lines 
        and the textual files
    :parameter bool save_plk: if to save the plk file with the dataset information
    :parameter str name_plk: the name we want to give to the pd.DataFrame
    :returns pd.DataFrame: a dataframe containing information on the images and 
        textual files
    :raises an error if there are not textual files
    '''
    
    file_list = glob(os.path.join(path,'*.txt'))
    n_files = len(file_list)
    print(n_files)
    assert n_files != 0, print('There are not files in ' + path)

    text, line_length, image_width, image_height = [], [], [], []
    for file in file_list:
        with open(file, 'r') as f:
            t = f.read()
            l = len(t)

        path_to_img = os.path.splitext(file)[0]+'.png'
        try:
            image = Image.open(path_to_img)
        except IOError as io_err:
            print(io_err)
            path_to_img = os.path.splitext(file)[0]+'.jpg'
            image = Image.open(path_to_img)
        text.append(t)
        line_length.append(l)
        image_width.append(image.width)
        image_height.append(image.height)
    
    df = pd.DataFrame({
        'text':text,
        'line_length':line_length,
        'image_width':image_width,
        'image_height':image_height
    })
    print(df.describe())
    if save_plk:
        df.to_pickle(name_plk+'.plk')
    return df