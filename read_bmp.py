#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# how-to-run: python3 read_bmp.py sample.bmp text.txt
# 
import struct
import collections
import math
import numpy as np
from PIL import Image
#import matplotlib.pylab as plt
import copy 
import os

### https://pl.python.org/forum/index.php?topic=2259.5;wap2
### http://matthiaseisen.com/pp/patterns/p0201/
### https://www.blog.pythonlibrary.org/2017/10/11/convert-a-photo-to-black-and-white-in-python/

def read_header( bf_filepath ):
   with open( bf_filepath, 'br' ) as bf:
      bmpfile_header = bf.read(52)
   return bmpfile_header

def read_bf( bf_filepath ):
   
   with open( bf_filepath, 'br' ) as bf:
       bmpfile_header = {
           'bfType':      bf.read(2), # typ pliku bitmapa == BM
           'bfSize':      bf.read(4), # wielkość pliku
           'bfReserved1': bf.read(2), # pole zarezerwowane pierwsze
           'bfReserved2': bf.read(2), # pole zarezerwowane drugie
           'bfOffBits':   bf.read(4), # offset bitów w tablicy pikseli.
       }
 
       bmpfile_infoheader = {
           # BITMAPCOREHEADER
           # OS21XBITMAPHEADER
           'bcSize':        bf.read(4), # wielkość nagłówka DIB
           'bcWidth':       bf.read(4), # Szerokość obrazu w pikselach
           'bcHeight':      bf.read(4), # Wysokość obrazu w pikselach
           'bcPlanes':      bf.read(2), # Liczba warstw kolorów
           'bcBitCount':    bf.read(2), # Liczba bitów na piksel
       }
     
       bf_off_bits = struct.unpack( '<L', bmpfile_header['bfOffBits'])[0]
       bf.seek(bf_off_bits, 0)
 
       depth  = bmpfile_infoheader['bcBitCount'][0]
       width  = struct.unpack( '<L', bmpfile_infoheader['bcWidth'])[0]
       height = struct.unpack( '<L', bmpfile_infoheader['bcHeight'])[0]
 
       pixel_array = []
 
       if depth == 24:
           # Bitmapa (True Color) - trzy kolejne bajty (w sumie 24 bitów)
           # określają kolory trzech kolejnych składowych (RGB)
           # jednego piksela.
           for i in range( abs(height)):
               row = []
               for j in range(width):
                   # Należy zwrócić szczególną uwagę na zapis kolejnych bajtów
                   # danych obrazu: nie jest to [R, G, B], ale [B, G, R].
                   B = bf.read(1)[0]
                   G = bf.read(1)[0]
                   R = bf.read(1)[0]
                   row.append( (R,G,B) )
               pixel_array.append( row )

   # Dla dodatniej wartości bcHeight, bitmapa jest typu z dołu do góry i jej
   # punkt początkowy znajduje się w lewym dolnym rogu. Dla ujemnych bcHeight
   # bitmapa jest typu z góry na dół i jej pocztek znajduje się w lewym
   # górnym rogu.
   pixel_array = pixel_array if height < 0 else pixel_array[::-1]
   
   return  pixel_array

#   for(y in range(0, ymax-1)):
# 0x29 -> 41
def find_contour( pixel_array ):
  iter = 0
  cross_section = {}
  for row in pixel_array:
    iter = iter + 1
    cross_section[iter] = False
    for col in row:
      if(col == (0,0,0)):
        cross_section[iter] = True
        break
  return cross_section

'''
  if(transitions != []):
    return transitions
  else:
    return count_transitions(array, True, True)
'''


def extract_canals( pixel_array, canal_number, number_of_values ):
   canal = dict()
   for i in range(0, number_of_values + 1):
      canal[i] = 0
   for row in pixel_array:
       for pixel in row:
           byte = pixel[canal_number]
           canal[byte] = canal[byte] + 1
   canal = collections.OrderedDict(sorted(canal.items()))
   return canal

'''
    
    iterate over theta to get m1(theta) and m2(theta)
    also P(theta) is needed 

'''
def Otsu(grouping):
  ret_dict = {}
  all = 0
  for k, v in grouping.items():
    all += v
  #print(all)
  for outer_key, outer_value in grouping.items():
    accm1 = 0
    allm1 = 0
    accm2 = 0
    allm2 = 0
    current = outer_key
    for inner_key, inner_value in grouping.items():
      if inner_key <= current:
        accm1 += inner_key * inner_value
        allm1 += inner_value
      else:
        accm2 += inner_key * inner_value
        allm2 += inner_value    
    #print(accm2)
    #print(allm2)
    #print(accm1)
    #print(allm1)
    Ptheta1 = 0
    Ptheta2 = 0
    m1 = 0
    m2 = 0
    if(allm1 != 0):
      m1 = accm1/allm1
      Ptheta1 = allm1/all
    if(allm2 != 0):
      m2 = accm2/allm2
      Ptheta2 = allm2/all
    sigma_sq = Ptheta1 * Ptheta2 * pow(m1-m2, 2)
    #print(m1)
    #print(m2)
    #print(Ptheta1)
    #print(Ptheta2)
    #print(sigma_sq)
    #print("===")
    ret_dict[current] = sigma_sq

    midpoint = max(ret_dict.items(), key=lambda x: x[1]) 
  return midpoint[0]

def png2bmp(input, output):
  file_in = input
  img = Image.open(file_in)
  file_out = output
  img.save(file_out)

def grayscale( filename ):
  x=Image.open(filename,'r')
  x=x.convert('L') #makes picture grayscale
  x=x.convert("RGB")
  x.save('grayscale.bmp')

def apply_threshold( filename, border ):
  x=Image.open(filename,'r')
  x=x.point(lambda p: p > border and 255)
  x=x.convert("RGB")
  x.save('threshold.bmp')

def crop( filename, fname_output, area ):
    img = Image.open(filename,'r')
    w, h = img.size
    cropped_img = img.crop(area)
    cropped_img.save(fname_output)
    print("[INFO] CROP saved: " + fname_output)

def rotate( filename ):
    img = Image.open(filename,'r')
    assert(img.size == (210, 60))
    img = img.rotate(-90, expand=1)
    assert(img.size == (60, 210))
    img.save("rotated.bmp")

def split_image_in_half( filename, output1, output2 ):
    img = Image.open(filename,'r')
    w, h = img.size
    cropped_img = img.crop((0,0,int(w/2),h))
    cropped_img.save(output1)
    cropped_img = img.crop((int(w/2),0,w,h))
    cropped_img.save(output2)
    print("[INFO] SPLIT saved: " + output1)
    print("[INFO] SPLIT saved: " + output2)
'''
def download_captcha(fname):
    import requests
    import lxml.etree
    import lxml.html

    url = r'https://ebok.poog.pgnig.pl/login.php'
    data = requests.get(url).content
    root = lxml.html.fromstring(data)
    text = root.xpath("//img[@class='token']/@src")[0]
    name = root.xpath("//input[@id='token']/@value")[0]

    captcha = "https://ebok.poog.pgnig.pl/" + text
    pic = requests.get(captcha).content

    f = open(fname, 'wb')
    f.write(pic)
    f.close()

    return
'''
def count_transitions(array):
    prev = False
    curr = False
    temp = 0
    transitions = []
    if(array[1] == True):   ## this is the special case when the text starts from top first pixel
        temp = 1
        prev = True
        curr = True
    for k, v in array.items():
        if(k==1):
            continue
        prev = curr
        curr = v
        if(curr != prev):
            if(temp == 0):
                temp = k-1
            else:
                transitions.append((temp, k-1))
                temp = 0
    return transitions

def process_image(image_file, output):
        # convert image to grayscale
        print("[INFO] grayscale image")
        grayscale(image_file)
        # read image as pixel array
        pixel_array = read_bf('grayscale.bmp')
        # count number of occurences of different colors (256 is size of dictionary)
        canal = extract_canals(pixel_array, 0, 256)

        # apply Otsu algorithm for the threshold value
        thr = Otsu(canal)
        print(thr)

        '''
        # we can also plot histogram
        lists = sorted(otsu.items())
        x, y = zip(*lists)  # unpack a list of pairs into two tuples
        plt.plot(x, y)
        plt.show()
        print(otsu)
        '''

        # apply threshold value to grayscale.bmp and save as threshold.bmp
        print("[INFO] apply Otsu")
        apply_threshold('grayscale.bmp', thr)
        # read bitmap to pixel_array
        pixel_array = read_bf('threshold.bmp')


        yyy = find_contour(pixel_array)
        print("yyy" + str(yyy))
        transition = count_transitions(yyy)
        threshold_y_top = transition[0][0]
        threshold_y_bottom = transition[0][1]

        rotate('threshold.bmp')
        pixel_array = read_bf('rotated.bmp')

        xxx = find_contour(pixel_array)
        print("xxx" + str(xxx))
        transitions = count_transitions(xxx)
        
        print(transitions)
        if(len(transitions) not in [2,3,4]):
            print("error: " + str(len(transitions)))
        elif(len(transitions) == 4):
            print("4")
            crop('threshold.bmp', output+"1.bmp", (transitions[0][0], threshold_y_top, transitions[0][1], threshold_y_bottom) )
            crop('threshold.bmp', output+"2.bmp", (transitions[1][0], threshold_y_top, transitions[1][1], threshold_y_bottom) )
            crop('threshold.bmp', output+"3.bmp", (transitions[2][0], threshold_y_top, transitions[2][1], threshold_y_bottom) )
            crop('threshold.bmp', output+"4.bmp", (transitions[3][0], threshold_y_top, transitions[3][1], threshold_y_bottom) )

        elif(len(transitions) == 3):
            print("3")
            maximal = max(transitions, key=lambda x: (x[1]-x[0])) 
            #print(maximal)
            crop('threshold.bmp', 'tmp.bmp', (maximal[0], threshold_y_top, maximal[1], threshold_y_bottom) )
            split_image_in_half("tmp.bmp", output+"3.bmp", output+"4.bmp")
            transitions.remove(maximal)
            crop('threshold.bmp', output+"1.bmp", (transitions[0][0], threshold_y_top, transitions[0][1], threshold_y_bottom) )
            crop('threshold.bmp', output+"2.bmp", (transitions[1][0], threshold_y_top, transitions[1][1], threshold_y_bottom) )

        elif(len(transitions) == 2):
            print("2")
            maximal = max(transitions, key=lambda x: (x[1]-x[0])) 
            crop('threshold.bmp', 'tmp.bmp', (maximal[0], threshold_y_top, maximal[1], threshold_y_bottom) )
            split_image_in_half("tmp.bmp", output+"1.bmp", output+"2.bmp")
            transitions.remove(maximal)

            maximal = max(transitions, key=lambda x: (x[1]-x[0])) 
            crop('threshold.bmp', 'tmp.bmp', (maximal[0], threshold_y_top, maximal[1], threshold_y_bottom) )
            split_image_in_half("tmp.bmp", output+"3.bmp", output+"4.bmp")
            transitions.remove(maximal)

def hello123():
  print("hello123")


'''
import sys, glob, ntpath
if __name__ == "__main__":
    print("[INFO] download captcha")
    download_captcha('sample.png')
    process_image('sample.png', "")
    exit(0)

'''