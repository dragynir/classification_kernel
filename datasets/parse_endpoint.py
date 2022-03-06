import os
import shutil
import pandas as pd
import numpy as np
import datetime


def get_file_datetime(f):
    d = datetime.datetime.fromtimestamp(os.path.getmtime(f))
    d = datetime.datetime(d.year, d.month, d.day)
    return d

def str_date2date(str_date):
    if str_date is None:
        return None

    d = datetime.datetime.strptime(str_date, '%d.%m.%Y').date()
    d = datetime.datetime.combine(d, datetime.time(0, 0))
    return d

def check_file_in_ranges(file_date, ranges):

    checks = []
    for r in ranges:
        in_range = False
        dates = [str_date2date(r[0]), str_date2date(r[1])]

        if dates[0] is None: # no start date
            assert dates[1] is not None, 'Wrong interval'
            if file_date <= dates[1]:
                in_range = True
        elif dates[1] is None: # no end date
            assert dates[0] is not None, 'Wrong interval'
            if file_date >= dates[0]:
                in_range = True
        elif (file_date >= dates[0]) and (file_date <= dates[1]):
            in_range = True

        checks.append(in_range)

    return any(checks)

def test_split_by_date(data_path, test_date_ranges, output_csv, copy_test_path=None, copy_crossval_path=None):

    df = pd.DataFrame({'imgpath': [], 'split': []})

    for folder in os.listdir(data_path):


        if copy_test_path:
            os.makedirs(os.path.join(copy_test_path, folder), exist_ok=True)
        if copy_crossval_path:
            os.makedirs(os.path.join(copy_crossval_path, folder), exist_ok=True)

        folder_path = os.path.join(data_path, folder)

        for img_name in os.listdir(folder_path):

            img_path = os.path.join(folder_path, img_name)
            file_date = get_file_datetime(img_path)

            if check_file_in_ranges(file_date, test_date_ranges):
                df = df.append({'imgpath': img_path, 'split': 'test'}, ignore_index=True)
                if copy_test_path:
                    cp_path = os.path.join(copy_test_path, folder, img_name)
                    shutil.copy(img_path, cp_path)

            else:
                df = df.append({'imgpath': img_path, 'split': 'crossval'}, ignore_index=True)
                if copy_crossval_path:
                    cp_path = os.path.join(copy_crossval_path, folder, img_name)
                    shutil.copy(img_path, cp_path)

    test_count = len(df[df['split'] == 'test'])
    crossval_count = len(df[df['split'] == 'crossval'])

    print(f'Test count: {test_count}')
    print(f'Crossval count: {crossval_count}')

    df.to_csv(output_csv)

if __name__ == '__main__':
    data_path = '/home/mborisov/data/' # path to folder with classes
    test_date_ranges = [('10.10.2021', None)] # set time intervals in which images will go to "copy_test_path"
    output_csv = './us/birdsy_split.csv' # where to sava images split 
    copy_test_path = '/home/mborisov/CLM/test/31_10_2021' # destination folder for images in "test_date_range"
    copy_crossval_path = None # # destination folder for images not in  "test_date_range"

    # select images from data_path and split in test and crossval by date (time intervals)
    test_split_by_date(data_path, test_date_ranges, output_csv, copy_test_path=copy_test_path, copy_crossval_path=copy_crossval_path)
