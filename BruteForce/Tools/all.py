import numpy


EXPERIMENT='multinomial_basic_features_unbalanced'
    

# The name of the docker image (used to display docker command to copy apply files to windows host)
DOCKER_IMAGE_NAME = 'dev_ds_1'

# Some common resources
PANDAS_STORE = '../PandasStore'
KAGGLE_EXE = '/root/anaconda3/bin/kaggle'
PICKLE_EXTENSION = '.pkl'
CLEAN_TRAINING_DATA = 'clean_training'
CLEAN_CHALLENGE_DATA = 'clean_challenge'

# Other useful constants
ZIP_EXTENSION = '.zip'
SEP_IN_FILE_NAME = '!'
KAGGLE_GET_SUBMISSIONS_COMMAND = KAGGLE_EXE +' competitions submissions quora-question-pairs --csv'
EXCEL_PRECISION='%.6f'

# Try to have a minimal decoration in notebook output
######################################################
from IPython.display import IFrame, display, HTML

def print_html(s):
    display(HTML(s, metadata=dict(isolated=True)))
    
def start_small():
    display(HTML('<span><small>', metadata=dict(isolated=True)))

def print_small(s):
    print_html('<small>',s,'</small>')

def end_small():
    display(HTML('</small></span>', metadata=dict(isolated=True)))

def start_italic():
    display(HTML('<i>', metadata=dict(isolated=True)))

def print_italic(s):
    print_html('<i>'+s+'</i>')

def end_italic():
    display(HTML('</i>', metadata=dict(isolated=True)))

def start_bold():
    display(HTML('<b>', metadata=dict(isolated=True)))

def print_bold(s):
    print_html('<b>'+s+'</b>')

def end_bold():
    display(HTML('</b>', metadata=dict(isolated=True)))

def print_bullet(s):
    print_html('<li>'+s)

def print_section(s):
    print_bold(s)
    print_html('<HR>')

import time

def print_done(s,top=0):
    if top >0:
        print_html('<span style="color:LIMEGREEN"><small><b><i>'+s+' in '+str(round(time.time()-top,1))+' s</i></b><p></p></small></span>')
    else:
        print_html('<span style="color:LIMEGREEN"><small><b><i>'+s+'</i></b><p></p></small></span>')

def print_info(s):
    print_html('<span style="color:LIMEGREEN"><small>'+s+'</small></span>')
    
def print_warning(s):
    print_html('<span style="color:LIGHTSALMON"><small>'+s+'</small></span>')

## Plenty of small tools to help not creating a huge mess


# File tools
#####################

import os

def env_path():
    return '../'+EXPERIMENT

def env_file_name(file_name,ext=''):
    return env_path()+'/'+file_name+ext

def absolute_env_file_name(file_name,ext=''):
    return str(os.path.abspath(env_path()+'/'+file_name+ext))


# Setup basic things
#############################

#tqdm_pandas is needed only for old versions of tqdm
from tqdm import tqdm_pandas
from tqdm.autonotebook  import tqdm
import pandas

# Very important: experiment_name is the key for plenty of saves and references
##############################################################################

def prepare_environnement(experiment_name):
    global EXPERIMENT 

    EXPERIMENT = experiment_name
    print_section('Prepare %s environment in %s' % (EXPERIMENT,env_path()))
    # This is supposed to enhance default pandas display
    pandas.set_option('display.width',200)
    pandas.set_option('display.max_colwidth',200)

    # with recent versions use tqdm.pandas(desc="my bar!")
    # tqdm_pandas(tqdm())
    tqdm.pandas()

    if not os.path.exists(env_path()):
        print_info('Create %s' % env_path())
        os.mkdir(env_path())
    copy_from_pandas_store_if_missing(CLEAN_TRAINING_DATA)
    copy_from_pandas_store_if_missing(CLEAN_CHALLENGE_DATA)
    print_done('Done')
    print

## Tools to cache important and costly data
##############################################

from shutil import copyfile

def global_pandas_store_file_name(file_name,ext='.pkl'):
    return PANDAS_STORE +'/'+file_name+ext

def local_pandas_store_file_name(file_name):
    return env_file_name(file_name,PICKLE_EXTENSION)

def copy_from_pandas_store_if_missing(file_name):
    if not os.path.exists(local_pandas_store_file_name(file_name)):
        print_info('Make local copy of %s' % global_pandas_store_file_name(file_name))
        copyfile(global_pandas_store_file_name(file_name),local_pandas_store_file_name(file_name))

def load_dataframe(file_name):
    df = pandas.read_pickle(local_pandas_store_file_name(file_name))
    return df

def save_dataframe(df,file_name):
    print_info('Save %s' % file_name )
    df.to_pickle(local_pandas_store_file_name(file_name))

def load_or_build_dataframe(dataframe_name,file_name,builder,dataframe,param1=None):
    start = time.time()
    print_section('%s: Load or rebuild %s' % (dataframe_name,file_name))
    if os.path.exists(local_pandas_store_file_name(file_name)):
        print_info("!!!!! %s is cached!!!" % local_pandas_store_file_name(file_name))
        df = load_dataframe(file_name)
    else:
        print_warning("!!!!! %s does not exists!!!" % local_pandas_store_file_name(file_name))
        print_warning('Rebuild and save it')
        if param1 is None:
            df = builder(dataframe)
        else:
            df = builder(dataframe,param1)
        save_dataframe(df,file_name)
    print_done('Done:%s contains %d lines' % (file_name,len(df)),top=start)
    return df

# Code to generate all combination of numeric features
######################################################

from itertools import compress, product
from itertools import chain, combinations

def all_numeric_columns(dataframe):
    l = list()
    for name in dataframe.columns:
        if dataframe.dtypes[name] in ['int64','float64'] and name not in ['test_id','id','qid1','qid2','is_duplicate','weight']:
            l.append( name)
    return l

def all_float_columns(dataframe):
    l = list()
    for name in dataframe.columns:
        if dataframe.dtypes[name] in ['float64']:
            l.append( name)
    return l

def all_subsets(ss):
    return list(chain(*map(lambda x: combinations(ss, x), range(0, len(ss)+1))))

def clean_combination_name(c):
    return str(numpy.asarray(c)).replace("' '","+").replace("['","").replace("']","").replace("'","").replace(" ","").replace('\n','+')

def clean_all_combination_names(dataframe):
    for c in all_subsets(all_numeric_columns(dataframe)):
        if len(c)>0:
            print('|%s|'% clean_combination_name(c))

# Plot tools
#######################################

import matplotlib.pyplot as plot
import seaborn as sns

# code to put plots in a grid

def multiplot_from_generator(g, num_columns, figsize_for_one_row=None):
    # call 'next(g)' to get past the first 'yield'
    next(g)
    # default to 15-inch rows, with square subplots
    if figsize_for_one_row is None:
        figsize_for_one_row = (15, 15/num_columns)
    try:
        while True:
            # call plt.figure once per row
            plot.figure(figsize=figsize_for_one_row)
            for col in range(num_columns):
                ax = plot.subplot(1, num_columns, col+1)
                next(g)
    except StopIteration:
        pass

# apply tools
#######################################

def apply_file_name(criteria,kind,model_key,ext='.csv'):
    return env_file_name(SEP_IN_FILE_NAME.join((EXPERIMENT,criteria,kind,clean_combination_name(model_key).replace('/','_div_'))),ext=ext)

def apply_absolute_file_name(criteria,kind,model_key,ext='.csv'):
    return str(os.path.abspath(apply_file_name(criteria,kind,model_key,ext=ext)))

# Zip tools
#######################################

from zipfile import ZipFile,ZIP_DEFLATED 
import zlib
import os
from pathlib import Path

def zip_file_name(original_file_name):
    return str(Path(original_file_name).with_suffix('.zip'))

def zip_file_and_delete(original_file_name):
    zip_name = zip_file_name(original_file_name) 
    zip = ZipFile(str(zip_name), 'w',compression=ZIP_DEFLATED)
    zip.write(original_file_name)
    zip.close()
    os.unlink(original_file_name)
    return zip_name

# Excel tools
#######################################

def excel_file_name(file_name):
    return env_file_name(file_name,ext='.xlsx')

def pandas_to_excel(dataframe,file_name):
    dataframe.to_excel(excel_file_name(SEP_IN_FILE_NAME.join([EXPERIMENT,file_name])),float_format=EXCEL_PRECISION)

def save_models_dict_to_excel(results,file_name='all_models'):
    file_name = excel_file_name('_'.join([EXPERIMENT,file_name]))
    print_section('save %d results in %s' % (len(results),file_name))
    results.to_excel(file_name,float_format="%.4f")
    print_done("Done")

# Kaggle submissions tool
#######################################

import subprocess

def load_kaggle_submissions():
    print_section('Load all Kaggle submissions')
    generic_submissions_name = EXPERIMENT+'_submissions'
    file_name_csv = absolute_env_file_name(generic_submissions_name,ext='.csv')
    file_name_excel = absolute_env_file_name(excel_file_name(generic_submissions_name))
    csv_output = open(file_name_csv,"w")
    proc = subprocess.Popen(KAGGLE_GET_SUBMISSIONS_COMMAND.split(),stdout=csv_output)
    proc.wait()
    csv_output.close()
    print_info('All submissions are available in .csv&nbsp;&nbsp;format with %s' % file_name_csv)
    submissions = pandas.read_csv(file_name_csv,error_bad_lines=True,warn_bad_lines=True)
    # fix dataframe so it is more convenient to use
    #submissions = submissions[submissions['status']=='complete' & submissions['description']!='first xgboost']
    submissions = submissions[submissions['status']=='complete']
    submissions = submissions[['date','publicScore','privateScore','description','fileName']]
    submissions['date'] = pandas.to_datetime(submissions['date'])
    submissions['publicScore'] = submissions['publicScore'].astype('float64')
    submissions['privateScore'] = submissions['privateScore'].astype('float64')
    submissions.to_excel(file_name_excel,float_format="%.4f")
    print_info('All submissions are available in .xlsx format with %s' % file_name_excel)
    return submissions

def put_first(l,e):
    l.insert(0,e)
    return list(dict.fromkeys(l))
    
def get_best_submissions(submissions,n=3,metric='publicScore',):
    print_info('Best %d submissions based on %s' % (n,metric))
    return submissions.nsmallest(n,metric)[put_first(submissions.columns.values.tolist(),metric)]

def get_last_submissions(submissions,n=3):
    print_info('Last %d submissions' % n)
    return submissions.nlargest(n,'date')[put_first(submissions.columns.values.tolist(),'date')]




