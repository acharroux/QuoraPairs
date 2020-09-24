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
    #print(s)
    
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

def print_info(s,top=0):
    if top >0:
        print_html('<span style="color:LIMEGREEN"><small>'+s+' in '+str(round(time.time()-top,1))+' s</small></span>')
    else:
        print_html('<span style="color:LIMEGREEN"><small>'+s+'</small></span>')
    
def print_warning(s,top=0):
    if top >0:
        print_html('<span style="color:LIGHTSALMON"><small>'+s+' in '+str(round(time.time()-top,1))+' s</small></span>')
    else:
        print_html('<span style="color:LIGHTSALMON"><small>'+s+'</small></span>')

def print_alert(s,top=0):
    if top >0:
        print_html('<span style="color:RED">'+s+' in '+str(round(time.time()-top,1))+' s</span>')
    else:
        print_html('<span style="color:RED">'+s+' </span>')



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
    tqdm.pandas(leave=True)

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



################################################### 
# Simple strategy : all defaults

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV


WEIGHT='weight'


# This will add a suffix to all keys of a dict
# Used to add _test,_train,_full to keys of infos about a model

PRINT_INFOS_ON_2_MODELS = {
    'accuracy_80_20': '%.4f',
    'score_80_20': '%.4f',
    'logloss_proba_80_20': '%.4f',
    'time_80_20': '%.1f',

    'accuracy_100_0': '%.4f',
    'score_100_0': '%.4f',
    'logloss_proba_100_0': '%.4f',
    'time_100_0': '%.1f',

    'nb_features': '%d',
    'column_names': '%s'
}

PRINT_INFOS_ON_MODEL = {
    'accuracy': '%.6f',
    'score': '%.6f',
    'logloss_proba': '%.6f',
    'time': '%.1f',
}


def add_suffix_to_keys(d,s):
    if s!='':
        return dict(zip([k+'_'+s for k in d.keys()],d.values()))
    else:
        return d

def add_suffix(m,s):
    if s!='':
        return '_'.join([m,s])
    else:
        return m

def format_model_infos(message,keys_formats,infos,suffix):
    values = list()
    if suffix != '':
        for k,f in keys_formats.items():
            values.append( f % infos[k+'_'+suffix])
    else:
        for k,f in keys_formats.items():
            values.append( f % infos[k])
    #return print_info( '%s %s' %(message,'&nbsp;|&nbsp;'.join(values)))
    return '%s %s' %(message,' | '.join(values))

def print_model_infos(message,keys_formats,infos,suffix):
    print_info(format_model_infos(message,keys_formats,infos,suffix))

def print_header_infos_model(key_formats):
    print_info('|'.join(key_formats.keys()))


def compute_metrics_model(model,input_df,target_df,suffix,sample_weight = None,show = True):
    prediction_df = model.predict(input_df)
    prediction_proba_df = model.predict_proba(input_df)
    res = metrics.classification_report(target_df,prediction_df,sample_weight = sample_weight,output_dict=True)
    accuracy = res['accuracy']
    score = res['weighted avg']['f1-score']
    logloss_proba = metrics.log_loss(target_df,prediction_proba_df,sample_weight = sample_weight)
    if show:
        print('Classification report on %s' % suffix)
        print(metrics.classification_report(target_df,prediction_df,sample_weight = sample_weight))
    return add_suffix_to_keys(
            {
             'accuracy':accuracy,
             'score':score,
             'logloss_proba':logloss_proba,
             'model':model
           },
           suffix)

#           

def  build_algorithm(algo_spec,show=False):
    if show:
        print_info('algorithm is:%s' % str(algo_spec['algorithm']()))
    return algo_spec['algorithm']()

def build_model_with_test(algo_spec,input_train,target_train,input_test,target_test,column_names,suffix,show=True):
    input_train_weight = None
    input_test_weight = None
    if WEIGHT in input_train.columns:
        input_train_weight = input_train[WEIGHT]
        input_test_weight = input_test[WEIGHT]

    # input_train & input_test must contains only features
    input_train = input_train[list(column_names)]
    input_test = input_test[list(column_names)]
    if show:
        if input_train_weight is not None:
            print_info('Model with weight')
        print_info( 'Training:%d lines Test:%d Nb Features: %d' % (len(input_train),len(input_test),len(input_train.columns)))
    # create the model
    model = build_algorithm(algo_spec)
    start = time.time()
    # learn !!
    model.fit(input_train,target_train,sample_weight = input_train_weight)
    duration = time.time()-start
    infos = compute_metrics_model(model,input_test,target_test,suffix,sample_weight = input_test_weight,show=show)
    infos.update({add_suffix('time',suffix):duration})
    if show:      
        print_model_infos(suffix,PRINT_INFOS_ON_MODEL,infos,suffix)
    return  infos
    
def build_model_100_0(algo_spec,input,column_names,target,show=True):
    SUFFIX = '100_0'
    input_full_weight = None
    if WEIGHT in input.columns:
        input_full_weight = input[WEIGHT]
    input_full = input[list(column_names)]
    target_full = target

    if show:
        if input_full_weight is not None:
            print_info('Model with weight')
        print_info( 'Training on %dx%d features' % (len(input_full),len(input_full.columns)))
    # create the model
    model = build_algorithm(algo_spec)
    start = time.time()
    # learn !!
    model.fit(input_full,target_full,sample_weight = input_full_weight)
    duration = time.time()-start
    infos = compute_metrics_model(model,input_full,target_full,SUFFIX,sample_weight = input_full_weight,show=show)
    infos.update({add_suffix('time',SUFFIX):duration})
    if show:
        print_model_infos(SUFFIX,PRINT_INFOS_ON_MODEL,infos,SUFFIX)
    return infos

def build_model_80_20(algo_spec,input,column_names,target,show=True):
    SUFFIX = '80_20'
    input_train,input_test,target_train,target_test = train_test_split(input,target,random_state=42,test_size=0.2)
    infos = build_model_with_test(algo_spec,input_train,target_train,input_test,target_test,column_names,SUFFIX,show=show)
    return infos

# old way to build model
# one on train+test=80+20
# one on full train
# all default parameters
def build_model_80_20_and_100_0(algo_spec,input,column_names,target,show=True):
    
    infos = build_model_80_20(algo_spec,input,column_names,target,show=show)
    infos.update(build_model_100_0(algo_spec,input,column_names,target,show=show))
    infos.update(
        {
            'nb_features':len(column_names),
            'column_names':clean_combination_name(column_names),
            'columns': column_names
        })
    return infos

def models_dict_default_to_df(models_dict):
    # reorder also the columns in a way I use more convenient
        return pandas.DataFrame.from_dict(models_dict, orient='index').reindex(columns=['logloss_proba_80_20','logloss_proba_100_0','nb_features','column_names','accuracy_80_20','accuracy_100_0','score_80_20','score_100_0','model_80_20','model_100_0','columns','time_80_20','time_100_0'])

def build_default_model_on_all_subset_of_simple_features(algo_spec,dataframe,target):
    start = time.time()
    all_combinations = list(all_subsets(all_numeric_columns(dataframe)))
    steps_for_progress = int(len(all_combinations)/20)
    print_section('%s : Build all models ((80,20)+(100,0)) on every combination of simple features - %d lines' % (EXPERIMENT,len(dataframe)))
    print_warning('%d*2 models built - only %d logged here' % (len(all_combinations),(int(len(all_combinations)/steps_for_progress))))
    models_dict = dict()
    print_header_infos_model(PRINT_INFOS_ON_2_MODELS)
    progress = tqdm(all_combinations)
    num_model = 0
    min_log_loss = 1000
    for c in progress:
        if (len(c)) >0:
            infos = build_model_80_20_and_100_0(algo_spec,dataframe,c,target,show=False)
            models_dict[clean_combination_name(c)] = infos
            # There is a smart panda progress bar but invisible in pdf
            # So try to minimize logs and still have some progress info
            if min(infos['logloss_proba_80_20'],infos['logloss_proba_100_0'])<min_log_loss:
                min_log_loss = min(infos['logloss_proba_80_20'],infos['logloss_proba_100_0'])
                print_info(format_model_infos('',PRINT_INFOS_ON_2_MODELS,infos,''))
                new_min=True
            else:
                new_min = False
            if (num_model % steps_for_progress) == 0 and not new_min:
                  print_warning(format_model_infos('',PRINT_INFOS_ON_2_MODELS,infos,''))
            num_model += 1
            progress.refresh()
    print_done('Done',top=start)
    # Design mistake : need to convert dict to dataframe :(
    return models_dict_default_to_df(models_dict)

################################################### 
# Hyper parameters

PRINT_INFOS_HYPER_ON_MODEL = {
    'accuracy': '%.6f',
    'score': '%.6f',
    'logloss_proba': '%.6f',
    'params': '%s'
}


def build_searcher(algo_spec,show=True):
    verbose = 0
    if show:
        verbose=3
    class_searcher = algo_spec['searcher']
    algorithm = algo_spec['algorithm']
    hyper_parameters = algo_spec['hyper_parameters']
    if show:
        print_info( 'Searcher : %s' % str(class_searcher(algorithm(),hyper_parameters)))
    if class_searcher == RandomizedSearchCV:
        return class_searcher(
            algorithm(),
            hyper_parameters,
            random_state=42,
            scoring='neg_log_loss',
            n_jobs=os.cpu_count(),
            pre_dispatch=2*os.cpu_count(),
            verbose=verbose,
            refit=True)
    else:
        return class_searcher(
            algorithm(),
            hyper_parameters,
            scoring='neg_log_loss',
            n_jobs=os.cpu_count(),
            pre_dispatch=2*os.cpu_count(),
            verbose=verbose,
            refit=True)


def build_model_with_test_hyper(algo_spec,input_train,target_train,input_test,target_test,column_names,suffix,show=True):
    input_train_weight = None
    input_test_weight = None
    if WEIGHT in input_train.columns:
        input_train_weight = input_train[WEIGHT]
        input_test_weight = input_test[WEIGHT]

    # input_train & input_test must contains only features
    input_train = input_train[list(column_names)]
    input_test = input_test[list(column_names)]
    if show:
        print_info('Model with hyper parameter')
        if input_train_weight is not None:
            print_info('Model with weight')
        print_info( 'Training:%d lines Test:%d Nb Features: %d' % (len(input_train),len(input_test),len(input_train.columns)))
    start = time.time()
    # Build the thing that will explore hyper parameters
    searcher = build_searcher(algo_spec,show=show)
    # Explore !!
    searcher.fit(input_train,target_train,sample_weight = input_train_weight)
    # Here is our best model 
    model = searcher.best_estimator_
    # Recompute our set of metrics
    infos = compute_metrics_model(model,input_test,target_test,suffix,sample_weight = input_test_weight,show=show)
    infos.update(
        {
            # keep a clear status of parameters found by searcher
            add_suffix('params',suffix):searcher.best_params_,
            add_suffix('time',suffix):time.time()-start
        })
    if show:      
        print_model_infos(suffix,PRINT_INFOS_HYPER_ON_MODEL,infos,suffix)
    return  infos
    
def build_model_100_0_hyper(algo_spec,input,column_names,target,show=True):
    SUFFIX = '100_0'
    input_full_weight = None
    if WEIGHT in input.columns:
        input_full_weight = input[WEIGHT]
    input_full = input[list(column_names)]
    target_full = target

    if show:
        print_info('Model with hyper parameters')
        if input_full_weight is not None:
            print_info('Model with weight')
        print_info( 'Training on %dx%d features' % (len(input_full),len(input_full.columns)))
    start = time.time()
    # Build the thing that will explore hyper parameters
    searcher = build_searcher(algo_spec,show=show)
    # Explore !!
    searcher.fit(input_full,target_full,sample_weight = input_full_weight)
    # Here is our best model
    model = searcher.best_estimator_
    # Recompute our set of metrics
    infos = compute_metrics_model(model,input_full,target_full,SUFFIX,sample_weight = input_full_weight,show=show)
    infos.update(
        {
            # keep a clear status of parameters found by searcher
            add_suffix('params',SUFFIX):searcher.best_params_,
            add_suffix('time',SUFFIX):time.time()-start
        })
    if show:
        print_model_infos(SUFFIX,PRINT_INFOS_HYPER_ON_MODEL,infos,SUFFIX)
    return infos

def build_model_80_20_hyper(algo_spec,input,column_names,target,show=True):
    SUFFIX = '80_20'
    input_train,input_test,target_train,target_test = train_test_split(input,target,random_state=42,test_size=0.2)
    infos = build_model_with_test_hyper(algo_spec,input_train,target_train,input_test,target_test,column_names,SUFFIX,show=show)
    return infos

# old way to build model
# one on train+test=80+20
# one on full train
def build_model_80_20_and_100_0_hyper(algo_spec,input,column_names,target,show=True):
    
    infos = build_model_80_20_hyper(algo_spec,input,column_names,target,show=show)
    infos.update(build_model_100_0_hyper(algo_spec,input,column_names,target,show=show))
    infos.update(
        {
            'nb_features':len(column_names),
            'column_names':clean_combination_name(column_names),
            'columns': column_names
        })
    return infos

PRINT_INFOS_HYPER_ON_2_MODELS = {
    'accuracy_80_20': '%.4f',
    'score_80_20': '%.4f',
    'logloss_proba_80_20': '%.4f',
    'params_80_20': '%s',

    'accuracy_100_0': '%.4f',
    'score_100_0': '%.4f',
    'logloss_proba_100_0': '%.4f',
    'params_100_0': '%s',

    'nb_features': '%d',
    'column_names': '%s'
}

INFOS_MODEL_TO_KEEP =  [
'logloss_proba_80_20',
'params_80_20',
'logloss_proba_100_0',
'params_100_0',
'nb_features',
'accuracy_80_20',
'accuracy_100_0',
'score_80_20',
'score_100_0',
'column_names',
'columns',
'model_100_0',
'model_80_20',
'time_80_20',
'time_100_0'
]

# bad design choice : a DataFrame can be more convenient than a dict
# But then, it is convenient to suppress all non numeric/string columns
def models_dict_hyper_to_df(models_dict):
    # reorder also the columns in a way I use more convenient
    return pandas.DataFrame.from_dict(models_dict,orient='index')[INFOS_MODEL_TO_KEEP]

def build_hyper_model_on_all_subset_of_simple_features(algo_spec,dataframe,target):
    start = time.time()
    all_combinations = list(all_subsets(all_numeric_columns(dataframe)))
    steps_for_progress = int(len(all_combinations)/20)
    print_section('%s : Build all models (80_20 & 100_0) on every combination of simple features - %d lines' % (EXPERIMENT,len(dataframe)))
    print_warning('%d*2 models optimized - only %d logged or best here. Watch for green' % (len(all_combinations),(int(len(all_combinations)/steps_for_progress))))
    models_dict = dict()
    print_header_infos_model(PRINT_INFOS_HYPER_ON_2_MODELS)
    progress = tqdm(all_combinations)
    min_log_loss=1000
    num_model = 0
    for c in progress:
        #if (len(c)) ==1:
        if (len(c)) >0:
            infos = build_model_80_20_and_100_0_hyper(algo_spec,dataframe,c,target,show=False)
            models_dict[clean_combination_name(c)] = infos
            if min(infos['logloss_proba_80_20'],infos['logloss_proba_100_0'])<min_log_loss:
                min_log_loss = min(infos['logloss_proba_80_20'],infos['logloss_proba_100_0'])
                print_info(format_model_infos('',PRINT_INFOS_HYPER_ON_2_MODELS,infos,''))
                new_min=True
            else:
                new_min = False
            # There is a smart panda progress bar but invisible in pdf
            # So try to minimize logs and still have some progress info
            if (num_model % steps_for_progress) == 0 and not new_min:
                  print_warning(format_model_infos('',PRINT_INFOS_HYPER_ON_2_MODELS,infos,''))
            num_model += 1
            progress.refresh()
    print_done('Done',top=start)
    # Design mistake : need to convert dict to dataframe :(
    return models_dict_hyper_to_df(models_dict)