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

def load_global_dataframe(file_name):
    df = pandas.read_pickle(global_pandas_store_file_name(file_name))
    return df

def save_dataframe(df,file_name):
    print_info('Save %s' % file_name )
    df.to_pickle(local_pandas_store_file_name(file_name))

def save_global_dataframe(df,file_name):
    print_info('Save %s into global repository' % file_name )
    df.to_pickle(global_pandas_store_file_name(file_name))

def load_or_build_dataframe(message,file_name,builder,dataframe,param1=None):
    start = time.time()
    print_section('%s: Load or rebuild %s' % (message,file_name))
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

def save_models_dict_to_excel(results,tag = 'all_models'):
    file_name = excel_file_name('_'.join([EXPERIMENT,tag]))
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


def compute_metrics_model(algo_spec,model,input_df,target_df,suffix,sample_weight = None,show = True):
    if 'predicter' in algo_spec:
        prediction_df,prediction_proba_df = algo_spec['predicter'](model,input_df,target_df)
    else:
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

    start = time.time()
    if 'learner' in algo_spec:
        # something specific like XGBoost
        model = algo_spec['learner'](algo_spec,input_train,target_train,input_test,target_test,input_train_weight,input_test_weight,show=show)
    else:
        # More standard
        model = build_algorithm(algo_spec)
        # learn !!
        model.fit(input_train,target_train,sample_weight = input_train_weight)

    duration = time.time()-start
    infos = compute_metrics_model(algo_spec,model,input_test,target_test,suffix,sample_weight = input_test_weight,show=show)

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
    start = time.time()
    if 'learner' in algo_spec:
        # something specific like XGBoost
        model = algo_spec['learner'](algo_spec,input_full,target_full,None,None,input_full_weight,None,show=show)
    else:
        # More standard
        model = build_algorithm(algo_spec)
        # learn !!
        model.fit(input_full,target_full,sample_weight = input_full_weight)

    duration = time.time()-start
    infos = compute_metrics_model(algo_spec,model,input_full,target_full,SUFFIX,sample_weight = input_full_weight,show=show)
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
    infos = compute_metrics_model(algo_spec,model,input_test,target_test,suffix,sample_weight = input_test_weight,show=show)
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
    infos = compute_metrics_model(algo_spec,model,input_full,target_full,SUFFIX,sample_weight = input_full_weight,show=show)
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



    #####Cross validation

from sklearn import model_selection


PRINT_CROSS_INFOS_ON_MODEL = {
    'logloss_proba_best': '%.6f',
    'logloss_proba_mean': '%.6f',
    'accuracy_best': '%d',
    'accuracy_mean': '%.6f',
    'score_best': '%.6f',
    'score_mean': '%.6f',
    'time_mean': '%.1f',
    'nb_features': '%d'
    }

ALL_CROSS_INFOS_ON_MODEL_TEST = [
    'logloss_proba_mean','logloss_proba_std','logloss_proba_min','logloss_proba_max', 'logloss_proba_best', 'logloss_proba_worst','logloss_proba_fold_best','logloss_proba_fold_worst',
    'nb_features',
    'column_names',
    'accuracy_mean','accuracy_std','accuracy_min','accuracy_max','accuracy_best','accuracy_worst','accuracy_fold_best','accuracy_fold_worst',
    'score_mean','score_std','score_min','score_max','score_best','score_worst','score_fold_best','score_fold_worst'
    'time_mean',
    'logloss_proba_model_best',
    'logloss_proba_model_worst',
    'accuracy_model_best',
    'accuracy_model_worst',
    'score_model_best',
    'score_model_worst',
    'columns',
    ]


# bad design choice : a DataFrame can be more convenient than a dict
# But then, it is convenient to suppress all non numeric/string columns
def models_cross_dict_to_df(models_dict):
    # reorder also the columns in a way I use more convenient
    return pandas.DataFrame.from_dict(models_dict, orient='index').reindex(columns=ALL_CROSS_INFOS_ON_MODEL_TEST)
# return a dict of metrics with fold indication
def build_model_fold(algo_spec,fold,input_train,target_train,input_test,target_test,column_names,show=True):
    # '' as all infos are in an outer dict 
    return build_model_with_test(algo_spec,input_train,target_train,input_test,target_test,column_names,'',show=show)


def build_naivebayes_model_with_cross_validation(algo_spec,input,column_names,target,folds=5,show=True):
    if show:
        print_warning('Cross validation on %d stratified folds on %d rows' % (folds,len(input)))
    start=time.time()
    i = 1
    cross_validation = model_selection.StratifiedKFold(n_splits=folds,shuffle=True,random_state=42)
    metrics = dict()
    for (train_index,test_index),i in zip(cross_validation.split(input,target),range(1,folds+1)):
        if show:
            print_info("Fold %d/%d" % (i,folds))
        X_train, X_test = input[train_index[0]:len(train_index)-1], input[test_index[0]:len(test_index)-1]
        Y_train, Y_test = target[train_index[0]:len(train_index)-1], target[test_index[0]:len(test_index)-1]
        #metrics[i] = build_naivebayes_model_with_test(X_train,Y_train,X_test,Y_test,column_names,show=show)
        metrics[i] = build_model_fold(algo_spec,i,X_train,Y_train,X_test,Y_test,column_names,show=show)
    if show:
        print_done('Cross validation done',top=start)
    metrics_df = pandas.DataFrame.from_dict(metrics,orient='index')
    means = dict()
    for m in all_numeric_columns(metrics_df):
        means[m+'_mean'] = metrics_df[m].mean()
        means[m+'_std'] = metrics_df[m].std()
        means[m+'_max'] = metrics_df[m].max()
        means[m+'_min'] = metrics_df[m].min()
        if ('logloss' in m) or ('time' in m):
            best = metrics_df[m].idxmin()
            worst = metrics_df[m].idxmax()
            best_value = metrics_df[m].min()
            worst_value = metrics_df[m].max()
        else:
            best = metrics_df[m].idxmax()
            worst = metrics_df[m].idxmin()
            best_value = metrics_df[m].max()
            worst_value = metrics_df[m].min()
        if 'time' not in m:
            means[m+'_model_best'] = metrics_df['model'][best]
            means[m+'_model_worst'] = metrics_df['model'][worst]
            means[m+'_fold_best'] = best
            means[m+'_fold_worst'] = worst
            means[m+'_best'] = best_value
            means[m+'_worst'] = worst_value            
    return means
        
def build_model_with_cross_validation(algo_spec,dataframe,column_names,target,folds=5,show=True):
    infos = build_naivebayes_model_with_cross_validation(algo_spec,dataframe,column_names,target,folds=5,show=show)
    infos.update(
        {
            'nb_features':len(column_names),
            'column_names':clean_combination_name(column_names),
            'columns': column_names
        })
    return infos

def build_cross_validation_model_on_all_subset_of_simple_features(algo_spec,dataframe,target,folds=5):
    start = time.time()
    all_combinations = list(all_subsets(all_numeric_columns(dataframe)))
    steps_for_progress = int(len(all_combinations)/20)
    print_section('%s : Build all models (with test+full) on every combination of features - %d lines' % (EXPERIMENT,len(dataframe)))
    print_warning('With cross validation on %d folds' % folds)
    print_warning('%d*%d models built - only %d logged here' % (len(all_combinations),folds,(int(len(all_combinations)/steps_for_progress))))
    models_dict = dict()
    print_header_infos_model(PRINT_CROSS_INFOS_ON_MODEL)
    progress = tqdm(all_combinations)
    num_model = 0
    min_log_loss= 1000.0
    new_min = False
    for c in progress:
        #if (len(c)) ==1:
        if (len(c)) >0:
            infos = build_model_with_cross_validation(algo_spec,dataframe,c,target,show=False)
            #
            models_dict[clean_combination_name(c)] = infos
            if infos['logloss_proba_best']<min_log_loss:
                min_log_loss = infos['logloss_proba_best']
                print_info(format_model_infos('',PRINT_CROSS_INFOS_ON_MODEL,infos,''))
                new_min=True
            else:
                new_min = False
            # There is a smart panda progress bar but invisible in pdf
            # So try to minimize logs and still have some progress info
            if (num_model % steps_for_progress) == 0 and not new_min:
                  print_warning(format_model_infos('',PRINT_CROSS_INFOS_ON_MODEL,infos,''))
            num_model += 1
            progress.refresh()
    print_done('Done',top=start)
    # Design mistake : need to convert dict to dataframe :(
    return models_cross_dict_to_df(models_dict)

def save_models_dict_to_excel(results,tag = 'all_models'):
        file_name = excel_file_name('_'.join([EXPERIMENT,tag]))
        print_section('save %d results in %s' % (len(results),file_name))
        results.to_excel(file_name,float_format="%.4f")
        print_done("Done")


    ### Scoring

def retrieve_model(results,criteria,kind,model_key):
    # hack
    k = 'model_'+kind
    if k in results.columns:
        model = results['model_'+kind][model_key]
    else:
        model = results[criteria+'_model_'+kind][model_key]

    column_names = results['columns'][model_key]
    return model,numpy.asarray(column_names)
    
def find_best_models(results,top,criteria,kind):
    if 'logloss' in criteria:
        return results.nsmallest(top,criteria+'_'+kind)
    else:
        return results.nlargest(top,criteria+'_'+kind)


import datetime

def show_docker_cp_command(absolute_file_name):
    return 'docker cp '+ DOCKER_IMAGE_NAME+':'+zip_file_name(absolute_file_name)+ ' c:\\temp\\outputs'

def show_kaggle_command(file_name,model_key):
    return 'kaggle competitions submit quora-question-pairs -f "' + zip_file_name(file_name) +'" -m "' + file_name + ':'+model_key+'"'

def show_docker_cp_commands(results):
    print_section('Use these commands to transfer apply results to windows host')
    for c in results['file_name'].apply(show_docker_cp_command):
        print_warning(c)
    print_done("")

def show_kaggle_commands(results):
    print_section('Use these commands to submit apply results to kaggle')
    results.apply(lambda r: print_warning(show_kaggle_command(r.file_name,r.model_key)),axis=1)
    print_done("")


# return a dataframe fully ready to be converted in csv and published into kaggle
def applyout_file_name(criteria,kind,model_key,tag=None,n=None):
    # unfortunatley our design generated huge file names :(
    # so we have to remove model_key form file name
    if n is not None:
        now = str(n)+'_'+datetime.datetime.now().strftime("%m_%d_%H_%M_%S")
    else:
        now = datetime.datetime.now().strftime("%m_%d_%H_%M_%S")
    if tag is None:
        # absolute_file_name_csv = apply_absolute_file_name(criteria,kind,model_key)
        absolute_file_name_csv = apply_absolute_file_name(criteria,kind,now)
    else:
        #absolute_file_name_csv = apply_absolute_file_name(tag+'_'+criteria,kind,model_key)
        absolute_file_name_csv = apply_absolute_file_name(tag+'_'+criteria,kind,now)
    return absolute_file_name_csv

def simple_apply(results,criteria,kind,model_key,input_dataframe,proba=True,algo_spec=None):
    model,column_names=retrieve_model(results,criteria,kind,model_key)
    assert model is not None
    assert column_names is not None
    assert len(column_names) > 0
    input_for_prediction=input_dataframe[column_names]
    res = pandas.DataFrame()
    if 'test_id' in input_dataframe.columns:
        res['test_id']=input_dataframe['test_id']
    if algo_spec is None:
        assert model.predict_proba is not None, 'Bad type of model ?'
        if proba:
            res['is_duplicate'] = pandas.Series(model.predict_proba(input_for_prediction)[:,1],name='is_duplicate')
        else:
            res['is_duplicate'] = pandas.Series(model.predict(input_for_prediction)[:,1],name='is_duplicate')
    else:
        assert'scorer' in algo_spec
        res['is_duplicate'] = algo_spec['scorer'](algo_spec,model,input_for_prediction,proba=proba)
    return res  


def submit_model(results,criteria,kind,model_key,input_dataframe,tag=None,proba=True,show_how_to_publish=True,kaggle=True,algo_spec=None,n=None):
    absolute_file_name_csv = applyout_file_name(criteria,kind,model_key,tag=tag,n=n)
    print_info('Doing apply')
    prediction = simple_apply(results,criteria,kind,model_key,input_dataframe,proba=proba,algo_spec=algo_spec)
    print_info('Generating CSV file')
    prediction.to_csv(absolute_file_name_csv,index=False)
    print_info('Zipping file')
    absolute_file_name_zip = zip_file_and_delete(absolute_file_name_csv)
    print_info('%s is ready' % absolute_file_name_zip)
    if show_how_to_publish:
        if kaggle:
            print_warning('Use this commands to submit apply results to kaggle')
            print_warning(show_kaggle_command(absolute_file_name_zip,model_key))
        else:
            print_warning('Use this command to transfer apply _results to Windows host')
            print_warning(show_docker_cp_command(absolute_file_name_csv))
    return absolute_file_name_csv
    
    
    
def submit_best_models(results,top,criteria,kind,input_dataframe,proba=True,kaggle=True,algo_spec=None,tag=None):
    best_models = find_best_models(results,top,criteria,kind)
    best_models['model_key']=numpy.asarray(best_models.index)
    best_models['num']=numpy.arange(1,top+1)
    files = list()
    for mk,n in zip(best_models['model_key'],range(1,top+1)):
        f = submit_model(results,criteria,kind,mk,input_dataframe,proba=proba,show_how_to_publish=False,kaggle=kaggle,tag=tag,algo_spec=algo_spec,n=n)
        files.append(f)
    best_models['file_name'] = files
    best_models['docker']=best_models['file_name'].apply(show_docker_cp_command)
    best_models['kaggle']=best_models.apply(lambda r: show_kaggle_command(r.file_name,r.model_key),axis=1)
    print_done('Done')
    if kaggle:
        show_kaggle_commands(best_models)
    else:
        show_docker_cp_commands(best_models)
    return best_models