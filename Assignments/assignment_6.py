'''
This is a template for your reproducible solution in the Alice competition.
It's obligatory that your script produces a submission file just 
by running `python solution_alice_<name>_<surname>.py`. 
If you have any dependecies apart from those in a Kaggle Docker image, 
it's your responsibility to provide an image (or at least a requirements file) 
to reproduce your solution.

Please avoid heavy hyperparameter optimization in this script. 

IMPORTANT: this script is to be shared only with organizers, as described in the
course roadmap https://mlcourse.ai/roadmap. Be careful not to share it in 
Kaggle Kernels, don't spoil the competitive spirit. 
'''

import os
import pickle
import numpy as np
import pandas as pd
import time
from collections import Counter
from contextlib import contextmanager
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression, SGDClassifier

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


PATH_TO_DATA = '../mlcourse.ai_Dataset/alice/'  # change if needed
AUTHOR = 'Vitalijs_Tickovs' # change here to <name>_<surname>
# it's a nice practice to define most of hyperparams here
SEED = 17
N_JOBS = 4
NUM_TIME_SPLITS = 10    # for time-based cross-validation
SITE_NGRAMS = (1, 5)    # site ngrams for "bag of sites"
MAX_FEATURES = 70000    # max features for "bag of sites"
BEST_LOGIT_C = 5.45559  # precomputed tuned C for logistic regression
LOGIT_C_GRID = [3.0, BEST_LOGIT_C, 7.5, 10.0, 12.0]
SGD_ALPHA_GRID = [1e-6, 3e-6, 1e-5]
if HAS_XGBOOST:
    XGB_PARAM_GRID = [
        {
            'n_estimators': [200],
            'learning_rate': [0.1],
            'max_depth': [4],
            'subsample': [0.9],
            'colsample_bytree': [0.9],
        },
        {
            'n_estimators': [300],
            'learning_rate': [0.05],
            'max_depth': [6],
            'subsample': [0.8],
            'colsample_bytree': [0.8],
        },
        {
            'n_estimators': [400],
            'learning_rate': [0.03],
            'max_depth': [7],
            'subsample': [0.85],
            'colsample_bytree': [0.85],
        },
    ]
else:
    XGB_PARAM_GRID = []
 

# nice way to report running times
@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f'[{name}] done in {time.time() - t0:.0f} s')


def prepare_sparse_features(path_to_train, path_to_test, path_to_site_dict,
                           vectorizer_params):
    times = ['time%s' % i for i in range(1, 11)]
    train_df = pd.read_csv(path_to_train,
                       index_col='session_id', parse_dates=times)
    test_df = pd.read_csv(path_to_test,
                      index_col='session_id', parse_dates=times)

    # Sort the data by time
    train_df = train_df.sort_values(by='time1')
    
    # read site -> id mapping provided by competition organizers 
    with open(path_to_site_dict, 'rb') as f:
        site2id = pickle.load(f)
    # create an inverse id _> site mapping
    id2site = {v:k for (k, v) in site2id.items()}
    # we treat site with id 0 as "unknown"
    id2site[0] = 'unknown'
    
    # Transform data into format which can be fed into TfidfVectorizer
    # This time we prefer to represent sessions with site names, not site ids. 
    # It's less efficient but thus it'll be more convenient to interpret model weights.
    sites = ['site%s' % i for i in range(1, 11)]
    train_sites = train_df[sites].fillna(0).astype('int')
    test_sites = test_df[sites].fillna(0).astype('int')

    def extract_site_tokens(site_name):
        tokens = [site_name]
        if site_name != 'unknown':
            parts = [part for part in site_name.split('.') if part]
            if len(parts) >= 2:
                tokens.append(f'domain_{parts[-2]}')
            if parts:
                tokens.append(f'tld_{parts[-1]}')
        return tokens

    site_token_map = {site_id: extract_site_tokens(site_name)
                      for site_id, site_name in id2site.items()}

    def build_sessions(site_matrix):
        sessions = []
        for row in site_matrix.values:
            session_tokens = []
            for site_id in row:
                site_id = int(site_id)
                if site_id <= 0:
                    continue
                session_tokens.extend(site_token_map.get(site_id, ['unknown']))
            if not session_tokens:
                session_tokens.append('unknown_session')
            sessions.append(' '.join(session_tokens))
        return sessions

    train_sessions = build_sessions(train_sites)
    test_sessions = build_sessions(test_sites)
    # we'll tell TfidfVectorizer that we'd like to split data by whitespaces only 
    # so that it doesn't split by dots (we wouldn't like to have 'mail.google.com' 
    # to be split into 'mail', 'google' and 'com')
    vectorizer = TfidfVectorizer(**vectorizer_params)
    X_train = vectorizer.fit_transform(train_sessions)
    X_test = vectorizer.transform(test_sessions)
    y_train = train_df['target'].astype('int').values
    
    # we'll need site visit times for further feature engineering
    train_times, test_times = train_df[times], test_df[times]
    
    return (X_train, X_test, y_train, vectorizer,
            train_times, test_times, train_sites, test_sites)


def add_features(times, site_ids):
    time1 = times['time1']
    hour = time1.dt.hour.fillna(-1).astype('int')
    morning = ((hour >= 7) & (hour <= 11)).astype('int').values.reshape(-1, 1)
    day = ((hour >= 12) & (hour <= 18)).astype('int').values.reshape(-1, 1)
    evening = ((hour >= 19) & (hour <= 23)).astype('int').values.reshape(-1, 1)
    night = ((hour >= 0) & (hour <= 6)).astype('int').values.reshape(-1, 1)

    sess_duration = (times.max(axis=1) - times.min(axis=1)).astype('timedelta64[s]') \
        .astype('int').values.reshape(-1, 1)
    weekday_series = time1.dt.weekday.fillna(-1).astype('int')
    weekday = weekday_series.values.reshape(-1, 1)
    month_series = time1.dt.month.fillna(0).astype('int')
    month = month_series.values.reshape(-1, 1)
    year_month_series = time1.dt.year.fillna(0).astype('int') * 100 + month_series
    year_month = (year_month_series.astype('float') / 1e5).values.reshape(-1, 1)
    weekend = (weekday_series >= 5).astype('int').values.reshape(-1, 1)

    site_values = site_ids.values.astype('int')
    site_presence = (site_values > 0)
    sites_count = site_presence.sum(axis=1).reshape(-1, 1)

    unique_sites_list = []
    repeat_ratio_list = []
    top_site_share_list = []
    entropy_list = []

    for row in site_values:
        non_zero_sites = [site for site in row if site != 0]
        if not non_zero_sites:
            unique_sites_list.append(0)
            repeat_ratio_list.append(0.0)
            top_site_share_list.append(0.0)
            entropy_list.append(0.0)
            continue
        counts = Counter(non_zero_sites)
        unique_sites_list.append(len(counts))
        total_visits = sum(counts.values())
        repeat_ratio_list.append((total_visits - len(counts)) / total_visits if total_visits else 0.0)
        top_site_share_list.append(max(counts.values()) / total_visits)
        probs = np.array(list(counts.values()), dtype='float32') / total_visits
        entropy_list.append(float(-(probs * np.log(probs + 1e-12)).sum()))

    unique_sites = np.array(unique_sites_list, dtype='float32').reshape(-1, 1)
    repeat_ratio = np.array(repeat_ratio_list, dtype='float32').reshape(-1, 1)
    top_site_share = np.array(top_site_share_list, dtype='float32').reshape(-1, 1)
    site_entropy = np.array(entropy_list, dtype='float32').reshape(-1, 1)

    start_minute = time1.dt.minute.fillna(0).astype('int').values.reshape(-1, 1)
    seconds_since_midnight = (time1 - time1.dt.normalize()).dt.total_seconds() \
        .fillna(0).values.reshape(-1, 1)

    time_seconds = times.apply(lambda col: col.astype('int64') // 10 ** 9)
    time_seconds = time_seconds.where(times.notna(), np.nan)
    time_diffs = time_seconds.diff(axis=1).to_numpy()
    mean_delta = np.nan_to_num(np.nanmean(time_diffs, axis=1), nan=0.0).reshape(-1, 1)
    std_delta = np.nan_to_num(np.nanstd(time_diffs, axis=1), nan=0.0).reshape(-1, 1)

    features = np.hstack([
        morning,
        day,
        evening,
        night,
        sess_duration,
        weekday,
        month,
        year_month,
        weekend,
        sites_count,
        unique_sites,
        repeat_ratio,
        top_site_share,
        site_entropy,
        start_minute,
        seconds_since_midnight,
        mean_delta,
        std_delta
    ]).astype('float32')
    return features


def select_best_model(X_train, y_train, cv):
    candidate_models = [
         (
             'logit_liblinear',
             LogisticRegression(random_state=SEED, solver='liblinear', max_iter=2000),
             {
                 'C': LOGIT_C_GRID,
                 'penalty': ['l1', 'l2'],
                 'class_weight': [None, 'balanced'],
             },
         ),
         (
             'logit_saga',
             LogisticRegression(random_state=SEED, solver='saga', max_iter=3000),
             [
                 {'C': LOGIT_C_GRID, 'penalty': ['l2'], 'class_weight': [None, 'balanced']},
                 {'C': LOGIT_C_GRID, 'penalty': ['l1'], 'class_weight': [None, 'balanced']},
                 {'C': LOGIT_C_GRID, 'penalty': ['elasticnet'], 'class_weight': [None, 'balanced'],
                  'l1_ratio': [0.2, 0.5, 0.8]},
             ],
         ),
         (
             'sgd_logistic',
             SGDClassifier(loss='log_loss', random_state=SEED, max_iter=5000, tol=1e-3),
             [
                 {'alpha': SGD_ALPHA_GRID, 'penalty': ['l2']},
                 {'alpha': SGD_ALPHA_GRID, 'penalty': ['elasticnet'], 'l1_ratio': [0.15, 0.5]},
             ],
         ),
    ]

    if HAS_XGBOOST and XGB_PARAM_GRID:
        candidate_models.append(
            (
                'xgboost',
                XGBClassifier(
                    objective='binary:logistic',
                    eval_metric='auc',
                    random_state=SEED,
                    tree_method='hist',
                    n_jobs=N_JOBS,
                    use_label_encoder=False,
                    verbosity=0,
                ),
                XGB_PARAM_GRID,
            ),
        )
    elif not HAS_XGBOOST:
        print('XGBoost not installed. Skipping the XGBoost model.')

    best_search = None
    best_name = None
    best_score = -np.inf


    for name, estimator, param_grid in candidate_models:
        print(f'\nTraining candidate model: {name}')
        search = GridSearchCV(estimator=estimator,
                              param_grid=param_grid,
                              scoring='roc_auc',
                              n_jobs=N_JOBS,
                              cv=cv,
                              verbose=1,
                              refit=True)
        search.fit(X_train, y_train)
        print(f'{name} best CV={search.best_score_:.5f} using params {search.best_params_}')
        if search.best_score_ > best_score:
            best_search = search
            best_name = name
            best_score = search.best_score_

    print(f'\nSelected model: {best_name} with CV={best_score:.5f}')
    return best_search, best_name


with timer('Building sparse site features'):
    X_train_sites, X_test_sites, y_train, vectorizer, train_times, test_times, train_sites, test_sites = \
        prepare_sparse_features(
            path_to_train=os.path.join(PATH_TO_DATA, 'train_sessions.csv'),
            path_to_test=os.path.join(PATH_TO_DATA, 'test_sessions.csv'),
            path_to_site_dict=os.path.join(PATH_TO_DATA, 'site_dic.pkl'),
            vectorizer_params={'ngram_range': SITE_NGRAMS,
                               'max_features': MAX_FEATURES,
                               'min_df': 2,
                               'sublinear_tf': True,
                               'tokenizer': lambda s: s.split()})


with timer('Building additional features'):
    train_dense = add_features(train_times, train_sites)
    test_dense = add_features(test_times, test_sites)
    scaler = StandardScaler()
    train_dense_scaled = scaler.fit_transform(train_dense)
    test_dense_scaled = scaler.transform(test_dense)
    train_dense_scaled = train_dense_scaled.astype('float32')
    test_dense_scaled = test_dense_scaled.astype('float32')
    X_train_final = hstack([X_train_sites, csr_matrix(train_dense_scaled)], format='csr')
    X_test_final = hstack([X_test_sites, csr_matrix(test_dense_scaled)], format='csr')


with timer('Cross-validation'):
    time_split = TimeSeriesSplit(n_splits=NUM_TIME_SPLITS)
    best_search, best_model_name = select_best_model(X_train_final, y_train, time_split)
    print(f'CV score ({best_model_name}) {best_search.best_score_:.5f}')


with timer('Test prediction and submission'):
    best_estimator = best_search.best_estimator_
    test_pred = best_estimator.predict_proba(X_test_final)[:, 1]
    pred_df = pd.DataFrame(test_pred, index=np.arange(1, test_pred.shape[0] + 1),
                       columns=['target'])
    pred_df.to_csv(f'submission_alice_{AUTHOR}.csv', index_label='session_id')
