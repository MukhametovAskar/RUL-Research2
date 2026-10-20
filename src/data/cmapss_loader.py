import pandas as pd
from sklearn.cluster import KMeans

COLS = ['unit', 'time', 'op1', 'op2', 'op3'] + [f's{i}' for i in range(1, 22)]
DROP_SENSORS = ['s1', 's5', 's6', 's10', 's16', 's18', 's19']


def apply_regime_specific_normalization(df_train, df_test, n_clusters=6):
    op_cols = ['op1', 'op2', 'op3']
    sensor_cols = [c for c in df_train.columns if c.startswith('s')]

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(df_train[op_cols])

    df_train = df_train.copy()
    df_test = df_test.copy()
    df_train['regime_idx'] = kmeans.predict(df_train[op_cols])
    df_test['regime_idx'] = kmeans.predict(df_test[op_cols])

    for cluster_id in range(n_clusters):
        tr_mask = df_train['regime_idx'] == cluster_id
        te_mask = df_test['regime_idx'] == cluster_id
        if tr_mask.sum() == 0:
            continue

        mins = df_train.loc[tr_mask, sensor_cols].min()
        maxs = df_train.loc[tr_mask, sensor_cols].max()
        diffs = maxs - mins
        diffs[diffs == 0] = 1.0

        df_train.loc[tr_mask, sensor_cols] = (df_train.loc[tr_mask, sensor_cols] - mins) / diffs
        if te_mask.sum() > 0:
            df_test.loc[te_mask, sensor_cols] = (df_test.loc[te_mask, sensor_cols] - mins) / diffs

    return df_train, df_test


def prepare_cmapss(train_path, test_path, rul_path, max_rul, n_clusters):
    train_df = pd.read_csv(train_path, sep=r'\s+', header=None, names=COLS)
    test_df = pd.read_csv(test_path, sep=r'\s+', header=None, names=COLS)
    true_rul = pd.read_csv(rul_path, sep=r'\s+', header=None, names=['RUL'])

    train_df.drop(columns=DROP_SENSORS, inplace=True, errors='ignore')
    test_df.drop(columns=DROP_SENSORS, inplace=True, errors='ignore')

    train_df, test_df = apply_regime_specific_normalization(train_df, test_df, n_clusters=n_clusters)

    cols_to_drop = ['op1', 'op2', 'op3', 'regime_idx']
    train_df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
    test_df.drop(columns=cols_to_drop, inplace=True, errors='ignore')

    features = [c for c in train_df.columns if c not in ('unit', 'time')]

    rul_train = pd.DataFrame(train_df.groupby('unit')['time'].max()).reset_index()
    rul_train.columns = ['unit', 'max_time']
    train_df = train_df.merge(rul_train, on='unit', how='left')
    train_df['RUL'] = train_df['max_time'] - train_df['time']
    train_df.drop(columns=['max_time'], inplace=True)
    train_df['RUL'] = train_df['RUL'].clip(upper=max_rul)

    rul_test = pd.DataFrame(test_df.groupby('unit')['time'].max()).reset_index()
    rul_test.columns = ['unit', 'max_time']
    true_rul['unit'] = true_rul.index + 1
    true_rul['max_rul'] = true_rul['RUL']
    rul_test = rul_test.merge(true_rul[['unit', 'max_rul']], on='unit', how='left')
    rul_test['total_life'] = rul_test['max_time'] + rul_test['max_rul']

    test_df = test_df.merge(rul_test[['unit', 'total_life']], on='unit', how='left')
    test_df['RUL'] = test_df['total_life'] - test_df['time']
    test_df.drop(columns=['total_life'], inplace=True)
    test_df['RUL'] = test_df['RUL'].clip(upper=max_rul)

    return train_df, test_df, features
