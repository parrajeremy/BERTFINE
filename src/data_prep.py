import pandas as pd
from math import ceil


if __name__== "__main__":
    df = pd.read_csv("../data/SWA-Fleet_Health.csv", encoding = "ISO-8859-1", dtype={'FACT_ATA':'str'})
    df['FACT'] = df.apply(lambda row: str(row['FACT']).replace('r\t', ''), axis=1)
    df['DISCREPANCY'] = df.apply(lambda row: str(row['DISCREPANCY']).replace('r\t', ''), axis=1)
    df['FACT'] = df.apply(lambda row: str(row['FACT']).replace("\n", ''), axis=1)
    df['DISCREPANCY'] = df.apply(lambda row: str(row['DISCREPANCY']).replace("\n", ''), axis=1)
    df['FACT'] = df.apply(lambda row: str(row['FACT']).replace("\r", ''), axis=1)
    df['DISCREPANCY'] = df.apply(lambda row: str(row['DISCREPANCY']).replace("\r", ''), axis=1)

    OCList = list(set(df.object_code.values))

    df['OCI'] = df.apply(lambda row: OCList.index(row['object_code']), axis=1)
    df['alpha'] = 'a'
    df['id'] = range(df.shape[0])

    df_bert = df[['id','OCI','alpha','DISCREPANCY']]
    df_bert.dropna(inplace=True)

    train_df_bert = df_bert.groupby('OCI', group_keys=False).apply(lambda df_tmp: df_tmp.sample(n=ceil(0.2 * df_tmp.shape[0]) if 0.2 * df_tmp.shape[0] > 1 else 1, random_state=1)) #n=ceil(0.2 * df_tmp.shape[0])
    dev_df_bert = df_bert[~df_bert['id'].isin(train_df_bert.id.values)]

    dev_df_bert['id'] = range(dev_df_bert.shape[0])
    train_df_bert['id'] = range(train_df_bert.shape[0])
    train_df_bert.to_csv('../data/train.tsv', sep='\t', index=False, header=False)
    dev_df_bert.to_csv('../data/dev.tsv', sep='\t', index=False, header=False)