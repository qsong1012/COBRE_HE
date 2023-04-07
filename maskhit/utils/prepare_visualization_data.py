import pandas as pd

study = 'TCGA_BRCA'
df_meta = pd.read_pickle('meta/tcga_brca_meta.pickle')
df_svs = pd.read_pickle('meta/tcga_brca_svs.pickle')


sel_ids = [
    'TCGA-BH-A1FN-01Z-00-DX1.CEE3C59B-6CF0-4D41-8334-6067BB5A8BF7',
]

df_svs = df_svs.loc[df_svs.id_svs.isin(sel_ids)]
df_meta = df_meta.loc[df_meta.id_patient.isin(df_svs.id_patient)]

res = []
for i, row in df_svs.iterrows():
    print(row['id_svs'])
    df_i = pd.read_pickle(
        f"data/{study}/{row['id_svs']}/mag_10-size_224/meta.pickle")
    df_i = df_i.loc[df_i.counts_20 > 0, ['pos']]
    df_i['id_patient'] = row['id_patient']
    df_i['id_svs'] = row['id_svs']
    df_i['svs_path'] = row['svs_path']
    res.append(df_i)

df_locs = pd.concat(res)
df_locs['slide_type'] = 'ffpe'
df_locs['cancer'] = study

df_meta.merge(df_locs[['id_patient','svs_path','id_svs']].drop_duplicates('id_svs'), on='id_patient')

df_locs.to_pickle(f'meta/vis_{study.lower()}_locs-split.pickle')
df_meta.to_pickle(f'meta/vis_{study.lower()}_meta-split.pickle')
