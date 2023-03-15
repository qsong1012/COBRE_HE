import pandas as pd

df = pd.read_pickle('meta/ft_all.pickle')
df = df.loc[df.cancer != 'TCGA_LUSC']
df_luad = pd.read_pickle('meta/tcga_luad_meta.pickle')
df = pd.concat([df, df_luad])
df = df.rename(columns={'split':'fold','submitter_id':'id_patient'})

df.to_pickle('meta/ft_all.pickle')


df = pd.read_pickle('meta/ft_svs_g10.pickle')
df = df.loc[df.cancer != 'TCGA_LUSC']
df_luad = pd.read_pickle('meta/tcga_luad_svs.pickle')
df_luad = df_luad.rename(columns={'file_original':'id_svs','submitter_id':'id_patient'})
df_luad['slide_type'] = 'ffpe'
df = pd.concat([df, df_luad])

df.to_pickle('meta/ft_svs_g10.pickle')



# brain

import pandas as pd
df_svs = pd.read_pickle('meta/tcga_brain_svs.pickle')
df_all = pd.read_pickle('meta/tcga_brain_meta.pickle')

df_svs.rename(columns={'submitter_id':'id_patient','file_original':'id_svs'}, inplace=True)
df_svs['slide_type'] = 'ffpe'

df_all.rename(columns={'submitter_id':'id_patient','split':'fold'}, inplace=True)


df_svs.to_pickle('meta/tcga_brain_svs.pickle')
df_all.to_pickle('meta/tcga_brain_meta.pickle')



# Lung

import pandas as pd
df_svs = pd.read_pickle('meta/hipt/Lung/svs_meta.pickle')

df_svs.rename(columns={'submitter_id':'id_patient','file_original':'id_svs'}, inplace=True)
df_svs['slide_type'] = 'ffpe'
df_svs.to_pickle('meta/hipt/Lung/svs_meta.pickle')

for split in range(10):
	df_all = pd.read_pickle(f'meta/hipt/Lung/split_{split}-meta-0.25.pickle')
	df_all.rename(columns={'submitter_id':'id_patient','split':'fold'}, inplace=True)
	df_all.to_pickle(f'meta/hipt/Lung/split_{split}-meta-0.25.pickle')





# BRCA

import pandas as pd
df_svs = pd.read_pickle('meta/hipt/TCGA_BRCA/svs_meta.pickle')

df_svs.rename(columns={'submitter_id':'id_patient','file_original':'id_svs'}, inplace=True)
df_svs['slide_type'] = 'ffpe'
df_svs.to_pickle('meta/hipt/TCGA_BRCA/svs_meta.pickle')

for split in range(10):
	df_all = pd.read_pickle(f'meta/hipt/TCGA_BRCA/split_{split}-meta-0.25.pickle')
	df_all.rename(columns={'submitter_id':'id_patient','split':'fold'}, inplace=True)
	df_all.to_pickle(f'meta/hipt/TCGA_BRCA/split_{split}-meta-0.25.pickle')



for split in range(10):
	df_all = pd.read_pickle(f'meta/hipt/TCGA_BRCA/split_{split}-meta.pickle')
	df_all.rename(columns={'submitter_id':'id_patient','split':'fold'}, inplace=True)
	df_all.to_pickle(f'meta/hipt/TCGA_BRCA/split_{split}-meta.pickle')


# Lung

import pandas as pd
df_svs = pd.read_pickle('meta/hipt/Kidney/svs_meta.pickle')

df_svs.rename(columns={'submitter_id':'id_patient','file_original':'id_svs'}, inplace=True)
df_svs['slide_type'] = 'ffpe'
df_svs.to_pickle('meta/hipt/Kidney/svs_meta.pickle')

for split in range(10):
	df_all = pd.read_pickle(f'meta/hipt/Kidney/split_{split}-meta-0.25.pickle')
	df_all.rename(columns={'submitter_id':'id_patient','split':'fold'}, inplace=True)
	df_all.to_pickle(f'meta/hipt/Kidney/split_{split}-meta-0.25.pickle')



for split in range(10):
	df_all = pd.read_pickle(f'meta/hipt/Kidney/split_{split}-meta.pickle')
	df_all.rename(columns={'submitter_id':'id_patient','split':'fold'}, inplace=True)
	df_all.to_pickle(f'meta/hipt/Kidney/split_{split}-meta.pickle')
