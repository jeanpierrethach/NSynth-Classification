import os
import argparse
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from utils import maybe_make_directory

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--graph_dir', type=str,
						default='./graphs',
						help='Directory path to the graphs output. (default: %(default)s)')

	args = parser.parse_args()
	maybe_make_directory(args.graph_dir)
	return args

args = parse_args()

def instrument_code(n):
	class_names=['bass', 'brass', 'flute', 'guitar', 
			 'keyboard', 'mallet', 'organ', 'reed', 
			 'string', 'synth_lead', 'vocal']
	return class_names[n]

distrib = {'9':5501,'2':8773,'10':10208,'1':12675,'7':13911,'8':19474,'3':32690,
		'5':34201,'6':34477,'4':51821,'0':65474}

df = pd.DataFrame.from_dict(distrib, orient='index').reset_index()
df = df.rename(columns={'index':'instrument', 0:'count'})
df = df.sort_values(by='count', ascending=False)
df = pd.concat([df['instrument'].astype(int).apply(instrument_code), df['count']], axis=1)

f, ax = plt.subplots(figsize=(10, 10))
ax = sns.barplot(
	y='instrument',
	x='count',
	data=df)
_ = ax.set(xlabel='Number of samples', ylabel='Instrument class', title='Instrument class distribution')

ax.figure.savefig(os.path.join(args.graph_dir, 'nsynth_class_dist.png'))