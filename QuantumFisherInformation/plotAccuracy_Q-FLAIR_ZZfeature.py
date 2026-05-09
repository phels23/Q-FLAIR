import pylab as plt
import numpy as np

import matplotlib as mpl
from matplotlib.legend_handler import HandlerTuple

mpl.rcParams.update({'text.usetex':True,'font.size': 12, 'font.family': 'serif', 'font.serif': ['Times', 'Computer Modern']})

fig,ax1=plt.subplots(figsize=(11.69*0.4,8.27*0.38))

Dimension = [7*7,14*14,28*28]

QFLAIR= [89.5,89.9,87.0]
ZZFeature = [60,52,49.8]

QFLAIR_rank = [12,13,5]
ZZFeature_rank = [49,196,784]

p1, = ax1.plot(Dimension,QFLAIR,'-d',color='black',fillstyle='none',label='Q-FLAIR')
p2, = ax1.plot(Dimension,ZZFeature,'--s',color='black',fillstyle='none',label='ZZFeatureMap')
ax1.set_xlabel('dimension')
ax1.set_ylabel(r'accuracy [$\%$]')
ax1.set_ylim(45,95)
ax1.set_xlim(0,800)

ax2 = ax1.twinx()
p3, = ax2.plot(Dimension,QFLAIR_rank,'-^',color='gray',fillstyle='none')
p4, = ax2.plot(Dimension,ZZFeature_rank,'--<',color='gray',fillstyle='none')
ax2.set_ylabel('rank',color='gray')
ax2.tick_params(axis='y',labelcolor='gray')

plt.legend([(p1,p3),(p2,p4)],['Q-FLAIR','ZZFeatureMap'],
               handler_map={tuple: HandlerTuple(ndivide=None)},
               loc=2,
               bbox_to_anchor=(0, 0.8))


plt.tight_layout()
plt.savefig('Accuracy_Q-FLAIR_ZZfeature.pdf')
