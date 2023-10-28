#!/usr/bin/python

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

from pymoo.indicators.hv import HV
from sklearn.cluster import DBSCAN
import random, math

labelsize_b = 14
linewidth = 2
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams["font.family"] = "Times New Roman"
colors = ['#DB1F48','#1C4670','#FF9636','#9D5FFB','#21B6A8','#D65780']
# colors = ['#ED4974','#16B9E1','#58DE7B','#F0D864','#FF8057','#8958D3']
# colors =['#FD0707','#0D0DDF','#129114','#DDDB03','#FF8A12','#8402AD']
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
colors = ["#1f78b4", "#33a02c", "#e31a1c", "#6a3d9a", "#fdbf6f", "#ff7f00"]
# colors = ["#006d2c", "#31a354", "#74c476", "#bae4b3", "#ececec", "#969696"]
colors = ["#004c6d", "#f18f01", "#81b214", "#c7243a", "#6b52a1", "#a44a3f"]

colors4 = ['#c989e6', '#8e4d9e', '#692b7a', '#8e8e9d']
colors4 = ['#B57EDC', '#9567BD', '#6F3E5D', '#7D53DE', '#5D527F']
colors4 = [
    "#8B008B",  # Purple
    "#A52A2A",  # Brown
    "#D2691E",  # Chocolate
    "#FF4500",  # Orange Red
    "#FFA500"   # Orange
]

views_of_category = [4,6,5,4,4]
markers = ['s','o','^','v','D','<','>','P','*'] 
hatches = ['/' ,'\\','--','x', '+', 'O','-',]
linestyles = ['solid','dashed','dotted','dashdot',(0, (3, 5, 1, 5)),(0, (3, 1, 1, 1))]
from collections import OrderedDict
linestyle_dict = OrderedDict(
    [('solid',               (0, ())),
     ('dashed',              (0, (5, 5))),
     ('dotted',              (0, (1, 5))),
     ('dashdotted',          (0, (3, 5, 1, 5))),
     ('densely dotted',      (0, (1, 1))),

     ('loosely dashed',      (0, (5, 10))),
     ('densely dashed',      (0, (5, 1))),

     ('densely dashdotted',  (0, (3, 1, 1, 1))),

     ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
     ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))])
linestyles = []
for i, (name, linestyle) in enumerate(linestyle_dict.items()):
    if i >= 9:break
    linestyles += [linestyle]

from matplotlib.patches import Ellipse

def line_plot(XX,YY,label,color,path,xlabel,ylabel,lbsize=labelsize_b,legloc='best',linestyles=linestyles,
				xticks=None,yticks=None,ncol=None, yerr=None, xticklabel=None,yticklabel=None,xlim=None,ylim=None,ratio=None,
				use_arrow=False,arrow_coord=(60,0.6),markersize=16,bbox_to_anchor=None,get_ax=0,linewidth=2,logx=False,
				use_throughput_annot=False,lgsize=None,oval=False,markevery=1,
				markersize_list=[],markers=markers,markerfacecolor='none',display_annot=[],si_annot=False,sr_annot=False,
				saving_annot=None,mps_annot=False,ablation_annot=False,sisr_annot=False,hv_annot=False,hv_annot2=False,
				bw_annot=False,mlp_annot=False):
	if lgsize is None:
		lgsize = lbsize
	if get_ax==1:
		ax = plt.subplot(211)
	elif get_ax==2:
		ax = plt.subplot(212)
	else:
		fig, ax = plt.subplots()
	ax.grid(zorder=0)
	handles = []
	for i in range(len(XX)):
		xx,yy = XX[i],YY[i]
		if logx:
			xx = np.log10(np.array(xx))
		if yerr is None:
			if not markersize_list:
				plt.plot(xx, yy, color = color[i], marker = markers[i], 
					label = label[i], linestyle = linestyles[i], 
					linewidth=linewidth, markersize=markersize, markerfacecolor='none', markevery=markevery)
			else:
				plt.plot(xx, yy, color = color[i], marker = markers[i], 
					label = label[i], linestyle = linestyles[i], 
					linewidth=linewidth, markersize=markersize_list[i], markevery=markevery)
		else:
			if markersize > 0:
				plt.errorbar(xx, yy, yerr=yerr[i], color = color[i],
					marker = markers[i], label = label[i], 
					linestyle = linestyles[i], 
					linewidth=linewidth, markersize=markersize, markerfacecolor='none', markevery=markevery,
					capsize=4)
			else:
				plt.errorbar(xx, yy, yerr=yerr[i], color = color[i],
					label = label[i], 
					linewidth=linewidth,
					capsize=4)
	plt.xlabel(xlabel, fontsize = lbsize)
	plt.ylabel(ylabel, fontsize = lbsize)
	plt.xticks(fontsize=lbsize)
	plt.yticks(fontsize=lbsize)
	if xlim is not None:
		ax.set_xlim(xlim)
	if ylim is not None:
		ax.set_ylim(ylim)
	if xticks is not None:
		plt.xticks(xticks,fontsize=lbsize)
	if yticks is not None:
		plt.yticks(yticks,fontsize=lbsize)
	if xticklabel is not None:
		ax.set_xticklabels(xticklabel)
	if yticklabel is not None:
		ax.set_yticklabels(yticklabel)
	if use_arrow:
		ax.text(
		    arrow_coord[0], arrow_coord[1], "Better", ha="center", va="center", rotation=45, size=lbsize,
		    bbox=dict(boxstyle="larrow,pad=0.3", fc="white", ec="black", lw=2))
	if ablation_annot:
		ax.text(0.37,29.5,"1.8dB-4dB\nreduction", ha="center", va="center", size=lbsize,color=color[3],fontweight='bold')
		ax.text(1,32,"3.75X bitrates", ha="center", va="center", size=lbsize,fontweight='bold')
		ax.annotate(text="",xy=(0.4,31.66),xytext=(1.4,31.66), arrowprops=dict(arrowstyle='<-',lw=2),size=lbsize)
	if hv_annot:
		ax.annotate(text="Hypervolume plateaus\nafter certain generations.",xy=(XX[0][6],YY[0][6]),xytext=(5,1750), arrowprops=dict(arrowstyle='->',lw=2),size=lbsize)
	if hv_annot2:
		ax.annotate(text="Hypervolume is near-optimal\nbeyond certain sampled views.",xy=(XX[0][5],YY[0][5]),xytext=(4.5,2570), arrowprops=dict(arrowstyle='->',lw=2),size=lbsize)
	if bw_annot:
		ax.annotate(text="Finetuning benefits\ndiminish.",xy=(XX[0][4],YY[0][4]),xytext=(4,21), arrowprops=dict(arrowstyle='->',lw=2),size=lbsize)
	if mlp_annot:
		ax.annotate(text="Multi-channel training\nbenefits diminish.",xy=(XX[0][1],YY[0][1]),xytext=(40,26.2), arrowprops=dict(arrowstyle='->',lw=2),size=lbsize)
	if sisr_annot:
		ax.text(-0.87,27.7,"Gain"+r'$\leq$'+"1dB\nif ratio"+r'$\leq 0.01$', ha="center", va="center", size=lbsize,fontweight='bold')
	if mps_annot:
		ax.text(22*16,29.5,"Full", ha="center", va="center", size=lbsize,fontweight='bold')
		ax.text(12*16,29.5,"Half", ha="center", va="center", size=lbsize,fontweight='bold')
		ax.text(5*16-2,33.3,"0.47-0.75dB\nloss at half", ha="center", va="center", size=lbsize,fontweight='bold')
		ax.annotate(text="",xy=(160,28),xytext=(160,34), arrowprops=dict(arrowstyle='-',lw=2),size=lbsize)
		ax.annotate(text="",xy=(320,28),xytext=(320,34), arrowprops=dict(arrowstyle='-',lw=2),size=lbsize)
	if saving_annot is not None:
		c = color[4] if saving_annot[1]>1 else color[2]
		h = 0.7 if saving_annot[1]>1 else -0.9e-6
		w = 0.11 if saving_annot[1]>1 else -0.2
		ax.text((saving_annot[2]+saving_annot[3])/2+w,saving_annot[1]+h,u'\u2193'+f"{saving_annot[0]}%", ha="center", va="center", size=lbsize+4,fontweight='bold')
		ax.annotate(text="",xy=(saving_annot[2],saving_annot[1]),xytext=(saving_annot[3],saving_annot[1]), arrowprops=dict(arrowstyle='->',lw=2,color=c),size=lbsize,fontweight='bold')
	if display_annot:
		for i in range(len(XX)):
			xx,yy = XX[i],YY[i]
			ax.annotate(label[i], xy=(xx[0],yy[0]), xytext=(xx[0]+display_annot[i][0],yy[0]+display_annot[i][1]), fontsize=lbsize-4,)
	if si_annot:
		for i in range(len(XX)):
			xx,yy = XX[i],YY[i]
			ax.text(xx[3],yy[3]+0.2, u'\u2713', ha="center", va="center", size=lbsize,fontweight='bold')
		# ax.text(-1.5,33.3, u'\u2713'+"AcID's Config", ha="center", va="center", size=lbsize,fontweight='bold')
		ax.annotate("DOLE", xy=(XX[3][-2],YY[3][-2]), xytext=(-1.5,33.3), arrowprops=dict(arrowstyle='->',lw=2), fontsize=lbsize,fontweight='bold')
	if sr_annot:
		for i in range(len(XX)):
			xx,yy = XX[i],YY[i]
			ax.text(xx[-1],yy[-1]-0.3, u'\u2713', ha="center", va="center", size=lbsize,fontweight='bold')
		# ax.text(-0.6,27.4, u'\u2713'+" AcID's Config", ha="center", va="center", size=lbsize,fontweight='bold')
		ax.annotate("DOLE", xy=(XX[0][-1],YY[0][-1]-0.5), xytext=(-0.5,27.2), arrowprops=dict(arrowstyle='->',lw=2), fontsize=lbsize,fontweight='bold')
	if use_throughput_annot:
		ax.annotate(text=f"$\u2191$"+'41%', xy=(XX[1][1],YY[1][1]), xytext=(0,0.8), arrowprops=dict(arrowstyle='->',lw=2),size=lgsize+2,fontweight='bold')
	if ncol!=0:
		if ncol is None:
			plt.legend(loc=legloc,fontsize = lgsize)
		else:
			if bbox_to_anchor is None:
				plt.legend(loc=legloc,fontsize = lgsize,ncol=ncol)
			else:
				if oval:
					plt.legend(loc=legloc,fontsize = lgsize,ncol=ncol,bbox_to_anchor=bbox_to_anchor, handles=handles)
				else:
					plt.legend(loc=legloc,fontsize = lgsize,ncol=ncol,bbox_to_anchor=bbox_to_anchor)
	if ratio is not None:
		xleft, xright = ax.get_xlim()
		ybottom, ytop = ax.get_ylim()
		ax.set_aspect(abs((xright-xleft)/(ybottom-ytop))*ratio)
	plt.tight_layout()
	if get_ax!=0:
		return ax
	fig.savefig(path,bbox_inches='tight')
	plt.close()


def groupedbar(data_mean,data_std,ylabel,path,yticks=None,envs = [2,3,4],
				methods=['Ours','Standalone','Optimal','Ours*','Standalone*','Optimal*'],use_barlabel_x=False,use_barlabe_y=False,
				ncol=3,bbox_to_anchor=(0.46, 1.28),sep=1.25,width=0.15,xlabel=None,legloc=None,labelsize=labelsize_b,ylim=None,
				use_downarrow=False,rotation=None,lgsize=None,yticklabel=None,latency_annot=False,bandwidth_annot=False,latency_met_annot=False,
				showaccbelow=False,showcompbelow=False,showrepaccbelow=False,breakdown_annot=False,frameon=True,c2s_annot=False,colors=colors):
	if lgsize is None:
		lgsize = labelsize
	fig = plt.figure()
	ax = fig.add_subplot(111)
	num_methods = data_mean.shape[1]
	num_env = data_mean.shape[0]
	center_index = np.arange(1, num_env + 1)*sep
	# colors = ['lightcoral', 'orange', 'yellow', 'palegreen', 'lightskyblue']
	# colors = ['coral', 'orange', 'green', 'cyan', 'blue']
	# colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]


	ax.grid()
	ax.spines['bottom'].set_linewidth(3)
	ax.spines['top'].set_linewidth(3)
	ax.spines['left'].set_linewidth(3)
	ax.spines['right'].set_linewidth(3)
	if rotation is None:
		plt.xticks(center_index, envs, size=labelsize)
	else:
		plt.xticks(center_index, envs, size=labelsize, rotation=rotation)
	plt.xticks(fontsize=labelsize)
	plt.yticks(fontsize=labelsize)
	ax.set_ylabel(ylabel, size=labelsize)
	if xlabel is not None:
		ax.set_xlabel(xlabel, size=labelsize)
	if yticks is not None:
		plt.yticks(yticks,fontsize=labelsize)
	if yticklabel is not None:
		ax.set_yticklabels(yticklabel)
	if ylim is not None:
		ax.set_ylim(ylim)
	for i in range(num_methods):
		x_index = center_index + (i - (num_methods - 1) / 2) * width
		hbar=plt.bar(x_index, data_mean[:, i], width=width, linewidth=2,
		        color=colors[i], label=methods[i], hatch=hatches[i], edgecolor='k')
		if data_std is not None:
		    plt.errorbar(x=x_index, y=data_mean[:, i],
		                 yerr=data_std[:, i], fmt='k.', elinewidth=3,capsize=4)
		if use_barlabel_x:
			if i in [2,3]:
				for k,xdx in enumerate(x_index):
					ax.text(xdx-0.07,data_mean[k,i]+3,f'{data_mean[k,i]:.4f}',fontsize = labelsize, rotation='vertical',fontweight='bold')
		if use_barlabe_y and i==2:
			for k,xdx in enumerate(x_index):
				ax.text(xdx-0.08,data_mean[k,i]+1,f'{data_mean[k,i]:.4f}',fontsize = labelsize, rotation='vertical',fontweight='bold')
		if use_downarrow:
			if i==1:
				for j in range(2,data_mean.shape[0]):
					ax.annotate(text='', xy=(x_index[j],data_mean[j,i]), xytext=(x_index[j],200), arrowprops=dict(arrowstyle='<->',lw=4))
					ax.text(x_index[j]-0.04, 160, '$\downarrow$'+f'{200-data_mean[j,i]:.0f}%', ha="center", va="center", rotation='vertical', size=labelsize ,fontweight='bold')
					# ax.text(center_index[j]-0.02,data_mean[j,i]+5,'$\downarrow$'+f'{200-data_mean[j,i]:.0f}%',fontsize = 16, fontweight='bold')
			else:
				for k,xdx in enumerate(x_index):
					ax.text(xdx-0.07,data_mean[k,i]+5,f'{data_mean[k,i]:.2f}',fontsize = labelsize,fontweight='bold')

		if latency_annot:
			if i==1:
				for k,xdx in enumerate(x_index):
					mult = data_mean[k,i]/data_mean[k,0]
					ax.text(xdx-0.3,data_mean[k,i]+2,f'{mult:.1f}\u00D7',fontsize = labelsize)
		if bandwidth_annot:
			if i==1:
				for k,xdx in enumerate(x_index):
					mult = int(10**data_mean[k,i]/10**data_mean[k,0])
					ax.text(xdx-0.4,data_mean[k,i]+0.1,f'{mult}\u00D7',fontsize = labelsize)
		if latency_met_annot:
			if i>=1:
				for k,xdx in enumerate(x_index):
					mult = (-data_mean[k,i] + data_mean[k,0])/data_mean[k,0]*100
					if i==2:
						ax.text(xdx-0.07,data_mean[k,i],'$\downarrow$'+f'{mult:.1f}%',fontsize = lgsize,rotation='vertical',fontweight='bold')
					else:
						ax.text(xdx-0.07,data_mean[k,i],'$\downarrow$'+f'{mult:.1f}%',fontsize = lgsize,rotation='vertical')
		if breakdown_annot:
			if i == 2:
				ax.text(x_index[0]-0.07,data_mean[0,2]+0.015,'$\downarrow$'+f'{int(1000*(data_mean[0,0]-data_mean[0,2]))}ms',fontsize = lgsize,rotation='vertical',fontweight='bold')
				ax.text(x_index[2]-0.07,data_mean[2,2]+0.005,'$\u2191$'+f'{int(1000*(data_mean[2,2]))}ms',fontsize = lgsize,rotation='vertical')
		if showaccbelow:
			if i<=1:
				ax.text(2.3,-2.3, "Better", ha="center", va="center", rotation=90, size=labelsize,
				    bbox=dict(boxstyle="rarrow,pad=0.3", fc="white", ec="black", lw=2))
				for k,xdx in enumerate(x_index):
					mult = -data_mean[k,i]
					if i!=1:
						ax.text(xdx-0.06,data_mean[k,i]-1.7,'$\downarrow$'+f'{mult:.1f}%',fontsize = labelsize,rotation='vertical')
					else:
						ax.text(xdx-0.06,data_mean[k,i]-1.7,'$\downarrow$'+f'{mult:.1f}%',fontsize = labelsize,rotation='vertical',fontweight='bold')
		if showrepaccbelow:
			if i<=1:
				ax.text(1.3,-7, "Better", ha="center", va="center", rotation=90, size=labelsize,
				    bbox=dict(boxstyle="rarrow,pad=0.3", fc="white", ec="black", lw=2))
				for k,xdx in enumerate(x_index):
					mult = -data_mean[k,i]
					ax.text(xdx-0.06,data_mean[k,i]-3.2,'$\downarrow$'+f'{mult:.1f}%',fontsize = lgsize,rotation='vertical')
		if showcompbelow:
			if i<=1:
				for k,xdx in enumerate(x_index):
					mult = -data_mean[k,i]
					ax.text(xdx-0.06,data_mean[k,i]-8,'$\downarrow$'+f'{mult:.1f}%',fontsize = labelsize-2,rotation='vertical')
	
	if c2s_annot:
		ax.text(2.8,33.2,"<0.5dB loss\n3090 to 1080", ha="center", va="center", size=labelsize,fontweight='bold')
	if ncol>0:
		if legloc is None:
			plt.legend(bbox_to_anchor=bbox_to_anchor, fancybox=True,
			           loc='upper center', ncol=ncol, fontsize=lgsize, frameon=frameon)
		else:
			plt.legend(fancybox=True,
			           loc=legloc, ncol=ncol, fontsize=lgsize, frameon=frameon)
	fig.savefig(path, bbox_inches='tight')
	plt.close()


# Read the array from the file
import re

def extract_numbers(line):
    return re.findall(r'\d+\.\d+|\d+', line)


def data_from_profiling(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
        for i,line in enumerate(lines):
            if i == 0:
                configs = np.array(eval(line))
            elif i == 1:
                metrics = np.array(eval(line)); metrics[...,0] *= -1
            elif i == 2:
                explored_configs = np.array(eval(line))
            elif i == 3:
                explored_metrics = np.array(eval(line)); explored_metrics[...,0] *= -1
            else:
                break
        
    return configs,metrics,explored_configs,explored_metrics
	
def plot_profile():
	configs,metrics,explored_configs,explored_metrics = data_from_profiling('meta/profiling.50.log')

	lbsize = 24
	base_psnr = [35.91001747493999]
	base_size = [75.92373657226562]

	# final profile
	fig, ax = plt.subplots()
	plt.scatter(metrics[:,1], metrics[:,0], 
			color=colors4[0], label=f'Pareto Optimal',facecolors='none', marker='*', s=200)

	# Add labels and legend
	plt.xticks(fontsize=lbsize)
	plt.yticks(fontsize=lbsize)
	plt.xlabel('Size (MB)', fontsize = lbsize)
	plt.ylabel('PSNR (dB)', fontsize = lbsize)
	plt.legend(ncol=1, fontsize = lbsize)
	plt.tight_layout()
	fig.savefig(f'images/profile_final.png',bbox_inches='tight')
	plt.close()

	for i in [0,4,9,19]:
		list_N = configs
		list_M = explored_configs[i]
		# Use broadcasting and np.all to check if each row in list_M exists in list_N
		rows_exist = np.all(list_N[:, None, :] == list_M[None, :, :], axis=-1)

		# Get the indices of the matching rows
		matching_indices = np.any(rows_exist, axis=0)

		fig, ax = plt.subplots()
		plt.scatter(explored_metrics[i, matching_indices, 1], explored_metrics[i, matching_indices, 0], 
			  color=colors4[0], label=f'Pareto Optimal',facecolors='none', marker='*', s=200)
		plt.scatter(explored_metrics[i, ~matching_indices, 1], explored_metrics[i, ~matching_indices, 0], 
			  color=colors4[3], label=f'Sub-optimal',facecolors='none',marker='o',s=200)
		plt.scatter(base_size, base_psnr, color='k', label=f'Baseline',marker='^',s=200)

		for y,x in explored_metrics[i, matching_indices]:
			# # Fill the rectangle at the bottom right with light grey
			plt.fill_between([x, base_size[0]], [y,y], y2=25, color='grey', alpha=0.1)

		# Add labels and legend
		plt.xticks(fontsize=lbsize)
		plt.yticks(fontsize=lbsize)
		plt.xlabel('Size (MB)', fontsize = lbsize)
		plt.ylabel('PSNR (dB)', fontsize = lbsize)
		plt.legend(ncol=1, fontsize = lbsize)
		plt.tight_layout()
		fig.savefig(f'images/profile{i}.pdf',bbox_inches='tight')
		plt.close()

def find_number_of_clusters(data, eps, min_samples):
    # Initialize DBSCAN with specified parameters
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)

    # Fit the model to the data
    labels = dbscan.fit_predict(data)

    # Count the number of unique clusters (excluding noise points with label -1)
    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    return num_clusters

def plot_pf_metrics():
	configs,metrics,explored_configs,explored_metrics = data_from_profiling('meta/profiling.50.log')

	base_psnr = [35.91001747493999]
	base_size = [75.92373657226562]


	ref_point = np.array([0,base_size[0]])

	ind = HV(ref_point=ref_point)

	hv_list = [[]]
	psnr_maxmin_list = [[],[]]
	size_maxmin_list = [[],[]]
	ncluster_list = [[]]
	for i in range(explored_metrics.shape[0]):
		# find the pareto front of each iteration
		pf = configs
		accumulated_configs = explored_configs[:i+1].reshape(-1,explored_configs.shape[-1])
		accumulated_metrics = explored_metrics[:i+1].reshape(-1,explored_metrics.shape[-1])

		# Use broadcasting and np.all to check if each row in list_M exists in list_N
		rows_exist = np.all(pf[:, None, :] == accumulated_configs[None, :, :], axis=-1)

		# Get the indices of the matching rows
		matching_indices = np.any(rows_exist, axis=0)

		# how it extends the boundary
		psnr_maxmin_list[0] += [accumulated_metrics[matching_indices, 0].max()]
		psnr_maxmin_list[1] += [accumulated_metrics[matching_indices, 0].min()]
		size_maxmin_list[0] += [accumulated_metrics[matching_indices, 1].max()]
		size_maxmin_list[1] += [accumulated_metrics[matching_indices, 1].min()]

		# reverse to calculate hv
		copy_arr = accumulated_metrics.copy()
		copy_arr[matching_indices, 0] *= -1
		hv_list[0] += [ind(copy_arr)]

		# get number of clusters
		eps = 1.0  
		min_samples = 1
		num_clusters = find_number_of_clusters(accumulated_metrics[matching_indices], eps, min_samples)
		ncluster_list[0] += [num_clusters]

	line_plot([range(1,explored_metrics.shape[0]+1)],hv_list,['Hypervolume'],colors,
			f'images/hv_vs_ngen.pdf',
			'# of Generations','HV',lbsize=24,lgsize=18,linewidth=2,
			ncol=0,markersize=4,ratio=0.4,xticks=range(0,55,10))
	line_plot([range(1,explored_metrics.shape[0]+1) for _ in range(2)],psnr_maxmin_list,['Max','Min'],colors,
			f'images/psnr_vs_ngen.pdf',
			'# of Generations','PSNR (dB)',lbsize=24,lgsize=18,linewidth=2,
			ncol=1,markersize=4,ratio=0.4,xticks=range(0,55,10))
	line_plot([range(1,explored_metrics.shape[0]+1) for _ in range(2)],size_maxmin_list,['Max','Min'],colors,
			f'images/size_vs_ngen.pdf',
			'# of Generations','Size (MB)',lbsize=24,lgsize=18,linewidth=2,
			ncol=1,markersize=4,ratio=0.4,xticks=range(0,55,10))
	line_plot([range(1,explored_metrics.shape[0]+1)],ncluster_list,[''],colors,
			f'images/ncluster_vs_ngen.pdf',
			'# of Generations','# of Clusters',lbsize=24,lgsize=18,linewidth=2,
			ncol=1,markersize=4,ratio=0.4,xticks=range(0,55,10))
	

def plot_knobs():
	graph_idx = 0
	lbsize = 20
	ratio = 2
	markersize = 200
	psnr_list = []
	size_list = []
	with open('knob.log','r') as f:
		lines = f.readlines()
		for line in lines:
			if line == '\n':
				fig, ax = plt.subplots()
				plt.scatter(size_list, psnr_list, color='r', marker='*',s=markersize)
				plt.xticks(fontsize=lbsize)
				plt.yticks(fontsize=lbsize)
				plt.xlabel('Size (MB)', fontsize = lbsize)
				plt.ylabel('PSNR (dB)', fontsize = lbsize)
				xleft, xright = ax.get_xlim()
				ybottom, ytop = ax.get_ylim()
				ax.set_aspect(abs((xright-xleft)/(ybottom-ytop))*ratio)
				plt.tight_layout()
				fig.savefig(f'images/knob{graph_idx}.pdf',bbox_inches='tight')
				plt.close()
				graph_idx += 1
				psnr_list = []
				size_list = []
				continue
			idx,psnr,size = line.strip().split(',')
			psnr_list += [float(psnr)]
			size_list += [float(size)]

def plot_device_profile():
	fps_list = []
	filenames = [f'profile_{n}.log' for n in ['mbp','iphone','surf','dell']]
	for filename in filenames:
		with open('meta/'+filename,'r') as f:
			lines = f.readlines()
			for line in lines:
				target,channels,_,fps = line.strip().split(',')
				fps_list += [float(fps)]
	fps_list = np.array(fps_list)
	fps_list = fps_list.reshape(4,2,6,5)
	labels = ['MBP','iPhone6','Dell','Surface']
	channel_list = [[16*i for i in range(1,7)] for _ in range(4)]
	for i in range(2):
		fps_mean = fps_list[:,i].mean(axis=-1)
		fps_std = fps_list[:,i].std(axis=-1)

		line_plot(channel_list,fps_mean,labels,colors,
				f'images/device_profile{i}.pdf',
				'# of Channels','FPS',lbsize=24,lgsize=18,linewidth=4,
				ncol=1,markersize=0,markevery=1,yerr=fps_std)
		

def plot_fps():
	fps_list = []
	filenames = [f'profile_{n}.log' for n in ['mbp','iphone','surf','dell']]
	for filename in filenames:
		with open('meta/'+filename,'r') as f:
			lines = f.readlines()
			for line in lines:
				target,channels,_,fps = line.strip().split(',')
				fps_list += [float(fps)]
	fps_list = np.array(fps_list)
	fps_list = fps_list.reshape(4,2,6,5)
	labels = ['Ours','Default']
	selected_cfgs = [[3,2,1,1],
				  	[3,1,0,0]]
	lbsize = 24
	colors = ['pink', 'lightblue']
	for i in range(2):
		fps_mean = fps_list[:,i].mean(axis=-1)
		fps_std = fps_list[:,i].std(axis=-1)
		mean_row0 = fps_mean[range(4),selected_cfgs[i]]
		std_row0 = fps_std[range(4),selected_cfgs[i]]
		mean_row1 = fps_mean[:,0]
		std_row1 = fps_std[:,0]
		y_mean = np.stack((mean_row0,mean_row1)).T
		y_std = np.stack((std_row0,std_row1)).T
		envs = ['MBP','iPhone6','Dell','Surface']
		groupedbar(y_mean,y_std,'Rendering Speed (fps)', 
			f'images/fps{i}.pdf',methods=labels,labelsize=lbsize,xlabel='Devices',
			envs=envs,ncol=1,width=1./4,sep=1,legloc='best',lgsize=lbsize,colors=colors)

		
def plot_mimatch_compute():
	fps_list = []
	filenames = [f'profile_{n}.log' for n in ['mbp','iphone','surf','dell']]
	for filename in filenames:
		with open('meta/'+filename,'r') as f:
			lines = f.readlines()
			for line in lines:
				target,channels,_,fps = line.strip().split(',')
				fps_list += [float(fps)]
	fps_list = np.array(fps_list)
	fps_list = fps_list.reshape(4,2,6,5)
	fps_list = np.transpose(fps_list, axes=(1,0,2,3))
	fps_mean = fps_list[:,:,0].mean(axis=-1)
	fps_std = fps_list[:,:,0].std(axis=-1)
	methods = ['MBP','iPhone6','Dell','Surface']
	envs = ['Synthetic','Realistic']
	groupedbar(fps_mean,fps_std,'FPS', 
		'images/mismatch_compute.pdf',methods=methods,labelsize=24,xlabel='Scene Types',
		envs=envs,ncol=2,width=1./6,sep=1,legloc='best',lgsize=20,colors=colors)
	
def measurements_to_cdf(latency,epsfile,labels,xticks=None,xticklabel=None,yticks=None,yticklabel=None,linestyles=linestyles,colors=colors,
                        xlabel='Normalized QoE',ylabel='CDF',ratio=None,lbsize = 18,lfsize = 18,linewidth=4,bbox_to_anchor=(0.5,-.5),
                        loc='upper center',ncol=3):
    # plot cdf
    fig, ax = plt.subplots()
    ax.grid(zorder=0)
    for i,latency_list in enumerate(latency):
        N = len(latency_list)
        cdf_x = np.sort(np.array(latency_list))
        cdf_p = np.array(range(1,N+1))/float(N)
        plt.plot(cdf_x, cdf_p, color = colors[i], label = labels[i], linewidth=linewidth, linestyle=linestyles[i])
    plt.xlabel(xlabel, fontsize = lbsize)
    plt.ylabel(ylabel, fontsize = lbsize)
    if xticks is not None:
        plt.xticks(xticks,fontsize=lbsize)
    if xticklabel is not None:
        ax.set_xticklabels(xticklabel)
    if yticks is not None:
        plt.yticks(yticks,fontsize=lbsize)
    if yticklabel is not None:
        ax.set_yticklabels(yticklabel)
    if ratio is not None:
        xleft, xright = ax.get_xlim()
        ybottom, ytop = ax.get_ylim()
        ax.set_aspect(abs((xright-xleft)/(ybottom-ytop))*ratio)
    if bbox_to_anchor is not None:
    	plt.legend(loc=loc,fontsize = lfsize,bbox_to_anchor=bbox_to_anchor, fancybox=True,ncol=ncol)
    else:
    	plt.legend(loc=loc,fontsize = lfsize, fancybox=True,ncol=ncol)
    plt.tight_layout()
    fig.savefig(epsfile,bbox_inches='tight')
    plt.close()

def plot_mismatch_rate():
	import glob
	objects = ['chair', 'drums', 'ficus', 'hotdog', 'lego', 'materials', 'mic', 'ship',
                         'fern', 'flower', 'fortress', 'horns', 'leaves', 'orchids', 'room', 'trex']
	
	rates = []
	for object_name in objects:
		# calculate file size
		mlp_size = os.path.getsize(object_name + '_phone/mlp.json')

		png_size = 0
		for file in glob.glob(object_name + '_phone/shape[0-9].pngfeat[0-9].png'):
			png_size += os.path.getsize (file)

		drc_size = 0
		for file in glob.glob(object_name + f'_phone/' + '*.obj'):
			drc_size += os.path.getsize (file)

		rates += [[(mlp_size + png_size + drc_size) / 2**20]]
	rates = np.array(rates).reshape(2,-1)

	labels = ['Synthetic','Realistic']
	measurements_to_cdf(rates,f'images/mismatch_rate.pdf',labels,linestyles=linestyles,
		colors=colors4,lfsize=24,ncol=1,lbsize=24,xlabel='Size (MB)',bbox_to_anchor=None,loc='best',
		)
	

def plot_bitwidth_finetuning():
	bw_list = [[],[]]
	psnr_list = [[],[]]
	with open('flower_C96_P8_weights/multi_rate.log','r') as f:
		lines = f.readlines()
		for i,line in enumerate(lines):
			bw,psnr,_,_ = line.strip().split(',')
			bw_list[i%2] += [int(bw)]
			psnr_list[i%2] += [float(psnr)]
	labels = ['w/o Finetuning','w/ Finetuning']
	line_plot(bw_list,psnr_list,labels,colors,
			f'images/bitwidth_fintuning.pdf',
			'#Bits','PSNR (dB)',lbsize=24,lgsize=24,linewidth=2,
			ncol=1,markersize=8,markevery=1,ratio=0.5,bw_annot=True)
	
def plot_vary_mlp():
	ch_list = [[],]
	psnr_list = [[],]
	with open('flower_C96_P8_weights/multi_channel.log','r') as f:
		lines = f.readlines()
		for i,line in enumerate(lines):
			ch,psnr,ssim,lpips = line.strip().split(',')
			if ch == '-1': 
				continue
			else:
				ch = 96-int(ch)
			ch_list[0] += [ch]
			psnr_list[0] += [float(psnr)]
	labels = ['w/ Multi-channel training']
	line_plot(ch_list,psnr_list,labels,colors,
			f'images/multi_channel_training.pdf',
			'# of Channels','PSNR (dB)',lbsize=24,lgsize=24,linewidth=2,
			ncol=0,markersize=8,markevery=1,ratio=0.5,mlp_annot=True)
	
def plot_nerf_speed():
	# Define a colormap and normalize it
	cmap = plt.get_cmap('viridis')  # You can choose any other colormap here
	norm = plt.Normalize(vmin=0, vmax=7)


	lbsize = 22
	fps_list = [[37.14,55.89, 53.67, 77.40, 178.26, 606.73, 744.91],
			 	[22.62, 8.30, 43.87, 207.26],
			 	[20],
				[1./33],
				[1./73],
				[1./1969],
				[1./2303],
				[1./5195]]
	
	log_fps_list = [[np.log10(value) for value in sublist] for sublist in fps_list]

	device_list = [[0,1,2,3,4,5,6],
					[2,4,5,6],
					[7],
					[7],
					[7],
					[7],
					[7],
					[7],]
	labels = ['MobileNeRF',
		   		'SNeRG',
				'RERF',
				'MLP-Maps',
				'NV',
				'C-NeRF',
				'D-NeRF',
				'DyNeRF']
	xticklabel = ['Pixel 3', 'iPhoneXS', 'Chromebook', 'Surface', '2070', '2070*', '2080Ti', '3090Ti']
	yticklabel = ['$10^{-4}$','$10^{-3}$','$10^{-2}$','$10^{-1}$'] + [f'$10^{i}$' for i in range(0,3)]
	colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

	fig, ax = plt.subplots()
	for i in range(8):
		plt.scatter(device_list[i], log_fps_list[i], 
				color=colors[i], label=labels[i],facecolors='none', marker=markers[i], s=200)

	# Add labels and legend
	plt.xticks(range(8),fontsize=lbsize,rotation=60)
	ax.set_xticklabels(xticklabel)
	plt.yticks(range(-4,3),fontsize=lbsize)
	ax.set_yticklabels(yticklabel)
	plt.xlabel('Device Name from Slow (Left) to Fast (Right)', fontsize = lbsize)
	plt.ylabel('Rendering Speed (fps)', fontsize = lbsize)
	plt.legend(ncol=2, fontsize = 17)
	plt.tight_layout()
	fig.savefig(f'images/nerf_speed.pdf',bbox_inches='tight')
	plt.close()

def plot_profile_vs_views():
	view_list = [1,2,3,4,5,10,20,0]
	profile_train_list = []
	profile_eval_list = []
	for n in [10,0]:
		with open(f'meta/profile_flowerH.32.10.{n}.eval.log', 'r') as file:
			lines = file.readlines()
			profile_eval = np.array(eval(lines[0]))[:,:2]
			profile_eval_list += [profile_eval]

	for n in view_list:
		with open(f'meta/profile_flowerH.32.10.{n}.train.log', 'r') as file:
			lines = file.readlines()
			profile_train = np.array(eval(lines[1]))
			profile_train[..., 0] *= -1
			profile_train_list += [profile_train]

	with open('meta/profile_flowerH.32.default.log', 'r') as file:
		lines = file.readlines()
		default_train = lines[0].strip().split(',')
		default_eval = lines[1].strip().split(',')

	# with open(f'meta/profile_flowerH.32.optimal.log', 'r') as file:
	# 	lines = file.readlines()
	# 	profile_optimal = np.array(eval(lines[1]))[:,:2]
	# 	profile_optimal[...,0] *= -1
	# 	profile_list += [profile_optimal]

	ref_point = np.array([0,float(default_eval[1])])

	ind = HV(ref_point=ref_point)

	hv_list = []

	lbsize = 24

	# profile
	markers = ['*','o',]
	colors = colors4[::2]
	labels = ['Sampling 10 Views','Sampling All Views']
	fig, ax = plt.subplots()
	for i in range(len(profile_eval_list)):
		profile = profile_eval_list[i]
		plt.scatter(profile[:,1], profile[:,0], color=colors[i], label=labels[i],facecolors='none', marker=markers[i],s=200)
		
	# Add labels and legend
	plt.xticks(fontsize=lbsize)
	plt.yticks(fontsize=lbsize)
	plt.xlabel('Size (MB)', fontsize = lbsize)
	plt.ylabel('PSNR (dB)', fontsize = lbsize)
	plt.legend(ncol=1, fontsize = lbsize)
	plt.tight_layout()
	fig.savefig(f'images/profile_vary_views.pdf',bbox_inches='tight')
	plt.close()

	for i in range(len(profile_train_list)):
		profile = profile_train_list[i]
		profile[..., 0] *= -1
		hv_list += [ind(profile)]
	print(i,hv_list)

	line_plot([view_list[:-1]+[29]],[hv_list],['Hypervolume'],["#004c6d"],
			f'images/hv_vs_views.pdf',
			'# of Sampled Views','HV',lbsize=24,lgsize=18,linewidth=2,xticklabel=[0,5,10,15,20,25,'Max'],xticks=[0,5,10,15,20,25,29],
			ncol=0,markersize=8,hv_annot2=True)

def plot_profile_test():
	profiles = []
	for n in [16,32,48,64]:
		configs,metrics,explored_configs,explored_metrics = data_from_profiling(f'meta/profile_flowerH.{n}.optimal.log')
		profiles += [metrics]

	base_psnr,base_size = 34.74024532808376,89.91993045806885

	lbsize = 24
	labels = ['Surface','Dell','iPhone6','MBP',]

	# final profile
	fig, ax = plt.subplots()
	ax.grid(zorder=0)
	for i in range(len(profiles)):
		profile = profiles[i]
		plt.scatter(profile[:,1], profile[:,0], 
				color=colors[i], label=labels[i],facecolors='none', marker=markers[i])
	plt.scatter([base_size], [base_psnr], color='k', label=f'Baseline',marker='*',)

	# Add labels and legend
	plt.xticks(fontsize=lbsize)
	plt.yticks(fontsize=lbsize)
	plt.xlabel('Size (MB)', fontsize = lbsize)
	plt.ylabel('PSNR (dB)', fontsize = lbsize)
	plt.legend(ncol=1, fontsize = lbsize)
	plt.tight_layout()
	fig.savefig(f'images/profile_test.pdf',bbox_inches='tight')
	plt.close()


def plot_clustered_stacked(dfall, filename, labels=None, horizontal=False, xlabel='', ylabel='',**kwargs):
    """Given a list of dataframes, with identical columns and index, create a clustered stacked bar plot. 
labels is a list of the names of the dataframe, used for the legend
title is a string for the title of the plot
H is the hatch used for identification of the different dataframe"""
    fig = plt.figure()
    n_df = len(dfall)
    n_col = len(dfall[0].columns) 
    n_ind = len(dfall[0].index)
    axe = plt.subplot(111)

    for df in dfall : # for each data frame
        axe = df.plot(kind="bar",
                      linewidth=0,
                      stacked=True,
                      ax=axe,
                      legend=False,
                      grid=False,
                      color=colors[:n_col],
                      edgecolor='k',
                      **kwargs)  # make bar plots

    h,l = axe.get_legend_handles_labels() # get the handles we want to modify
    for i in range(0, n_df * n_col, n_col): # len(h) = n_col * n_df
        for j, pa in enumerate(h[i:i+n_col]):
            for rect in pa.patches: # for each index
                rect.set_x(rect.get_x() + 1 / float(n_df + 1) * i / float(n_col))
                rect.set_hatch(hatches[i//n_col]) #edited part     
                rect.set_width(1 / float(n_df + 1))

    axe.set_xticks((np.arange(0, 2 * n_ind, 2)) / 2.)
    axe.set_xticklabels(df.index, rotation = 0)
    axe.tick_params(axis='both', which='major', labelsize=20)
    axe.set_xlabel(xlabel, size=24)
    axe.set_ylabel(ylabel, size=24)

    # Add invisible data to add another legend
    n=[]        
    for i in range(n_df):
        n.append(axe.bar(0, 0, color="white", hatch=hatches[i],edgecolor='black'))

    n2 = []
    for i,clr in enumerate(colors[:n_col]):
        n2.append(axe.bar(0, 0, color=clr))
    if labels is not None: 
	    l3 = plt.legend(n2, ['Continuous Training','Binarized Training','Data Preparation','Finetuning'], loc=[.01, 0.58], fontsize=18) 
	# l2 = plt.legend(n, labels, loc=[.01, 0.47], fontsize=18) 
    axe.add_artist(l3)
    plt.tight_layout()
    fig.savefig(filename, bbox_inches='tight')
    plt.close()
    return axe

def plot_time():
	import pandas as pd
	
	t1 = [210*60+54,209*60+1,209*60+19,212*60+38,209*60+6,224*60+21,
	   225*60+13,222*60+37,222*60+1,222*60+4]
	t2 = [206*30+11+60*60+20, 209*60+13+61*60+15, 240*60+33+61*60+12, 216*60+57+61*60+10,
	   		250*60+43+61*60+14,216*60+20+60*60+37,210*60+7+62*60+17,208*60+30+60*60+42,
			223*60+16+62*60+39,245*60+22+62*60+40,219*60+46+61*60+51]
	t3 = [173*60+73, 182*60+86, 172*60+67,192*60+66,172*60+42,173*60+74,174*60+105,]

	o1 = [38811,38261,41160,40675,38211]
	o2 = [13254+4111,14874+3993,15453+4566,13974+4173,13096+3993]
	o3 = [10878,10707,2489,1474,10631]
	o4 = [64*60+6+46*60+30+46*60+33+51*60+4,76*60+15+50*60+38+46*60+5+50*60+37,
	   		70*60+51*60+3+51*60+3+51*60+3,87*60+47+50*60*3+58,
			50*60+13+46*60+56+46*60+50+46*60+56]

	train_time = [[[sum(t)/len(t) for t in [t1,t2,t3]] + [0],
				[sum(o)/len(o) for o in [o1,o2,o3,o4]]]]
	train_time = np.array(train_time)/3600

	indices = ["Default","Ours"]
	columns = ["Stage1", "Stage2", "Stage3",'Stage4']
	df1 = pd.DataFrame(train_time[0],
					index=indices,
					columns=columns)

	plot_clustered_stacked([df1],'images/time_breakdown.pdf',labels=[],
		xlabel='Methods',ylabel='Training Duration (Hours)',horizontal=False)

def plot_bw_gain():
	objects = ['lego','hotdog']
	psnr_list = []
	for object in objects:
		with open(f'{object}_C64_P5_weights/multi_rate.log','r') as f:
			lines = f.readlines()
			for i,line in enumerate(lines):
				bw,psnr,ssim,lpips = line.strip().split(',')
				if int(bw)>4:break
				psnr_list += [float(psnr)]
	psnr_list = np.array(psnr_list).reshape(len(objects),4,2)
	psnr_list = np.transpose(psnr_list, axes=(1,2,0)).reshape(8,len(objects)).tolist()

	lbsize = 24

	# Create labels for x-axis
	labels = ['1', '2', '3', '4']

	# Define positions for boxplots
	deviation = 0.15
	positions = np.array([i//2-deviation if i%2==0 else i//2+deviation for i in range(8)])

	fig, ax = plt.subplots()
	# ax.grid(zorder=0)

	# Create a boxplot with grouped data
	bp1 = plt.boxplot(psnr_list[::2],positions=positions[::2],widths=0.3,patch_artist=True)
	bp2 = plt.boxplot(psnr_list[1::2],positions=positions[1::2],widths=0.3,patch_artist=True)

	# fill with colors
	colors = ['pink', 'lightblue']
	for i,bplot in enumerate([bp1, bp2]):
		for box in bplot['boxes']:
			box.set_facecolor(colors[i])

	# Set custom xticks and xticklabels
	plt.xticks(range(0,4), labels)

	# Set labels and title
	plt.xlabel('Bit Width', fontsize = lbsize)
	plt.ylabel('PSNR (dB)', fontsize = lbsize)

	# Create legends
	ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ['w/o BWF', 'w/ BWF'], loc='best', fontsize = lbsize)
	plt.tight_layout()
	fig.savefig(f'images/bw_gain.pdf',bbox_inches='tight')
	plt.close()

def plot_mlp_gain():
	objects = ['lego','hotdog']
	psnr_list = []
	for object in objects:
		with open(f'{object}_C64_P5_weights/multi_channel.log','r') as f:
			lines = f.readlines()
			for i,line in enumerate(lines):
				_,psnr,ssim,lpips = line.strip().split(',')
				psnr_list += [float(psnr)]
	psnr_list = np.array(psnr_list).reshape(len(objects),5)
	psnr_list = np.transpose(psnr_list, axes=(1,0)).tolist()

	lbsize = 24

	# Create labels for x-axis
	labels = [16*i for i in range(4,0,-1)] + ['Default']

	fig, ax = plt.subplots()

	# Create a boxplot with grouped data
	bp1 = plt.boxplot(psnr_list,widths=0.3,patch_artist=True)

	# fill with colors
	colors = ['pink', 'lightblue']
	for i,bplot in enumerate([bp1, ]):
		for box in bplot['boxes']:
			box.set_facecolor(colors[i])

	# Set custom xticks and xticklabels
	plt.xticks(range(1,len(labels)+1), labels)

	# Set labels and title
	plt.xlabel('# of Channels', fontsize = lbsize)
	plt.ylabel('PSNR (dB)', fontsize = lbsize)

	# Create legends
	plt.tight_layout()
	fig.savefig(f'images/cw_gain.pdf',bbox_inches='tight')
	plt.close()

# plot_profile()
plot_device_profile()
# plot_bitwidth_finetuning()
# plot_nerf_speed()
# plot_profile_vs_views()
# plot_vary_mlp()
# plot_profile_test()
# plot_pf_metrics()
# plot_mimatch_compute()
# plot_mismatch_rate()
# plot_bw_gain()
# plot_mlp_gain()
# plot_time()
plot_fps()