#!/usr/bin/python

import matplotlib.pyplot as plt

def plot_gragh_left(ax, plot_tensors, title, o1, o2, o3):
	if plot_tensors == "real":
		xnames = s3tsrs_names + s4tsrs_names
	elif plot_tensors == "graph":
		xnames = s3tsrs_pl_names + s4tsrs_pl_names

	ind = 1.2 * np.arange(len(o1))
	ylim_var = 1

	rects1 = ax.bar(left=ind, height=o1, width=mywidth, color='limegreen', zorder=2, lw=0.5, label='gpu-coo')
	rects2 = ax.bar(left=ind + mywidth, height=o2, width=mywidth, color='m',  zorder=2, lw=0.5, label='gpu-hicoo')
	rects3 = ax.plot(ind + mywidth, o3, color='r', lw=3, label='roofline')

	ax.set_title(title, fontsize=20)
	ax.set_ylabel('Performance (GFLOPS)', fontsize=16)
	ax.set_xticks(ind + mywidth)
	ax.set_xticklabels(xnames, fontsize=12, rotation=90)

	ax.set_xlim(min(ind) - mywidth, max(ind) + mywidth * 3)
	ax.set_ylim( [0, max(max(o1), max(o2), max(o3)) + ylim_var] )

	# ax.legend()
	ax.grid(axis='y')

	# ax.text(4, -3, "3D", fontweight='bold', fontsize=16)

	return rects1, rects2, rects3


def plot_gragh(ax, plot_tensors, title, o1, o2, o3):
	if plot_tensors == "real":
		xnames = s3tsrs_names + s4tsrs_names
	elif plot_tensors == "graph":
		xnames = s3tsrs_pl_names + s4tsrs_pl_names

	ind = 1.2 * np.arange(len(o1))
	ylim_var = 1

	rects1 = ax.bar(left=ind, height=o1, width=mywidth, color='limegreen', zorder=2, lw=0.5, label='gpu-coo')
	rects2 = ax.bar(left=ind + mywidth, height=o2, width=mywidth, color='m',  zorder=2, lw=0.5, label='gpu-hicoo')
	rects3 = ax.plot(ind + mywidth, o3, color='r', lw=3, label='roofline')

	ax.set_title(title, fontsize=20)
	ax.set_xticks(ind)
	ax.set_xticklabels(xnames, fontsize=12, rotation=90)

	ax.set_xlim(min(ind) - mywidth, max(ind) + mywidth * 3)
	ax.set_ylim( [0, max(max(o1), max(o2), max(o3)) + ylim_var] )

	# ax.legend()
	ax.grid(axis='y')


def plot_gragh_modes(ax, plot_tensors, title, o1, o2, o3):
	if plot_tensors == "real":
		xnames = s3tsrs_names + s4tsrs_names
	elif plot_tensors == "graph":
		# xnames = s3tsrs_pl_names + s4tsrs_pl_names
		xnames = s3tsrs_pl_names 

	ind = 1.2 * np.arange(len(o1))
	ylim_var = 1

	rects1 = ax.bar(left=ind, height=o1, width=mywidth, color='limegreen', zorder=2, lw=0.5, label='m1')
	rects2 = ax.bar(left=ind + mywidth, height=o2, width=mywidth, color='m',  zorder=2, lw=0.5, label='m2')
	rects3 = ax.bar(left=ind + mywidth, height=o2, width=mywidth, color='m',  zorder=2, lw=0.5, label='m3')

	ax.set_title(title, fontsize=20)
	ax.set_ylabel('Performance (GFLOPS)', fontsize=16)
	ax.set_xticks(ind + mywidth)
	ax.set_xticklabels(xnames, fontsize=12, rotation=90)

	ax.set_xlim(min(ind) - mywidth, max(ind) + mywidth * 3)
	ax.set_ylim( [0, max(max(o1), max(o2), max(o3)) + ylim_var] )

	# ax.legend()
	ax.grid(axis='y')

	# ax.text(4, -3, "3D", fontweight='bold', fontsize=16)

	return rects1, rects2, rects3