'''

April 13, 2021
Title: Implementation of regression techniques on time-series data to
generate future predictions
Module: Input processing and transformation

'''

import os
import re
import numpy as np
import pandas as pd


#Iterate through each of the sub-directories and extract the group, instance name as well as given data in mem.log to create a dataframe
'''
dataframe format 
"{group}:{instance}:{timestamp}:{Memory Allocated}:{Memory Used}:{CPU Allocated}:{CPU
Used}:{Network bandwidth utilization}:{Storage space utilization}"
'''
class InputClass:
	def __init__(self, directory):
		self.directory = directory
	def process_data(self):
		data = []
		df = pd.DataFrame()
		for folder in sorted(os.listdir(directory)):
		    print("Opening folder "+folder+"...")
		    for file_log in sorted(os.listdir(directory+'/'+folder)):
			folder_name = re.split('_*_', folder)
			with open(directory+'/'+folder+'/'+file_log) as lines:
			    lines = lines.readlines()
			for line in lines:
			    part = line.split(":")
			    if len(part) >= 8:
				data.append((str(folder_name[0]+folder_name[1]), str(folder_name[2]), str(part[0][1:])+":"+str(part[1])+":"+str(part[2].split('\"')[0]), str(part[3].split('\"')[1]), part[4], part[5], part[6], part[7], part[8].split('\"')[0]))
			df = df.append(data)
			data = []
		df.columns=['group', 'instance', 'timestamp', 'Memory_Allocated', 'Memory_Used', 'CPU_Allocated', 'CPU_Used', 'Network_bandwidth_utilization', 'Storage_space_utilization']

		#to dump the dataframe into a csv file
		print("Starting write into csv file...")
		df.to_csv('./data.csv')
		print("Complete!")
if __name__ ==  '__main__':
	directory = './group82_resource_utilization'
	input_data = InputClass(directory)
	input_data.process_data()

