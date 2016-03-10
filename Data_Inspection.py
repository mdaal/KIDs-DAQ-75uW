import numpy as np
import tables
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import KAM

data_directory_prefix = 'Z:\\user\\miguel\\KIPS_Data'

class data_inspector:
	def __init__(self,data_dir):
		self.start_directory = os.getcwd()
		self.data_directory_path = data_directory_prefix + os.sep + data_dir
		self.measurement_metadata = {}

	def load_hf5(self, database_filename, tablepath):
		''' table path is path to the database to be loaded starting from root. e.g. self.load_hf5('/Run44b/T201312102229')
		database_filename is the name of the hf5 database to be accessed for the  table informaiton'''
		database_filename = self.data_directory_path + os.sep+ database_filename
		if not os.path.isfile(database_filename):
			logging.error('Speficied h5 database does not exist. Aborting...')
			return 
		
		wmode = 'a'
		self.measurement_metadata = {} # Clear metadata

		# use "with" context manage to ensure file is always closed. no need for fileh.close()
		with tables.open_file(database_filename, mode = wmode) as fileh:
			table = fileh.get_node(tablepath)	
			self.Sweep_Array = table.read()
			for key in table.attrs.keys:
				exec('self.measurement_metadata["{0}"] = table.attrs.{0}'.format(key))

		self.sweep_data_columns = self.Sweep_Array.dtype

	def plot_sweep(self, sweep_array_parameter_name):
		plt.rcParams["axes.titlesize"] = 10
		fig = plt.figure( figsize=(8, 6), dpi=100)
		ax = fig.add_subplot(111)
		for i in xrange(self.Sweep_Array.size):
			line = ax.plot(self.Sweep_Array[i]['Frequencies'],20*np.log10(np.abs(self.Sweep_Array[i]['S21'])),'b-',label = '{} V'.format(self.Sweep_Array[i][sweep_array_parameter_name]))
			ax.set_xlabel('Frequency [Hz]')
			ax.set_ylabel('$20*Log_{10}[|S_{21}|]$ [dB]')

			ax.set_title('Power Sweep; Run: {0}; Sensor: {1}; Ground: {2}'.format(self.measurement_metadata['Run'], 
																	self.measurement_metadata['Sensor'], 
																	self.measurement_metadata['Ground_Plane'], 
																	))		
			ax.legend(loc = 'best', fontsize = 5)
			ax.grid()
			plt.draw()
			#plt.show()


	def plot_noise(self, Plot_Off_Res = True):
		plt.rcParams["axes.titlesize"] = 10
		fig = plt.figure( figsize=(8, 6), dpi=100)
		ax = fig.add_subplot(111)

		line = ax.loglog(self.Sweep_Array[0]['Noise_Freq_Vector'],self.Sweep_Array[0]['Noise_II_On_Res'], label = 'PII On')
		line = ax.loglog(self.Sweep_Array[0]['Noise_Freq_Vector'],self.Sweep_Array[0]['Noise_QQ_On_Res'], label = 'PQQ On')
		line = ax.loglog(self.Sweep_Array[0]['Noise_Freq_Vector'],np.abs(self.Sweep_Array[0]['Noise_IQ_On_Res']), label = 'PIQ On')
		if Plot_Off_Res: 
			line = ax.loglog(self.Sweep_Array[0]['Noise_Freq_Vector'],self.Sweep_Array[0]['Noise_II_Off_Res'], label = 'PII Off')
			line = ax.loglog(self.Sweep_Array[0]['Noise_Freq_Vector'],self.Sweep_Array[0]['Noise_QQ_Off_Res'], label = 'PQQ Off')
			line = ax.loglog(self.Sweep_Array[0]['Noise_Freq_Vector'],np.abs(self.Sweep_Array[0]['Noise_IQ_Off_Res']), label = 'PIQ Off')


		ax.set_xlabel('Frequency')
		ax.set_ylabel('$V^2/Hz$')

		ax.set_title(r'I and Q PSD, $f_{{noise}}$ = {fn:0.3f} MHz'.format(fn = self.Sweep_Array[0]['Noise_Freq_On_Res']/1e6))		
		ax.legend(loc = 'best', fontsize = 6)
		ax.grid(which='both')
		ax.set_ylim(bottom = 8e-17)
		plt.draw()