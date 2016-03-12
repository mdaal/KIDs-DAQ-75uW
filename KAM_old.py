'''This is the KIDs Analysis Module, KAM'''

import urllib2
import scipy.io #for loading .mat file
import os
import numpy as np


import tables
import matplotlib as mpl

#mpl.use("pgf")
pgf_with_pdflatex = {
    "pgf.texsystem": "pdflatex",
    "pgf.preamble": [
         r"\usepackage[utf8x]{inputenc}",
         r"\usepackage[T1]{fontenc}",
         #r"\usepackage{cmbright}",
         ]
}

# pgf_with_latex = {                      # setup matplotlib to use latex for output
#     "pgf.texsystem": "pdflatex",        # change this if using xetex or lautex
#     "text.usetex": True,                # use LaTeX to write all text
#     "font.family": "serif",
#     "font.serif": [],                   # blank entries should cause plots to inherit fonts from the document
#     "font.sans-serif": [],
#     "font.monospace": [],
#     "axes.labelsize": 10,               # LaTeX default is 10pt font.
#     "text.fontsize": 10,
#     "legend.fontsize": 8,               # Make the legend/label fonts a little smaller
#     "xtick.labelsize": 8,
#     "ytick.labelsize": 8,
#     #"figure.figsize": figsize(0.9),     # default fig size of 0.9 textwidth
#     "pgf.preamble": [
#         r"\usepackage[utf8x]{inputenc}",    # use utf8 fonts becasue your computer can handle it :)
#         r"\usepackage[T1]{fontenc}",        # plots will be generated using this preamble
#         ]
#     }


mpl.rcParams.update(pgf_with_pdflatex)
import matplotlib.pyplot as plt
# plt.switch_backend('pgf')



import matplotlib.gridspec as gridspec




import datetime
from scipy.optimize import minimize, curve_fit, leastsq# , root,newton_krylov, anderson
from scipy.interpolate import interp1d
from scipy import constants

import numpy.ma as ma
import sys # for status percentage

import platform
mysys = platform.system()

import warnings #trying to get a warning every time rather than just the first time.
warnings.filterwarnings('always')

database_location = 'Data' + os.sep + 'My_Data_Library.h5'
Working_Dir = os.getcwd()
Plots_Dir = '/Users/miguel_daal/Documents/Projects/Thesis/Thesis/chap6/images/plots'
if os.path.exists(Plots_Dir) == False:
	print('Speficied plots directory does not exist... Using current directory')
	Plots_Dir = Working_Dir


try:
	execfile('KIPs_Access.txt')
except:
	print('KIPs_Access.txt not found. Create this file to download data.')
	# remote access file must have these two lines
	# username = _________  # e.g. 'johndoe'
	# password = _________  # e.g. '294hr5'
	##############

class loop:
	'''The purpose of this class is to hold data associated with resonance loops and fitting them'''
	def __init__(self):
		self.index = None
		self.z =  None
		self.freq = None

		#output of circle fit
		self.r = None
		self.a = None #fit circle center is a+i*b
		self.b = None
		self.outer_iterations = None
		self.inner_iterations = None

		#circle fit parameters
		self.s = None
		self.Gx = None
		self.Gy = None
		self.g = None
		self.sigma = None
		self.circle_fit_exit_code = None

		#loop fit estimates quantities
		self.fr_est = None
		self.FWHM_est = None
		self.depth_est = None
		self.Q_est = None

		# intermediate fit quantities
		self.normalization = 1# used for nonlinear of  generated data probably should  get rid of
		

		# phase fit quantities
		self.Q = None 
		self.Qc = None
		self.Qi = None
		self.fr = None 
		self.FWHM = None
		self.phi = None # in  radians not degrees
		self.theta = None # in  radians not degrees
		self.R = None # outer loop radius
		self.chisquare = None
		self.pvalue = None
		self.phase_fit_method = None
		self.phase_fit_success = None
		self.phase_fit_z = None
		self.phase_fit_mask = None

		# complete fit quantities
		# self.c___ is the phase/magnitude fit
		self.cQ = None 
		self.cQc = None
		self.cQi = None
		self.cfr = None 
		self.cphi = None
		self.cR = None
		self.ctheta = None
		self.cchisquare = None
		self.cphase_fit_success = None

		# self.s___ is the self fit result where sigma^2 is determined from data
		self.sQ = None 
		self.sQc = None
		self.sQi = None
		self.sfr = None 
		self.sphi = None
		self.sR = None
		self.stheta = None
		self.schisquare = None
		self.sphase_fit_success = None


	def __del__(self):
		pass
		

class metadata:
	'''Every data set (e.g. as survey, power sweep, temp sweep) is stored as a pytable. 
	Each pytable has an metadata instance associated with it. 
	This specifies the contents of the metadata.
	'''
	def __init__(self):
		#metadata imported from scan data
		self.Time_Created = None
		self.Atten_Added_At_NA = None # redundant if self.Atten_NA_Input and self.Atten_NA_Output are defined; should be merged somehow
		self.NA_Average_Factor = None
		self.Fridge_Base_Temp = None
		self.Box = None
		self.Ground_Plane = None
		self.Ground_Plane_Thickness = None
		self.LNA = None
		self.IFBW = None
		self.Test_Location = None
		self.Minimum_Q = None
		self.Notes = None
		self.Num_Points_Per_Scan = None
		self.Wait_Time = None
		self.Press = None
		self.Min_Freq_Resolution = None
		self.Run = None
		self.Sensor = None
		self.Fridge_Run_Start_Date = None
		self.Fsteps  = None
		#self.IS_Sonnet_Simulation = None
		self.Data_Source = None
		self.Atten_At_4K = None
		self.Atten_NA_Output = None # positive value in dB
		self.Atten_NA_Input = None # positive value in dB
		self.Atten_RTAmp_Input = None # positive value in dB
		self.RTAmp_In_Use = None
		self.Meaurement_Duration = None
		self.Num_Heater_Voltages = None
		self.Num_Powers = None
		self.Num_Ranges = None 		
		self.Num_Temperatures = None #number of temperature points taken after every scan for each heater voltage/power
		self.Thermometer_Configuration = None
		


		# manual entry metadata
		self.Electrical_Delay = None # Seconds ---  computed at time of data library generation
		self.Resonator_Width = None #list if more than one
		self.Resonator_Thickness = None #list if more than one
		self.Resonator_Impedance = None
		self.Resonator_Eeff = None # Resonator Dielectric Constant
		self.Feedline_Impedance = None
		self.Cable_Calibration = None
		self.Temperature_Calibration = None # a list of tuples [(heatervoltge1, temperature), (heatervoltage2,temperature, ...)]
		self.System_Calibration = None

		self.RTAmp = None
		self.Digitizer = None





class thermometry:
	def __init__(self):
		pass
	def load_MonitoringVI_file(self, filename, temp_list = None, process_therm = 1):
		'''Reads in thermometer data text file created by  MonitoringVI, and plots the temperature as a function of time.
		
		temp_list is a list of tuples, [(heater_voltage,temperature), ...] which are plotted on top of the temperature
		versus time points. This allows one to visually check the calibration, temp_list.

		process_therm is the column number of the thermometer whose data is processed by several filtering algorithms and
		plotted.
		'''

		import io
		from scipy.signal import gaussian,wiener, filtfilt, butter,  freqz
		from scipy.ndimage import filters
		from scipy.interpolate import UnivariateSpline
		pos = filename.rfind(os.sep)

		try:
			with io.open(filename[:pos+1]+ 'Make_ScanData.m',mode='r') as f:
				while 1:
					line  = f.readline()
					if line == '': # End of file is reached
						break
					elif line.find('ScanData.Heater_Voltage') >= 0:
						Voltages = line[line.find('['):line.find(']')+1]
						break
		except:
			print('Unable to find or read Make_ScanData.m for list of heater voltages')
			Voltages = 'Unknown'

 		with io.open(filename,mode='r') as f:
 			
 			temp_data_header = ''
 			while temp_data_header.strip() =='':
 				temp_data_header = f.readline()

 			therm_list = [t for t in temp_data_header.strip().split('\t')[1:] if (t.strip() != 'None') & (t.strip() != '')]
 			

		temp_data = np.loadtxt(filename, dtype=np.float, comments='#', delimiter=None, converters=None, skiprows=3, usecols=None, unpack=False, ndmin=0)
		
		num_col  = temp_data.shape[1]
		start_col = 1 #index of first column in data that has thermometer data
		if process_therm > num_col - start_col:
			print('process_therm = {} exceeds number of thermometers in data. Choose an lower number. Aborting...'.format(process_therm))
			return

		# Gaussian Filter
		num_pts_in_gaussian_window = 20
		b = gaussian(num_pts_in_gaussian_window, 10)
		ga = filters.convolve1d(temp_data[:,process_therm], b/b.sum())

		# buterworth Filter
		npts = temp_data[:,process_therm].size
		end = temp_data[-1,0]
		dt = end/float(npts)
		nyf = 0.5/dt	
		b, a = butter(4, .1)#1.5/nyf)
		fl = filtfilt(b, a, temp_data[:,process_therm])
    	
    	#Spline Fit
		sp = UnivariateSpline(temp_data[:,0], temp_data[:,process_therm])

		#weiner filter
		wi = wiener(temp_data[:,process_therm], mysize=40, noise=10)

		fig1 = plt.figure( facecolor = 'w',figsize = (10,10))
		ax = fig1.add_subplot(1,1,1)		
		
		if isinstance(temp_list, list):
			for temp_tuple in temp_list:
				hline = ax.axhline(y = temp_tuple[1],linewidth=1, color='g', alpha = 0.3 ,linestyle = ':',   label = None)


		color_incr = 1.0/(num_col-start_col)
		for therm_num in xrange(start_col, num_col): # plot all thermometer data present
			line = ax.plot(temp_data[:,0], temp_data[:,therm_num],color=(0,color_incr*therm_num,0), alpha = 0.4 if therm_num != 1 else 1, linewidth = 3,label = therm_list.pop(0) if therm_list[0] != None else 'Therm{0}'.format(therm_num))

		#plot filter outputs for THE FIRST thermometer only 
		line2 = ax.plot(temp_data[:,0], ga, 'y', linewidth = 3, label = 'Gaussian Conv') # Gaussian Convolution
		line3 = ax.plot(temp_data[:,0], fl, 'c', linewidth = 3, label = 'Butterworth') # butterworth 
		line4 = ax.plot(temp_data[:,0], sp(temp_data[:,0]), 'k', linewidth = 3, label = 'Spline') # bspline
		line5 = ax.plot(temp_data[:,0], wi, 'r', linewidth = 3, label = 'Weiner') # weiner

		ax.grid(b=True, which='major', color='b', alpha = 0.2, linestyle='-')
		ax.grid(b=True, which='minor', color='b', alpha = 0.2,linestyle='--')
		ax.set_title('Heater Voltages = {}'.format(Voltages), fontsize=12)
		ax.set_ylabel('Temperature [Kelvin]')
		ax.set_xlabel('Seconds')
		ax.legend(loc = 'best', fontsize=10,scatterpoints =1, numpoints = 1, labelspacing = .1)
		plt.show()
		

class sweep:
	'''This class accesses resonance data and fits resonances'''

	
	
	def __init__(self):
		self.loop = loop()
		self.metadata = metadata()
		
		self.data_set_contents = np.dtype([
			("Run"							, 'S10'),
			("Time_Created"					, 'S40'), # 'S40' for format December 23, 2012 12:34:65.675 PM;  'S12' for format '%Y%m%d%H%M' 
			("Num_Ranges"					, np.uint8), # uint8 is Unsigned integer (0 to 255)
			("Num_Powers"					, np.uint8),
			("Num_Temperature_Readings"		, np.uint8), 
			("Num_Temperatures"				, np.uint8),
			("Sensor"						, 'S20'),
			("Ground_Plane"					, 'S30'),
			("Path"							, 'S100'),
			])

	def _read_scandata_from_file(self,filename_or_path):
		
		# index = filename_or_path.rfind(os.sep)
		# if index > -1: # filename_or_path is a path
		# 	current_path = os.getcwd()
		# 	os.chdir(filename_or_path[0:index])
		# 	mat = scipy.io.loadmat(filename_or_path[index+1:])
		# 	os.chdir(current_path)
		# else: # filename_or_path is a filename
		# 	mat = scipy.io.loadmat(filename_or_path)

		mat = scipy.io.loadmat(filename_or_path)
		self.data = mat
		self.metadata.Data_Source = filename_or_path

	def _download_data(self, URL):
		''' Authenticats to URL containing data.
		Copies the .mat file licated at URL to a local file in local directory.
		.mat file is a Scan_Data matlab structure.
		returns numpy data structure contauning .mat file.
		deletes local file.'''


		passman = urllib2.HTTPPasswordMgrWithDefaultRealm()
		# this creates a password manager
		passman.add_password(None, URL, username, password)
		# because we have put None at the start it will always
		# use this username/password combination for  urls
		# for which `URL` is a super-url

		authhandler = urllib2.HTTPBasicAuthHandler(passman)
		# create the AuthHandler

		opener = urllib2.build_opener(authhandler)

		urllib2.install_opener(opener)
		# All calls to urllib2.urlopen will now use our handler
		# Make sure not to include the protocol in with the URL, or
		# HTTPPasswordMgrWithDefaultRealm will be very confused.
		# You must (of course) use it when fetching the page though.

		pagehandle = urllib2.urlopen(URL)
		# authentication is now handled automatically for us

		#import tempfile # Attempt to download data into a temp file
		#f = tempfile.NamedTemporaryFile(delete=False)
		#f.write(pagehandle.read())
		#f.close()
		#mat = scipy.io.loadmat(f.name)

		output = open('test.mat','wb')
		print('Download Initiated...')
		output.write(pagehandle.read())
		print('Download Completed...')
		output.close()
		#global mat
		mat = scipy.io.loadmat('test.mat')

		#this id how to tell what variables are stored in test.mat
		#print scipy.io.whosmat('test.mat')



		#html = pagehandle.read()
		#pagehandle.close()

		#soup = BeautifulSoup(html)
		#soup.contents
		os.remove('test.mat')
		self.data = mat
		self.metadata.Data_Source = URL

	def plot_loop(self,  aspect='auto', show = True):
		''' Plots currently selected complex transmission in the I,Q plane. Reutrns a tuple, (fig, ax, line),
		where  fig is the figure object, ax is the axes object and line is the line object for the plotted data.

		aspect='equal' makes circles round, aspect='auto' fills the figure space.

		*Must have a loop picked in order to use this function.*
		'''
		try: 
			z = self.loop.z
		except:
			print("Data not available. You probably forgot to load it.")
			return


		fig = plt.figure( figsize=(6.5, 6.5), dpi=100)
		ax = fig.add_subplot(111,aspect=aspect)
		line, = ax.plot(z.real,z.imag,'bo')
		ax.set_xlabel(r'$\Re[S_{21}(f)]$')
		ax.set_ylabel(r'$\Im[S_{21}(f)]$')
		ax.yaxis.labelpad = -2
		ax.set_title('Run: {0}; Sensor: {1}; Ground: {2}; Record Date: {3}'.format(self.metadata.Run, self.metadata.Sensor, self.metadata.Ground_Plane, self.metadata.Time_Created),fontsize=10)
		





		if show == True:
			plt.show()
		return  (fig, ax, line)

	def plot_transmission(self, show = True):
		''' Plots currently selected complex transmission in dB as a function of frequency. Reutrns a tuple, (fig, ax, line),
		where fig is the figure object, ax is the axes object and line is the line object for the plotted data.

		*Must have a loop picked in order to use this function.*
		'''
		try: 
			z = self.loop.z
			freq = self.loop.freq
		except:
			print("Data not available. You probably forgot to load it.")
			return

		plt.rcParams["axes.titlesize"] = 10
		fig = plt.figure( figsize=(8, 6), dpi=100)
		ax = fig.add_subplot(111)
		line = ax.plot(freq,20*np.log10(abs(z)),'b-',)
		ax.set_xlabel('Frequency [Hz]')
		ax.set_ylabel('$20*Log_{10}[|S_{21}|]$ [dB]')

		ax.set_title('Run: {0}; Sensor: {1}; Ground: {2}; Record Date: {3}'.format(self.metadata.Run, self.metadata.Sensor, self.metadata.Ground_Plane, self.metadata.Time_Created))
		if show == True:
			plt.show()
		return  (fig, ax, line)

	def _extract_type(self, obj, return_type = None, field = None):
		'''scanandata object, obj, has a lot of single element arrays of arrays. this function gets the element.
		e.g scandata may have [[[ele]]] instead of callling ele = scandata[0][0][0], use this function to get ele.
		if ele is another structured numpy array with field name 'myfield', using keyword field = 'myfield' will get
		the data at field.
		the function will cast ele to be in the data type return_typer. e.g. return_type = 'str' returns a string. 
		If return_type is None, ele is returned as whatever type it was saved as in [[[ele]]] '''
		
		def cast(_obj):
			if (return_type is not None) & (_obj is not None) : #if (return_type != None) & (_obj != None) :
				_obj = return_type(_obj)
				#pass#exec("_obj = {0}(_obj)".format(return_type))
			return _obj

		def itemloop(_obj):
			while True:
				try:
					_obj = _obj.item()
				except:
					return cast(_obj)
			return cast(_obj)


		if field == None:
			obj = itemloop(obj)

		else:
			while obj.dtype == np.dtype('O'):
				obj = obj.item()
				
			if isinstance(obj.item(), unicode): 
				obj = None
				print('Expected dictionary containing field named {0} is not found. Returning None'.format(field))				
			else: #if the object does not simply contain a string, e.g  [u'InP #2'], do this
				try:
					obj = obj[field]
				except:
					obj = None
					print('Field named {0} is not found. Returning None'.format(field))	
			# try:
			# 	obj = obj[field]
			# except:
			# 	obj = None
			# 	print('Field named {0} is not found. Returning None'.format(field))
			obj = itemloop(obj)
		return obj

	def _define_sweep_data_columns(self, fsteps, tpoints):
		self.metadata.Fsteps = fsteps
		self.metadata.Num_Temperatures  = tpoints

		if tpoints < 1: # we dont want a shape = (0,) array. We want at least (1,)
			tpoints = 1

		self.sweep_data_columns = np.dtype([
			("Fstart"         			, np.float64), # in Hz
			("Fstop"          			, np.float64), # in Hz
			("Heater_Voltage" 			, np.float64), # in Volts
			("Pinput_dB"      			, np.float64), # in dB
			("Preadout_dB"     			, np.float64), # in dB  - The power at the input of the resonator, not inside the resonator
			("Thermometer_Voltage_Bias"	, np.float64), # in Volts
			("Temperature_Readings"    	, np.float64,(tpoints,)), # in Kelvin
			("Temperature"		    	, np.float64), # in Kelvin
			("S21"            			, np.complex128, (fsteps,)), # in complex numbers, experimental values.
			("Frequencies"    			, np.float64,(fsteps,)), # in Hz
			("Q"						, np.float64),
			("Qc"						, np.float64),
			("Fr"						, np.float64), # in Hz
			("Is_Valid"					, np.bool),
			("Chi_Squared"              , np.float64),
			("Mask"						, np.bool,(fsteps,)), # array mask selecting data used in phase fit
			("R"						, np.float64), #outer loop radius
			("r"						, np.float64), # resonance loop radius	
			("a"						, np.float64),	
			("b"						, np.float64),
			#("Normalization"			, np.float64),
			("Theta"					, np.float64),
			("Phi"						, np.float64),
			("cQ"						, np.float64),
			("cQc"						, np.float64),
			("cFr"						, np.float64), # in Hz
			("cIs_Valid"				, np.bool),
			("cChi_Squared"             , np.float64),
			("cPhi"						, np.float64),
			("cTheta"					, np.float64),
			("cR"						, np.float64),
			("sQ"						, np.float64),
			("sQc"						, np.float64),
			("sFr"						, np.float64), # in Hz
			("sIs_Valid"				, np.bool),
			("sChi_Squared"             , np.float64),
			("sPhi"						, np.float64),
			("sTheta"					, np.float64),
			("sR"						, np.float64),

			#("S21_Processed"            , np.complex128, (fsteps,)), # Processed S21 used in phase fit 
			])


	def _define_sweep_array(self,index,**field_names):
		#for field_name in self.sweep_data_columns.fields.keys():
		for field_name in field_names:
			self.Sweep_Array[field_name][index] = field_names[field_name]
						
	def load_scandata(self, file_location):
		''' file_location is the locaiton of the scandata.mat file. It can be a URL, filename or /path/filename.
		assumes that self.data is in the form of matlab ScanData Structure'''

		#delete previous metadata object
		del(self.metadata)
		self.metadata = metadata()

		if file_location.startswith('http'):
			self._download_data(file_location)
		else:
			self._read_scandata_from_file(file_location)

		ScanData = self.data['ScanData']
		
		# These tags specify the data to pull out of self.data['ScanData']. syntax is 
		# (field of self.data['ScanData'] to extract, self.metadata name to save to ('key:sub-key' ifself.metadata.key is a dict), 
		#			 type of value (arrays are None),optional sub-field of self.data['ScanData'] to extract)
		tags = [('Run','Run', str), ('Start_Date','Fridge_Run_Start_Date',str), ('Location','Test_Location', str), 
				('Sensor','Sensor',str), ('Ground_Plane','Ground_Plane',str), ('Box','Box',str), ('Press','Press',str), 
				('Notes','Notes',str),('Time','Time_Created',str),('Temperature','Fridge_Base_Temp',float),
				('Powers','Powers', None), ('Resolution','Min_Freq_Resolution', np.float), ('IFBW','IFBW', np.float),
				('Heater_Voltage','Heater_Voltage',None), ('Average_Factor','NA_Average_Factor', np.float), 
				('Minimum_Q','Minimum_Q', np.float), ('Range','Range',None), ('Added_Atten','Atten_Added_At_NA', np.float),  
				('Num_Points_Per_Scan','Num_Points_Per_Scan',np.float), ('Freq_Range', 'Freq_Range',None), 
				('Pause','Wait_Time',np.float), ('LNA', 'LNA:LNA', str), ('HEMT', 'LNA:Vg', str,'Vg'),
				('HEMT', 'LNA:Id', str,'Id'),  ('HEMT', 'LNA:Vd', str,'Vd'), ('Atten_4K', 'Atten_At_4K', np.float32),
				('Atten_NA_Output', 'Atten_NA_Output',np.float32), ('Atten_NA_Input','Atten_NA_Input',np.float32),
				('Atten_RTAmp_Input','Atten_RTAmp_Input',np.float32), ('RTAmp_In_Use', 'RTAmp_In_Use', int),
				('Elapsed_Time', 'Meaurement_Duration', np.float),('Thermometer_Configuration','Thermometer_Configuration',None),
				('Thermometer_Bias','Thermometer_Voltage_Bias', None)]

		for t in tags:
			try:
				if t[1].find(':')>-1: #The case of a dictionary
					t1 = t[1].split(':')

					#This try/except block is for the case where self.metadata.__dict__['?'] is a dictionary
					try:
						self.metadata.__dict__[t1[0]].update([(t1[1],self._extract_type(ScanData[t[0]], return_type = t[2],field = t[3] if len(t) > 3 else None))])
					except:
						self.metadata.__dict__[t1[0]] = dict([(t1[1],self._extract_type(ScanData[t[0]], return_type = t[2],field = t[3] if len(t) > 3 else None))])	
				else:
					self.metadata.__dict__[t[1]] = self._extract_type(ScanData[t[0]], return_type = t[2],field = t[3] if len(t) > 3 else None)
			except: 
				#the case that the field does not exist or that its in an unexpected format
				#print('Field named {0}{1} is not found. Setting value to None'.format(t[0], (':'+t[3]) if len(t) > 3 else '')) # this usesScandata nomenclature
				print('Field named {0} is not found.'.format(t[1])) # this uses self.metadata nomenclature
		try:
			self.metadata.Powers                = self.metadata.Powers.squeeze() #for case there are multiples powers
		except:
			self.metadata.Powers                = np.array([self.metadata.Powers]) # for the case there is only one power		

		
		# Remove nested array for Thermometer_Voltage_Bias data, if this data exists
		if hasattr(self.metadata,'Thermometer_Voltage_Bias'):
			self.metadata.Thermometer_Voltage_Bias  = self.metadata.Thermometer_Voltage_Bias.reshape((self.metadata.Thermometer_Voltage_Bias.shape[1],))

		if self.metadata.Thermometer_Configuration is not None:#if self.metadata.Thermometer_Configuration != None:
			self.metadata.Thermometer_Configuration = (str(self.metadata.Thermometer_Configuration.squeeze()[0][0]),str(self.metadata.Thermometer_Configuration.squeeze()[1][0]))

		# Reshape  Heater_Voltage array and  Remove final Heater voltage from self.Heater_Voltage (The final value is just the heater value at which to leave fridge )
		self.metadata.Heater_Voltage = self.metadata.Heater_Voltage.reshape((self.metadata.Heater_Voltage.shape[1],))
		self.metadata.Heater_Voltage = self.metadata.Heater_Voltage[0:-1]


		print('Loading Run: {0}'.format(self.metadata.Run))
		print('There are {0} heater voltage(s), {1} input power(s), and {2} frequecy span(s)'.format(self.metadata.Heater_Voltage.shape[0],self.metadata.Powers.shape[0], self.metadata.Freq_Range.shape[0]))
		heater_voltage_num = 0; power_sweep_num = 0; fsteps = 0; tpoints = 0;

		# determine fsteps = length of the freq/S21 array
		if self.metadata.Heater_Voltage.shape[0] == 1:
			fsteps = self.metadata.Freq_Range[heater_voltage_num][1]['PowerSweep'][0][0][power_sweep_num][2].squeeze()[()].size # non temp sweep, single freq_range, powersweep
			try: # Try to determine the number of temperture readings per scan. If data does not contain temp readings, pass
				tpoints = self.metadata.Freq_Range[heater_voltage_num][1]['PowerSweep'][0][0][power_sweep_num][3].squeeze()[()].size
			except:
				pass
		else:					
			for freq_range_num in xrange(self.metadata.Freq_Range.shape[0]):			
				steps = self.metadata.Freq_Range[freq_range_num][1]['Temp'][0][0][heater_voltage_num][1]['PowerSweep'][0][0][power_sweep_num][2].squeeze()[()].size
				fsteps = max(steps,fsteps)
				try: # Try to determine the number of temperture readings per scan. If data does not contain temp readings, pass
					points = self.metadata.Freq_Range[freq_range_num][1]['Temp'][0][0][heater_voltage_num][1]['PowerSweep'][0][0][power_sweep_num][3].squeeze()[()].size
					tpoints = max(points,tpoints)
				except:
					pass


		self._define_sweep_data_columns(fsteps,tpoints)

		
		self.metadata.Num_Powers 			= self.metadata.Powers.size
		self.metadata.Num_Heater_Voltages 	= self.metadata.Heater_Voltage.size
		self.metadata.Num_Ranges 			= self.metadata.Range.shape[0]
		try:
			self.metadata.Cable_Calibration = self._Cable_Calibration
			print('Cable Calibraiton data found and saved in Sweep_Array metadata.')
		except:
			pass

		try:
			self.metadata.Temperature_Calibration = self._Temperature_Calibration
			print('Temperature Calibraiton data found and saved in Sweep_Array metadata.')
		except:
			pass

		if self.metadata.Num_Temperatures > 0:
			print('Temperature readings found for scan(s). {0} readings per scan'.format(self.metadata.Num_Temperatures))
		### Examples of dealing with Freq_Range Data structure imported from Matlab .mat file			
		#    k.Freq_Range[heater_voltage_num][1]['PowerSweep']
		# 					  k.Freq_Range[0][1]['PowerSweep']
		# j.Freq_Range[0][1]['Temp'][0][0][0][1]['PowerSweep']
		# dt = np.dtype(('O', (2,3)))
		# entended = np.zeros(0,dtype = dt)
		# dt = np.dtype(('O',('O',[('Temp',('O',('O')))])))
		# dt = np.dtype([('O',[('O',[('Temp',[('O',('O'))])])])])
		# #this is the closest I can come to replecating the structure of a Temp Power Sweep
		# dt = np.dtype([('O',[('Temp','O',(1,1))],(1,2))])

		
		i=0
		self.Sweep_Array = np.zeros(self.metadata.Heater_Voltage.shape[0]*self.metadata.Powers.shape[0]*self.metadata.Freq_Range.shape[0], dtype = self.sweep_data_columns)
		for freq_range_num in xrange(self.metadata.Freq_Range.shape[0]):
				if self.metadata.Heater_Voltage.shape[0] == 1:
					heater_voltages = self.metadata.Freq_Range # non temp sweep, single freq_range, powersweep
				else:					
					heater_voltages = self._extract_type(self.metadata.Freq_Range[freq_range_num,1]['Temp'])
				#start here for single res powersweep
				for heater_voltage_num in xrange(heater_voltages.shape[0]):
					sweep_powers = self._extract_type(heater_voltages[heater_voltage_num,1], field = 'PowerSweep')
					for sweep in sweep_powers[:,0:sweep_powers.shape[1]]:
						self._define_sweep_array(i, Fstart = self.metadata.Range[freq_range_num,0],
													Fstop = self.metadata.Range[freq_range_num,1],
													Heater_Voltage = self.metadata.Heater_Voltage[heater_voltage_num],
													Thermometer_Voltage_Bias = self.metadata.Thermometer_Voltage_Bias[heater_voltage_num] if hasattr(self.metadata,'Thermometer_Voltage_Bias') else 0,#set to zero unless there is an array of temps in the ScanData
													Pinput_dB = sweep[0].squeeze()[()] - self.metadata.Atten_Added_At_NA if self.metadata.Atten_Added_At_NA != None else sweep[0].squeeze()[()], #we only want the power coming out of the source, i.e. the NA
													S21 = sweep[1].squeeze()[()],
													Frequencies = sweep[2].squeeze()[()],
													Temperature_Readings = sweep[3].squeeze()[()] if (sweep.size > 3) and (np.shape(sweep[3].squeeze()[()])[0] != 0) else np.array([0]), #set to zero unless there is an array of temps in the ScanData
													Is_Valid = True)
						i = i + 1

		if  hasattr(self.metadata,'Thermometer_Voltage_Bias'):
			del(self.metadata.__dict__['Thermometer_Voltage_Bias'])
		del(self.metadata.__dict__['Powers'])
		del(self.metadata.__dict__['Heater_Voltage'])
		del(self.metadata.__dict__['Range'])
		del(self.metadata.__dict__['Freq_Range'])

		# if self.metadata.Atten_NA_Output == None: #redundant to have both
		# 	del(self.metadata.__dict__['Atten_NA_Output'])
		# else:
		# 	del(self.metadata.__dict__['Atten_Added_At_NA'])

	def load_touchstone(self,filename, pick_loop = True):
		''' The function loads S21 and Freq from  Sonnet .s2p or .s3p files into the Sweep_Array structured np array
		All Sij are extracted, but only  S21 is saved into Sweep_Array. Future editions of this code might  find need 
		to load other Sij becuase S21.

		The function only loads one transmission array (S21).  pick_loop = True immediatly selectes this loop as the 
		current loop.
		'''

		import tempfile
		import io

		#delete previous metadata object
		del(self.metadata)
		self.metadata = metadata()

		dt_s2p = [('Freq', np.float64), ('S11r', np.float64), ('S11i', np.float64), ('S12r', np.float64), ('S12i', np.float64), 
										('S21r', np.float64), ('S21i', np.float64), ('S22r', np.float64), ('S22i', np.float64)]
		
		dt_s3p = [('Freq', np.float64), ('S11r', np.float64), ('S11i', np.float64), ('S12r', np.float64), ('S12i', np.float64), ('S13r', np.float64), ('S13i', np.float64),
										('S21r', np.float64), ('S21i', np.float64), ('S22r', np.float64), ('S22i', np.float64), ('S23r', np.float64), ('S23i', np.float64),
										('S31r', np.float64), ('S31i', np.float64), ('S32r', np.float64), ('S32i', np.float64), ('S33r', np.float64), ('S33i', np.float64)] 


		with tempfile.TemporaryFile() as tmp:
			with io.open(filename, mode='r') as f:
				# The following while loop copies the .sNp file into a temp file, which is destroyed when closed,
				# such that the tmp file is formated in the way np.loadtxt can read the data.
				indented = False
				prev_line = ''
				m = 1. # for frequency base conversion
				while 1: 
					line  = f.readline().replace('\n','')

					pos = f.tell()
					if line == '': # End of file is reached
						break
					elif line.startswith('! Data File Written:'): # Save as Metadata
						self.metadata.Time_Created = str(line.split('! Data File Written:')[1].strip())
						tmp.write(line + '\n')
					elif line.startswith('! From Project:') | line.startswith('! From Emgraph Data:'): # Save as Metadata
						self.metadata.Run = str(line.split(':')[1].strip())
						#self.metadata.IS_Sonnet_Simulation = True
						tmp.write(line + '\n')
					elif line[0] == '#':
						line  = line.replace('#','!#')
						if line.find('GHZ') >=-1:
							m = 1.0e9
						freq_convert = lambda s: s*m #Convert to Hertz
						tmp.write(line + '\n')	
					
					elif line[0] == ' ': # in S matrix definition block
						prev_line = prev_line + ' ' + line.strip() + ' '
						next_line = f.readline()
						# if next line is NOT indented date, then S matrix definition block is finished 
						# and we write it to tmp on a single line.
						# for .s2p files the S matrix is fully defined on one line of f
						# for .s3p files, the S matrix is defined in three lines. second two are indented.
						if not ((next_line[0] == '') | (next_line[0] == ' ')):
							tmp.write(prev_line)
							tmp.write('\n')
							prev_line = ''
						f.seek(pos,0)
		
					elif line[0] == '!':
						tmp.write(line + '\n')

					else:
						tmp.write(line)
						next_line = f.readline()
						# add \n to line if it does not begin a S matrix definition block
						if not ((next_line[0] == '') | (next_line[0] == ' ')):
							tmp.write('\n')
						f.seek(pos,0)

			tmp.seek(0,0)
			if filename.endswith('.s2p'):
				dt = dt_s2p
			elif filename.endswith('.s3p'):
				dt = dt_s3p	
			Touchstone_Data = np.loadtxt(tmp, dtype=dt, comments='!', delimiter=None, converters=None, skiprows=0, usecols=None, unpack=False, ndmin=0)
		
		tpoints = 0
		self._define_sweep_data_columns(Touchstone_Data.size, tpoints)
		j = np.complex(0,1)

		self.Sweep_Array = np.zeros(1, dtype = self.sweep_data_columns)
		
		self._define_sweep_array(0, Fstart = freq_convert(Touchstone_Data['Freq'].min()), #Hz
									Fstop = freq_convert(Touchstone_Data['Freq'].max()), #Hz
									S21 = Touchstone_Data['S21r']+j*Touchstone_Data['S21i'],
									Frequencies = freq_convert(Touchstone_Data['Freq']), #Hz
									#Pinput_dB = 0,
									Is_Valid = True,
									#Mask = False, needs to be an array of lengh of S21
									Chi_Squared = 0,
									)


		self.metadata.Data_Source = filename
		#self.metadata.Min_Freq_Resolution = np.abs(Touchstone_Data['Freq'][:-1]-Touchstone_Data['Freq'][1:]).min()
		self.metadata.Min_Freq_Resolution = np.abs(Touchstone_Data['Freq'][0] - Touchstone_Data['Freq'][-1])/Touchstone_Data['Freq'].size #use average freq resolution
	
		if pick_loop == True: #since there is only one loop in Sweep_Array, we might as well pick it as the current loop
			self.pick_loop(0)
			#self.normalize_loop()

	def downsample_loop(self,N):
		''' Reduce number of loop/freq data point by every Nth point and discarding all others'''
		self.loop.z = self.loop.z[0:-1:N]
		self.loop.freq = self.loop.freq[0:-1:N]

	def save_hf5(self, filename = database_location, overwrite = False):
		'''Saves current self.Sweep_Array into table contained in the hdf5 file speficied by filename.
		If overwite = True, self.Sweep_Array will overwright whatever is previous table data there is.
		'''
		
		if not os.path.isfile(filename):
			print('Speficied h5 database does not exist. Creating new one.')
			pos = filename.find('/')
			if pos >= 0:
				try:
					os.makedirs(filename[0:pos+1])
				except OSError:
					print('{0} exists...'.format(filename[0:pos+1]))
			wmode = 'w'
		else:
			print('Speficied h5 database exists and will be updated.')
			wmode = 'a'
			
		db_title = 'Aggregation of Selected Data Sets'
		group_name = 'Run' + self.metadata.Run
		group_title = self.metadata.Test_Location
		try:
			# case for scan data date
			d = datetime.datetime.strptime(self.metadata.Time_Created, '%B %d, %Y  %I:%M:%S.%f %p') # slightly wrong %f is microseconds. whereas we want milliseconds.
		except:
			pass
		try:
			#Case for sonnet date
			d = datetime.datetime.strptime(self.metadata.Time_Created, '%m/%d/%Y %H:%M:%S')
		except:
			pass
		sweep_data_table_name = 'T' + d.strftime('%Y%m%d%H%M')

		

		with tables.open_file(filename, mode = wmode, title = db_title ) as fileh:
			try:
				table_path = '/' + group_name + '/' + sweep_data_table_name
				sweep_data_table = fileh.get_node(table_path)

				if overwrite == True:
					print('Table {0} exists. Overwriting...'.format(table_path))
					sweep_data_table.remove()
					sweep_data_table = fileh.create_table('/'+ group_name,sweep_data_table_name,description=self.sweep_data_columns,title = 'Sweep Data Table',filters=tables.Filters(0), createparents=True)
				else:
					print('Table {0} exists. Aborting...'.format(table_path))
					return
			except:
				print('Creating table {0}'.format('/'+ group_name+'/'+sweep_data_table_name))
				sweep_data_table = fileh.create_table('/'+ group_name,sweep_data_table_name,description=self.sweep_data_columns,title = 'Sweep Data Table',filters=tables.Filters(0), createparents=True)
			
			# copy Sweep_Array to sweep_data_table
			sweep_data_table.append(self.Sweep_Array)

			# Save metadata
			for data in self.metadata.__dict__.keys():
				exec('sweep_data_table.attrs.{0} = self.metadata.{0}'.format(data))
				if self.metadata.__dict__[data] == None:
					print('table metadata {0} not defined and is set to None'.format(data))	

			sweep_data_table.flush()	

			# try:
			# 	TOC = fileh.get_node('/Contents') # is a table
			# except:
			# 	print('Creating h5 data set table of contents')
			# 	TOC = fileh.create_table('/', 'Contents', self.data_set_contents, "Table listing all tables contained in h5 file", tables.Filters(0)) #tables.Filters(0) means there is no data compression

			# TOC.append()

		# title = 'Data from Run ' + self.metadata.Run + ', Sensor: ' + self.metadata.Sensor + ', Ground Plane: ' + self.metadata.Ground_Plane

		# #determine type of  measurement...
		# if  (self.Sweep_Array.size == 1) | (np.abs(self.Sweep_Array['Fstop'] - self.Sweep_Array['Fstart']).max() >= 100e6):
		# 	groupname = 'Survey'
		# elif (np.unique(self.Sweep_Array['Heater_Voltage']).size > 1) && (np.unique(self.Sweep_Array['Pinput_dB']).size == 1):
		# 	groupname = 'T_Sweep'
		# elif (np.unique(self.Sweep_Array['Heater_Voltage']).size == 1) && (np.unique(self.Sweep_Array['Pinput_dB']).size > 1):
		# 	groupname = 'P_Sweep'
		# elif (np.unique(self.Sweep_Array['Heater_Voltage']).size > 1) && (np.unique(self.Sweep_Array['Pinput_dB']).size > 1):
		# 	groupname = 'TP_Sweep'
		# else:
		# 	groupname = 'Sweep'

		# 	groupname = 'T' + str(np.unique(self.Sweep_Array['Heater_Voltage']).size) + 'P' +  str(np.unique(self.Sweep_Array['Pinput_dB']).size)	

	def decompress_gain(self, Compression_Calibration_Index = -1, Show_Plot = True, Verbose = True):
		''' Assumes the two lowest input powers of the power sweep are not gain compressed, thus
		cannot be used if the two lowest powers are gain compressed. '''
		from matplotlib.ticker import MultipleLocator, FormatStrFormatter, MaxNLocator

		Sweep_Array_Record_Index = self.loop.index 
		V = self.Sweep_Array['Heater_Voltage'][Sweep_Array_Record_Index]
		Fs = self.Sweep_Array['Fstart'][Sweep_Array_Record_Index]
		P = self.Sweep_Array['Pinput_dB'][Sweep_Array_Record_Index]

		Sweep_Array = np.extract((self.Sweep_Array['Heater_Voltage'] == V) & ( self.Sweep_Array['Fstart']==Fs) , self.Sweep_Array)


		num_sweep_powers = Sweep_Array['Pinput_dB'].shape[0]

		if num_sweep_powers <= 4:
			print('Number of sweep powers, {0}, is insufficient to perform gain decompression.'.format(num_sweep_powers))
			return
		#else:
		#	print('Performing gain decompression on {0} sweep powers.'.format(num_sweep_powers))

		Pin = np.power(10, Sweep_Array['Pinput_dB']/10.0) #mW, Probe Power

		#ChooseCompression calobration data from Power Sweep Data. 
		#It is the S21(Compression_Calibration_Index) for every sweep power 
		compression_calibration_data = np.power(np.abs(Sweep_Array['S21'][:,Compression_Calibration_Index]),2) #Pout/Pin,  
		# alternatively : np.average(Sweep_Array['S21'][:,Compression_Calibration_Index:Compression_Calibration_Index+n],axis = 1) #average over  n freq points.
		Pout = compression_calibration_data*Pin 

		### TO BE DELETED
		#calculated_power_gain is power gain calculated from the slope of the two smallest input powers in Pin
		# min_index = np.where(Pin == Pin.min())[0][0] # Index of the min values of Pin, unpacked from tuple
		# dif = Pin - Pin.min() 
		# min_plus = dif[np.nonzero(dif)].min() + Pin.min() # Second lowest value of Pin
		# min_plus_index = np.where(Pin == min_plus )[0][0] #  Index of second lowest Pin value - Previous command used to give min_index then Pin.min and min_plus  were really close : min_plus_index = np.where(np.isclose(Pin,min_plus))[0][0] # index of the second lowest Pin value, unpacked from tuple
		###

		# calculated_power_gain is power gain calculated from the slope of the two smallest input powers in Pin
		values, indices = np.unique(Pin, return_index=True)
		min_index,min_plus_index =  indices[:2]   
		# When Pin = 0, 0 != Pout = Pin*gaain. There is an offset, i.e. a y-intercept, b, such at y = m*x+b. Next, we find m.  
		calculated_power_gain = (Pout[min_plus_index] - Pout[min_index])/(Pin[min_plus_index ]-Pin[min_index]) 

		#Pout_ideal is the output power assuming linear gain
		Pout_ideal = lambda p_in: calculated_power_gain*(p_in-Pin[0]) + Pout[0]

		Probe_Power_Mag = np.power(10,self.Sweep_Array[Sweep_Array_Record_Index]['Pinput_dB']/10) #-- Substitute for input power
		S21 = self.Sweep_Array[Sweep_Array_Record_Index]['S21']
		S21_Pout = np.power(np.abs(S21),2)*Probe_Power_Mag

		# create interpolation funcation to what Pin would be at an arbitrary Pout
		decompression_function = interp1d(Pout,Pin,kind = 'linear')

		# for polynomial to Pout vs Pin curve and use this to extrapolate values where Pout in not in interpolation domain
		def decompression_function_fit(pout, a,b,c):
			return a*np.power(pout,2)+b*pout+c
		popt,pcov = curve_fit(decompression_function_fit, Pout, Pin)
		decompression_function_extrap = lambda pout : decompression_function_fit(pout,popt[0],popt[1],popt[2])

		
		def decompress_element(z):
			z_Pout = np.power(np.abs(z),2)*Probe_Power_Mag
			if z_Pout <= Pout.min(): #Do nothinge when z_Pout is less than the interpolation range, Pout.min() to Pout.max()
				return z
			elif Pout.min() < z_Pout < Pout.max(): # Interpolate to find ideal Pout (assuming linear gain) when z_Pout is in interpolation domain 
				return z*np.sqrt(Pout_ideal(decompression_function(z_Pout))/Probe_Power_Mag)/np.abs(z)
			else: # Pout.max() <= z_Pout --  Extrapolate to find ideal Pout when z_Pout is above interpolation domain
				return z*np.sqrt(Pout_ideal(decompression_function_extrap(z_Pout))/Probe_Power_Mag)/np.abs(z)

		decompress_array = np.vectorize(decompress_element) # Vectorize for speed

		self.loop.z = S21_Decompressed = decompress_array(S21)

		if Verbose == True:
			print('Gain decompression calculation is based on {0} sweep powers.'.format(num_sweep_powers))
			print('Power out at zero input power is {0} mW'.format(calculated_power_gain*(0-Pin[0]) + Pout[0]))

		if Show_Plot:
			fig1           = plt.figure(figsize = (15,5))
			Freq           = self.Sweep_Array[Sweep_Array_Record_Index]['Frequencies']
			#majorFormatter = FormatStrFormatter('%d')
			majormaxnlocator    = MaxNLocator(nbins = 5)
			minormaxnlocator    = MaxNLocator(nbins = 5*5)
			#minorLocator   = MultipleLocator((Freq.max() - Freq.min())/25)
			

			ax1 = fig1.add_subplot(131)
			ax1.set_xlabel('Power In [mW]')
			line1 = ax1.plot(Pin,Pout, 'b-', label = 'Measured')
			line2 = ax1.plot(Pin,Pout_ideal(Pin), 'r-', label = 'Ideal')
			ax1.set_ylabel('Power Out [mW]', color='b')
			ax1.set_title('Gain Compression', fontsize=9)
			ax1.legend(loc = 'best', fontsize=9)
			plt.setp(ax1.get_xticklabels(),rotation = 45, fontsize=9)
			ax1.grid()
			#fig1.canvas.manager.resize(800,800)

			
			ax2 = fig1.add_subplot(132, aspect='equal')
			line2 = ax2.plot(S21.real,S21.imag, color='blue', linestyle='solid', linewidth = 3, label = 'Measured') 
			line1 = ax2.plot(S21_Decompressed.real, S21_Decompressed.imag, 'g-',linewidth = 3, label = 'Corrected')
			ax2.grid()
			ax2.set_title('Resonance Loop', fontsize=9)
			plt.setp(ax2.get_xticklabels(),rotation = 45)
			#ax2.legend(loc = 'best')

			
			ax3 = fig1.add_subplot(133)
			ax3.set_xlabel('Freq [Hz]')
			line1 = ax3.plot(Freq,10*np.log10(np.abs(S21)), 'b-',label = 'Measured',linewidth = 3)
			line2 = ax3.plot(Freq,10*np.log10(np.abs(S21_Decompressed)), 'g-', label = 'Corrected',linewidth = 3)
			ax3.set_ylabel('$|S_{21}|$ [dB]', color='k')
			ax3.legend(loc = 'best', fontsize=9)
			ax3.xaxis.set_major_locator(majormaxnlocator)
			#ax3.tick_params( axis='both', labelsize=9)
			plt.setp(ax3.get_xticklabels(),rotation = 45, fontsize=9)
			#ax3.xaxis.set_major_formatter(majorFormatter)
			ax3.xaxis.set_minor_locator(minormaxnlocator)
			ax3.set_title('Resonance Dip', fontsize=9)
			ax3.grid()

			fig1.subplots_adjust(wspace = 0.6,bottom = 0.09, top = 0.1)
			fig1.suptitle('Run: {0}, Sensor: {1}, Ground Plane: {2}, Readout Power: {3} dBm, Date: {4}'.format(self.metadata.Run, self.metadata.Sensor,self.metadata.Ground_Plane,self.Sweep_Array[Sweep_Array_Record_Index]['Pinput_dB'],self.metadata.Time_Created), fontsize=10)
			#plt.tight_layout()
			plt.setp(fig1, tight_layout = True)
			plt.show()

	def sweep_array_info(self):
		''' prints information about the Sweep_Array currently loaded'''
		Input_Powers = np.unique(self.Sweep_Array['Pinput_dB'])
		Heater_Voltages = np.unique(self.Sweep_Array['Heater_Voltage'])
		Temperature_Points = np.shape(self.Sweep_Array['Temperature_Readings'])[1]
		Number_of_Freq_Ranges = max(np.unique(self.Sweep_Array['Fstart']),np.unique(self.Sweep_Array['Fstop']))
		print('{0:03.0f} - Total number of sweeps.\n{1:03.0f} - Number of readout powers.\n{2:03.0f} - Number of readout temperatures.\n{3:03.0f} - Number of temperatures readings.\n{4:03.0f} - Number of frequency bands.'.format(
			self.Sweep_Array.shape[0],
			Input_Powers.shape[0],
			Heater_Voltages.shape[0],
			Temperature_Points,
			Number_of_Freq_Ranges.shape[0]))

	def construct_hf5_toc(self,filename = database_location):
		''' Creates a table of contents (toc) of the hf5 database storing all the sweep_data.
		very useful for finding the name and locaiton of a table in the database'''
		if not os.path.isfile(filename):
			print('Speficied h5 database does not exist. Aborting...')
			return 
		
		wmode = 'a'

		# use "with" context manage to ensure file is always closed. no need for fileh.close()
		with tables.open_file(filename, mode = wmode) as fileh:
			table_list = [g for g in fileh.walk_nodes(classname = 'Table')]
			num_tables = len(table_list)
			TOC = np.zeros(num_tables, dtype = self.data_set_contents)
			index = 0
			for table in table_list:
				TOC['Run'][index] 						= table.get_attr('Run') 
				TOC['Time_Created'][index] 				= table.get_attr('Time_Created')
				TOC['Num_Temperature_Readings'][index]	= table.get_attr('Num_Temperatures') if table.get_attr('Num_Temperatures') !=None else 0
				#TOC['Num_Ranges'][index] 				= table.get_attr('Num_Ranges') if 'Num_Ranges' in table.attrs._v_attrnames else 1
				TOC['Num_Ranges'][index] 				= table.get_attr('Num_Ranges') if table.get_attr('Num_Ranges') !=None else 0
				TOC['Num_Powers'][index] 				= table.get_attr('Num_Powers') if table.get_attr('Num_Powers') !=None else 0
				TOC['Num_Temperatures'][index] 			= table.get_attr('Num_Heater_Voltages') if table.get_attr('Num_Heater_Voltages') !=None else 0
				TOC['Sensor'][index] 					= table.get_attr('Sensor') if table.get_attr('Sensor') !=None else ''
				TOC['Ground_Plane'][index] 				= table.get_attr('Ground_Plane') if table.get_attr('Ground_Plane') !=None  else ''
				TOC['Path'][index] 						= table._v_pathname
				index += 1 

		self.TOC = TOC
		print(TOC)

	def load_hf5(self, tablepath, filename = database_location):
		''' table path is path to the database to be loaded starting from root. e.g. self.load_hf5('/Run44b/T201312102229')
		filename is the name of the hf5 database to be accessed for the  table informaiton'''

		if not os.path.isfile(filename):
			print('Speficied h5 database does not exist. Aborting...')
			return 
		
		wmode = 'a'
		
		#delete previous metadata object
		del(self.metadata)
		self.metadata = metadata()
		del(self.loop)
		self.loop = loop()

		# use "with" context manage to ensure file is always closed. no need for fileh.close()
		with tables.open_file(filename, mode = wmode) as fileh:
			table = fileh.get_node(tablepath)	
			self.Sweep_Array = table.read()
			for data in self.metadata.__dict__.keys():
				try:
					exec('self.metadata.{0} = table.attrs.{0}'.format(data))
				except:
					print('Table metadata is missing {0}. Setting to None'.format(data))
					exec('self.metadata.{0} = None'.format(data))
		self.sweep_data_columns = self.Sweep_Array.dtype

	def pick_loop(self,index):
		'''Use this function to pick the current loop/transmission data from withing the Sweep_Array. 
		Index is the indes number of sweep/loop to be slected as the current loop.'''
		self.loop.index = index 
		#self.loop.normalization = None
		self.loop.z = self.Sweep_Array[index]['S21']
		self.loop.freq = self.Sweep_Array[index]['Frequencies']
		
	def normalize_loop(self, base = 0, offset = 5):
		''' normalize loop so that mag(S21)< 1. determine normalization by averaging np.abs(S21[base:offset]).mean()
		return normalization'''
		S21 = self.loop.z
		f= self.loop.freq	
		
		normalization = np.abs(S21[base:offset]).mean() # consider using medium()?
		self.loop.normalization = normalization
		S21_normalized = S21/normalization
		self.loop.z = S21_normalized

		return normalization 

	def remove_cable_delay(self, Show_Plot = True, Verbose = True, center_freq = None, Force_Recalculate = False):
		'''
		If self.metadate.Electrical_Delay is not None, then use this value as cable delay and remove from data 

		If self.metadate.Electrical_Delay is None:
		- Determine cable delay by finding delay value, tau, which minimizes distance between adjacent S21 points. 
		Then cancel out tau in S21 data and save corrected loop in self.loop.z. Set self.metadate.Electrical_Delay = tau.
		- If S21 is large array, down sample it first before performing minimization

		If self.metadate.Electrical_Delay is None and center_freq is given:
		-If center_freq is given, this function computes the electrical delay by determining the bandwidth over which the S21 
		circle completes a full revolution starting at center_freq and ending at ~ center_freq + tau^-1. Where tau is approximated
		as the vaule deterined by minimum  distance above. 
		-center_freq should only be used when S21 is is sufficiently broadband to generously cover center_freq and ~center_freq + tau^-1.
		center_freq is in Hertz.

		Return tau in any case.

		If Force_Recalculate == False Electrical delay will be recalculated and reset in metadata
		'''

		S21 = self.loop.z
		f= self.loop.freq

		j = np.complex(0,1)
		n = 1

		if (self.metadata.Electrical_Delay == None) or (Force_Recalculate == True):
			cable_delay_max = 200e-9 # Seconds - guess as to maximum value of cable delay
			cable_delay_guess  = 80e-9 # Seconds
			freq_spacing = np.abs(f[0] - f[1])
			# phase change between adjacent frequency points is 360 * tau * freq_spacing --> want tau * freq_spacing < 1 to see loop
			if (3*freq_spacing * cable_delay_max < 1) & (f.size > 3200):
				n1 = int(np.floor( 1./ (3*freq_spacing * cable_delay_max) )) #want as least 3 points per circle
				n2 = int(np.floor(f.size/3200))
				n = min(n1,n2)


			def obj(t):
				'''This objective fuction yields the sum of squared distance between adjacent (if n = 1) or n-separated S21 points.
				'''
				
				S21a = np.exp(2*np.pi*j*f[1::n]*t)*S21[1::n] # go in steps of n 
				S21b = np.exp(2*np.pi*j*f[:-1:n]*t)*S21[:-1:n] # go in steps of n 
				diff = S21a-S21b
				return (diff*diff.conjugate()).real.sum()

			

			# # Could use Nelder-Mead
			# out = minimize(obj,cable_delay_guess, method='Nelder-Mead',tol=1e-20,options={'disp':False})
			# cable_delay = out.x[0] # in seconds
			
			out = minimize(obj,cable_delay_guess, method='Powell',tol=1e-20,options={'disp':False, 'ftol':1e-14,'xtol':1e-14})
			cable_delay_min_distance = out.x.item() #in Seconds 
			cable_delay  = cable_delay_min_distance 

			if center_freq is not None:
				cable_delay_bandwidth = 1/cable_delay_min_distance #Hz - Estimate of winding bandwith using tau

				closest_index_to_center_freq = np.where(np.abs(f-center_freq) == np.abs(f-center_freq).min()) 
				s21 = S21*np.exp(np.complex(0,-np.angle(S21[closest_index_to_center_freq]))) #rotate circle so that S21[center_freq] is close to positive x axis, and angle(S21[center_freq]) ~ 0
				
				
				condition = ((center_freq - .30*cable_delay_bandwidth) < f) & (f<center_freq+.30*cable_delay_bandwidth)
				f_lower_band =np.extract(condition,f)
				s21_lower_band = np.extract(condition,s21)
				ang_lower_band = np.extract(condition,np.angle(s21)) #np.angle has range [+pi,-pi]
				interp_lower_band = interp1d(ang_lower_band, f_lower_band,kind='linear')
				lower_x_axis_crossing_freq = interp_lower_band(0).item()

				center_freq = center_freq + cable_delay_bandwidth #shift to upper band
				condition = ((center_freq - .30*cable_delay_bandwidth) < f) & (f<center_freq+.30*cable_delay_bandwidth)
				f_upper_band =np.extract(condition,f)
				s21_upper_band = np.extract(condition,s21)
				ang_upper_band = np.extract(condition,np.angle(s21)) #np.angle has range [+pi,-pi]
				interp_upper_band = interp1d(ang_upper_band, f_upper_band,kind='linear')
				upper_x_axis_crossing_freq = interp_upper_band(0).item()

				winding_bandwidth = upper_x_axis_crossing_freq - lower_x_axis_crossing_freq

				cable_delay_winding = 1/winding_bandwidth
				cable_delay = cable_delay_winding #override cable_delay_min_distance 
		else:
			cable_delay = self.metadata.Electrical_Delay
			center_freq = None

		S21_Corrected = np.exp(2*np.pi*f*j*cable_delay)*S21
		
		if Verbose == True:
			if n>1:
				print('S21 downsampled by factor n = {}.'.format(n))
			if (self.metadata.Electrical_Delay == None) or (Force_Recalculate == True):
				print('cable delay is {} seconds by minimum distance method'.format(cable_delay_min_distance))	
			else: 
				print('cable delay is {} seconds as found in metadata'.format(self.metadata.Electrical_Delay))
			if center_freq is not None:
				print('cable delay is {} seconds by loop winding method'.format(cable_delay_winding))
			
				
		if Show_Plot:
			fig = plt.figure( figsize=(9,6))#, dpi=150)
			ax = {}	
			def plot_loops(ax):
				from matplotlib.ticker import MaxNLocator
				majormaxnlocator    = MaxNLocator(nbins = 5)
				minormaxnlocator    = MaxNLocator(nbins = 5*5)
				#ax2 = fig.add_subplot(111, aspect='equal')
				line2 = ax.plot(S21.real,S21.imag, color='blue', linestyle='solid', linewidth = 3, label = 'Measured') 
				line1 = ax.plot(S21_Corrected.real, S21_Corrected.imag, 'g-',linewidth = 3, label = 'Corrected')
				ax.grid()
				ax.set_title('Resonance Loop', fontsize=9)
				plt.setp(ax.get_xticklabels(),rotation = 45)
				ax.legend(loc = 'best')

			if center_freq is None:
				gs = gridspec.GridSpec(1, 1)
				ax[1] = plt.subplot(gs[0, 0],aspect='equal')
				plot_loops(ax[1])

			else:
				gs = gridspec.GridSpec(2, 3)#,width_ratios=[2,2,1])

				ax[1] = plt.subplot(gs[:,:2],aspect='equal')
				ax[2] = plt.subplot(gs[0, 2])
				ax[3] = plt.subplot(gs[1, 2], aspect='equal' )
				plot_loops(ax[1])
				curve = ax[2].plot(f_lower_band,ang_lower_band, linestyle = '-')
				curve = ax[2].plot(f_upper_band,ang_upper_band, linestyle = '-')
				curve = ax[3].plot(s21_lower_band.real,s21_lower_band.imag, linestyle = '-')
				curve = ax[3].plot(s21_upper_band.real,s21_upper_band.imag, linestyle = '-')
				plt.setp(ax[2].get_xticklabels(),rotation = 45)	
				plt.setp(ax[3].get_xticklabels(),rotation = 45)				


			#fig.subplots_adjust(wspace = 0.6,bottom = 0.09, top = 0.1)
			#plt.setp(fig, tight_layout = True)
			plt.show()

		self.metadata.Electrical_Delay = cable_delay
		self.loop.z = S21_Corrected
		return cable_delay

	def trim_loop(self,N = 20,Verbose = True,):
		import numpy.ma as ma
		f = f1 = ma.array(self.loop.freq)
		z = z1 = ma.array(self.loop.z)
		# estimate resonant freq using resonance dip
		zr_mag_est = np.abs(z).min()
		zr_est_index = np.where(np.abs(z)==zr_mag_est)[0][0]

		# estimate max transmission mag using max valuse of abs(z)
		z_max_mag = np.abs(z).max()

		#Depth of resonance in dB
		depth_est = 20.0*np.log10(zr_mag_est/z_max_mag)

		#Magnitude of resonance dip at half max
		res_half_max_mag = (z_max_mag+zr_mag_est)/2

		#find the indices of the closest points to this magnitude along the loop, one below zr_mag_est and one above zr_mag_est
		a = np.square(np.abs(z[:zr_est_index+1]) - res_half_max_mag)
		lower_index = np.argmin(a)
		a = np.square(np.abs(z[zr_est_index:]) - res_half_max_mag)
		upper_index = np.argmin(a) + zr_est_index

		#estimate the FWHM bandwidth of the resonance
		f_upper_FWHM = f[upper_index]
		f_lower_FWHM = f[lower_index]
		FWHM_est = np.abs(f_upper_FWHM - f_lower_FWHM)
		fr_est = f[zr_est_index]

		#Bandwidth Cut: cut data that is more than N * FWHM_est away from zr_mag_est
		z = z2 = ma.masked_where((f > fr_est + N*FWHM_est) | (fr_est - N*FWHM_est > f),z)
		f = f2 = ma.array(f,mask = z.mask)

		self.loop.z = ma.compressed(z)
		self.loop.freq = ma.compressed(f)

		if Verbose: 
			print('Bandwidth cut:\n\t{1} points outside of fr_est +/- {0}*FWHM_est removed, {2} remaining data points'.format(N, *self._points_removed(z1,z2)))

	def _points_removed(self,initial, final):
		''' Compute and return the number of point removed from inital due to a cut. 
		return this number and the number of points in final'''
		try:
			initial_number = initial.size - initial.mask.sum()
		except:
			initial_number = initial.size

		try: 	
			final_number = final.size - final.mask.sum()
		except:
			final_number = final.size
		return (initial_number - final_number), final_number

	def circle_fit(self, Show_Plot = True):

		S21 = self.loop.z
		Freq = self.loop.freq

		LargeCircle = 10
		def pythag(m,n):
			'''compute pythagorean distance
			   sqrt(m*m + n*n)'''
			return np.sqrt(np.square(m) + np.square(n))

		def eigen2x2(a,b,c):
			'''a,b,c - matrix components 	[[a c]
											 [c d]]	
			   d1,d2 - eigen values where |d1| >= |d2|
			   (Vx,Vy) - unit eigen vector of d1,  Note: (-Vy,Vx) is eigen vector for d2
			'''
			disc = pythag(a-b,2*c) # discriminant
			d1 = max(a+b + disc, a+b - disc)/2
			d2 = (a*b-c*c)/d1

			if np.abs(a-d1) > np.abs(b-d1):
				f = pythag(c,d1-a)
				if f == 0.0:
					Vx = 1.
					Vy = 0.
				else:
					Vx = c/f
					Vy = (d1-a)/f
			else:
				f = pythag(c,d1-b)
				if f == 0.0:
					Vx = 1.
					Vy = 0.
				else:
					Vx = (d1-b)/f
					Vy = c/f					
			return d1,d2,Vx,Vy

		def F(x,y,a,b):
			''' computes and returns the value of the objective fuction.
			do this for the case of a large circle and a small circle  '''
			
			if (np.abs(a) < LargeCircle) and (np.abs(b) < LargeCircle): # Case of Small circle
				xx = x - a
				yy = y - b
				D = pythag(xx,yy)

				r = D.mean()

				return (np.square(D - r)).mean()
			else:	# Case of Large circle
				a0 = a - x.mean()
				b0 = b - y.mean()
				d = 1.0/pythag(a0,b0)
				dd = d*d
				s = b0*d
				c = a0*d

				xx = x - x.mean()
				yy = y - y.mean()
				z = np.square(xx) + np.square(yy)
				p = xx*c + yy*s
				t = d*z - 2.0*p
				g = t/(1+np.sqrt(1.+d*t))
				W = (z+p*g)/(2.0+d*g)
				Z = z

				return Z.mean() - W.mean()*(2.0+d*d*W.mean())

		def GradHessF(x,y,a,b):
			'''Compute gradient of F, GradF = [F1,F2] and Hessian of F, HessF = [[A11 A12]
																				  A12 A22]]
			at point p = [a,b].
			Note Hessian is symmetric. 
			'''
			if (np.abs(a) < LargeCircle) and (np.abs(b) < LargeCircle): # Case of Small circle
				xx = x - a
				yy = y - b
				r = pythag(xx,yy)
				u = xx/r
				v = yy/r

				Mr = r.mean()
				Mu = u.mean()
				Mv = v.mean()
				Muu = (u*u).mean()
				Mvv = (v*v).mean()
				Muv = (u*v).mean()
				Muur = (u*u/r).mean()
				Mvvr = (v*v/r).mean()
				Muvr = (u*v/r).mean()
			


				F1 = a + Mu * Mr - x.mean()
				F2 = b + Mv * Mr - y.mean()

				A11 = 1.0 - Mu * Mu - Mr * Mvvr
				A22 = 1.0 - Mv * Mv - Mr * Muur
				A12 = -1.0 * Mu * Mv + Mr * Muvr

			else:	# Case of Large circle
				a0 = a - x.mean()
				b0 = b - y.mean()
				d = 1.0/pythag(a0,b0)
				dd = d*d
				s = b0*d
				c = a0*d

				xx = x - x.mean()
				yy = y - y.mean()
				z = np.square(xx) + np.square(yy)
				p = xx*c + yy*s
				t = 2.0*p - d*z 
				w = np.sqrt(1.0-d*t)				
				g = -1.0*t/(1.0+w)
				g1 = 1.0/(1.0+d*g)
				gg1 = g*g1
				gg2 = g/(2.0 + d * g)
				aa = (xx+g*c)/w
				bb = (yy+g*s)/w	

				X = (xx*gg1).mean()
				Y = (yy*gg1).mean()
				R = (z+t*gg2).mean()
				T = (t*gg1).mean()
				W = (t*gg1*gg2).mean()	
				AA = (aa*aa*g1).mean()
				BB = (bb*bb*g1).mean()
				AB = (aa*bb*g1).mean()
				AG = (aa*gg1).mean()
				BG = (bb*gg1).mean()
				GG = (g*gg1).mean()	

				U = (T-b*W)*c*0.5 - X + R*c*0.5
				V = (T-b*W)*s*0.5 - Y + R*s*0.5

				F1 = d * ((dd*R*U - d*W*c + T*c)*0.5 - X)
				F2 = d * ((dd*R*V - d*W*s + T*s)*0.5 - Y)

				UUR = ((GG-R*0.5)*c + 2.0*(AG-U))*c + AA
				VVR = ((GG-R*0.5)*s + 2.0*(BG-V))*s + BB
				UVR = ((GG-R*0.5)*c + (AG-U))*s + (BG-V)*c + AB

				A11 = dd*(U*(2.0*c-dd*U) - R*s*s*0.5 - VVR*(1.0+dd*R*0.5))
				A22 = dd*(V*(2.0*s-dd*V) - R*c*c*0.5 - UUR*(1.0+dd*R*0.5))
				A12 = dd*(U*s + V*c + R*s*c*0.5 - dd*U*V + UVR*(1.0 + dd*R*0.5))
			return F1,F2,A11,A22,A12
		
		def sigma(x,y,loop):
			'''estimate of Sigma = square root of RSS divided by N
			gives the root-mean-square error of the geometric circle fit'''
			dx = x-loop.a
			dy = x-loop.b
			loop.sigma = (pythag(dx,dy)-loop.r).mean()
			return loop

		def CircleFitByChernovHoussam(x,y, init, lambda_init):
			import copy
			import sys


			REAL_EPSILON = sys.float_info.epsilon
			REAL_MAX = sys.float_info.max


			IterMAX=200
			check_line= True
			#dmin = 1.0

			ParLimit2 = 100.
			epsilon = 1.e+7*REAL_EPSILON
			factor1 = 32.
			factor2 = 32.
			ccc = 0.4
			factorUp = 10.
			factorDown = 0.1

			new = copy.copy(init)
			#new = sys.modules[__name__].loop() #This is how to access the loop class from inside this function
			#old = loop()

			new.s = F(x,y,init.a,init.b) # compute root mean square error
			F1,F2,A11,A22,A12 = GradHessF(x,y,init.a,init.b) # compute gradient vector and Hessian matrix
			new.Gx = F1
			new.Gy = F2
			new.g = pythag(F1,F2) # The gradient vector and its norm
			lambda_ = lambda_init
			sBest = gBest = progess = REAL_MAX

			enough = False
			i = 0
			ii = 0
			while not enough:
				if i > 0:
					# evaluate the progress made during the previous iteration
					progress = (np.abs(new.a - old.a)+np.abs(new.b - old.b))/(np.square(old.a) + np.square(old.b) + 1.0)
				old = copy.copy(new)

				i = i+1
				if i > IterMAX: #termination due to going over the limit
					enough = True
					break
				d1,d2,Vx,Vy = eigen2x2(A11,A22,A12) #eigendecomposition of the Hessian matrix
				dmin = min(d1,d2) #recording the smaller e-value
				AB = pythag(old.a,old.b) + 1.0 # approximation to the circle size
				# main stopping rule: terminate iterations if 
				# the gradient vector is small enough and the 
				# progress is not substantial 
				if (old.g < factor1*REAL_EPSILON) and (progress<epsilon):
					#print('primary stopping rule')
					enough = True
					break
				# secondary stopping rule (prevents some stupid cycling)
				if (old.s >= sBest) and (old.g >= gBest):
					print(old.s, sBest, old.g, gBest)
					#print('secondary stopping rule')
					enough = True
					break

				if (sBest > old.s):
					sBest = old.s  # updating the smallest value of the objective function found so far
				if (gBest > old.g): 
					gBest = old.g  # updating the smallest length of the gradient vector found so far

				G1 = Vx*F1 + Vy*F2  # rotating the gradient vector
				G2 = Vx*F2 - Vy*F1  # (expressing it in the eigensystem of the Hessian matrix)

				while not enough: # starting point of an "inner" iteration (adjusting lambda)
					# enforcing a lower bound on lambda that guarantees that
					# (i)  the augmented Hessian matrix is positive definite
					# (ii) the step is not too big (does not exceed a certain 
					# fraction of the circle size) the fraction is defined by 
					# the factor "ccc")
					if lambda_ < (np.abs(G1)/AB/ccc) - d1:
						lambda_ = np.abs(G1)/AB/ccc - d1
					if lambda_ < (np.abs(G2)/AB/ccc) - d2: 
						lambda_ = np.abs(G2)/AB/ccc - d2

					# compute the step (dX,dY) by using the current value of lambda
					dX = old.Gx*(Vx*Vx/(d1+lambda_)+Vy*Vy/(d2+lambda_)) + old.Gy*Vx*Vy*(1.0/(d1+lambda_)-1.0/(d2+lambda_))
					dY = old.Gx*Vx*Vy*(1.0/(d1+lambda_)-1.0/(d2+lambda_)) + old.Gy*(Vx*Vx/(d2+lambda_)+Vy*Vy/(d1+lambda_))

					# updating the loop parameter
					new.a = old.a - dX
					new.b = old.b - dY

					if (new.a==old.a) and (new.b==old.b): #if no change, terminate iterations
						enough  = True
						break

					#check if the circle is very large
					if np.abs(new.a)>ParLimit2 or np.abs(new.b)>ParLimit2:
						#when the circle is very large for the first time, check if 
						#the best fitting line gives the best fit

						if check_line:   # initially, check_line= True, then it is set to zero

							#compute scatter matrix
							dx = x - x.mean()
							dy = y - y.mean()
							Mxx = (dx*dx).sum()
							Myy = (dy*dy).sum()
							Mxy = (dy*dx).sum()
							dL1,dL2,VLx,VLy = eigen2x2(Mxx,Myy,Mxy)  # eigendecomposition of scatter matrix

							#compute the third mixed moment (after rotation of coordinates)
							dx = (x - x.mean())*VLx + (y - y.mean())*VLy
							dy = (y - y.mean())*VLx - (x - x.mean())*VLy
							Mxxy = (dx*dx*dy).sum()

							#rough estimate of the center to be used later to recoved from the wrong valley
							if Mxxy > 0.0:
								R = ParLimit2
							else:
								R = -ParLimit2

							aL = -VLy*R
							bL =  VLx*R                 
							check_line = False

						# check if the circle is in the wrong valley
						if (new.a*VLy - new.b*VLx)*R>0.0: 
							# switch to the rough circle estimate (precomupted earlier)
							new.a = aL;                 
							new.b = bL;                 
							new.s = F(x,y,new.a,new.b)    # compute the root-mean-square error
							
							# compute the gradient vector and Hessian matrix
							F1,F2,A11,A22,A12 = GradHessF(x,y,new.a,new.b)  

							# the gradient vector and its norm 
							new.Gx = F1;  
							new.Gy = F2;   
							new.g = pythag(F1,F2)  
							lambda_ = lambda_init     #reset lambda
							sBest = gBest = REAL_MAX  #reset best circle characteristics 
							break
					
					# compute the root-mean-square error
					new.s = F(x,y,new.a,new.b) 
					# compute the gradient vector and Hessian matrix
					F1,F2,A11,A22,A12 = GradHessF(x,y,new.a,new.b)

					# the gradient vector and its norm  
					new.Gx = F1  
					new.Gy = F2   
					new.g = pythag(F1,F2) 

					# check if improvement is gained
					if new.s < sBest*(1.0+factor2*REAL_EPSILON):  #yes, improvement
						lambda_ *= factorDown     # reduce lambda
						break 
					else:
						ii += 1
						if ii > IterMAX: #termination due to going over the limit
							enough = True
							break
						lambda_ *= factorUp #increace lambda
						continue
			

			old.r = pythag(x - old.a, y - old.b).mean() 
			old.outer_iterations = i
			old.inner_iterations = ii
			loop = old
			exit_code = 0
			if old.outer_iterations  > IterMAX:
				exit_code  = 1

			if old.inner_iterations  > IterMAX:
				exit_code = 2

			if (dmin <= 0.0) and (exit_code==0):
				exit_code  = 3

			loop.circle_fit_exit_code = exit_code
			loop = sigma(x,y,loop)

			return loop
		



		x = S21.real
		y = S21.imag


		self.loop.a =  0#guess.real#0
		self.loop.b =  0#guess.imag #0
		lambda_init = 0.001
		#self.loop = CircleFitByChernovHoussam(x,y, self.loop, lambda_init)
		if True: #self.loop.circle_fit_exit_code != 0:
			#print('Circle Fit Failed! Trying again...')
			#another initial guess
			norm = np.abs(S21[1:5].mean())
			S21 = S21/norm
			guess = np.mean(S21)
			self.loop.a =  guess.real#0
			self.loop.b =  guess.imag #0
			lambda_init = 0.001
			x = S21.real
			y = S21.imag
			self.loop = CircleFitByChernovHoussam(x,y, self.loop, lambda_init)
			self.loop.a = self.loop.a*norm
			self.loop.b = self.loop.b*norm
			self.loop.r = self.loop.r*norm
			self.loop.z = S21*norm

			if self.loop.circle_fit_exit_code != 0:
				print('!!!!!!!!!!!!!!    Circle Fit Failed Again! Giving Up...')

		if Show_Plot:
			fig, ax = self.plot_loop(show = False)[:2]		
			t = np.linspace(0, 2.0*np.pi, num=50, endpoint=True)
			j = np.complex(0,1); zc = self.loop.a + j*self.loop.b;  r = self.loop.r
			line = ax.plot(zc.real + r*np.cos(t),zc.imag + r*np.sin(t),'y-', label = 'Circle Fit')
			line = ax.plot([zc.real],[zc.imag],'yx', markersize = 10, markeredgewidth = 4, label = 'Center')
			ax.set_aspect('equal')
			plt.show()	

	def phase_fit(self, Fit_Method = 'Multiple', Verbose = True, Show_Plot = True):
		'''
		Note: its best to determine angles and angle differences by starting with complex numbers 
		(interpreted as vectors) and then finding their angles with, np.angle or self._angle. It is
		not as accurate and prone to issues with domains (e.g. [-180,180]) to use arcsin or arccos.
		'''
		from scipy.stats import chisquare
		
		if isinstance(Fit_Method,str): #Allow for single string input for Fit_Method
		   Fit_Method={Fit_Method}
		   
		
		j = np.complex(0,1)
		try:
			zc = self.loop.a + j*self.loop.b
			r = self.loop.r
		except:
			print('Phase fit needs loop center and radius, which are not currently defined. Aborting phase fit.')
			return



		f = f0 = self.loop.freq
		z = z0 = self.loop.z
		

		# Remove duplicate frequency elements in z and f, e.g. places where f[n] = f[n+1]
		f_adjacent_distance =  np.hstack((np.abs(f[:-1]-f[1:]), [0.0]))
		z = z1 = ma.masked_where(f_adjacent_distance==0.0, z)
		f = f1 = ma.array(f,mask = z.mask) #Syncronize mask of f to match mask of z
		


		#Estimate Resonance frequency using minimum Dip or max adjacent distance
		Use_Dip = 1 
		if Use_Dip: #better for non-linear resonances with point near loop center
			zr_mag_est = np.abs(z).min()
			zr_est_index = np.where(np.abs(z)==zr_mag_est)[0][0]
		else:
			z_adjacent_distance = np.abs(z[:-1]-z[1:])
			zr_est_index = np.argmax(z_adjacent_distance) 
			zr_mag_est = np.abs(z[zr_est_index])


		#Transmission magnitude off resonance 
		Use_Fit = 1
		if Use_Fit:
			z_max_mag = np.abs(zc)+r
		else: #suspected to be better for non-linear resonances
			z_max_mag = np.abs(z).max()

		#Depth of resonance in dB
		depth_est = 20.0*np.log10(zr_mag_est/z_max_mag)

		#Magnitude of resonance dip at half max
		res_half_max_mag = (z_max_mag+zr_mag_est)/2

		#find the indices of the closest points to this magnitude along the loop, one below zr_mag_est and one above zr_mag_est
		a = np.square(np.abs(z[:zr_est_index+1]) - res_half_max_mag)
		lower_index = np.argmin(a)#np.where(a == a.min())[0][0]
		a = np.square(np.abs(z[zr_est_index:]) - res_half_max_mag)
		upper_index = np.argmin(a) + zr_est_index

		#estimate the FWHM bandwidth of the resonance
		f_upper_FWHM = f[upper_index]
		f_lower_FWHM = f[lower_index]
		FWHM_est = np.abs(f_upper_FWHM - f_lower_FWHM)
		fr_est = f[zr_est_index]

		
		#consider refitting the circle here, or doing ellipse fit.



		#translate circle to origin, and rotate so that z[zr_est_index] has angle 0 
		z = z2 = ma.array((z.data-zc)*np.exp(-j*(self._angle(zc))), mask = z.mask)

		#Compute theta_est before radious cut to prevent radius cut from removing z[f==fr_est]
		theta_est = self._angle(z[zr_est_index]) #self._angle(z[zr_est_index])	

		#Radius Cut: remove points that occur within r_cutoff of the origin of the centered data. 
		#(For non-linear resonances that have spurious point close to loop center)	
		r_fraction_in = 0.75
		r_fraction_out = 1.75
		r_cutoff_in  = r_fraction_in*r
		r_cutoff_out = r_fraction_out*r		
		z = z3 = ma.masked_where((np.abs(z2)<r_cutoff_in) | (np.abs(z2)>r_cutoff_out),z2, copy = True)
		# for substantially deformed loops we make sure that no more than Max_Removed_Radius_Cut points are removed from inner radious cut
		Max_Removed_Radius_Cut = 25
		while self._points_removed(z2, z3)[0] > Max_Removed_Radius_Cut:
			r_fraction_in = r_fraction_in - 0.02
			r_cutoff_in  = r_fraction_in*r
			z = z3 = ma.masked_where((np.abs(z2)<r_cutoff_in) | (np.abs(z2)>r_cutoff_out),z2, copy = True)
			print 'loosening inner radius cut: r_fraction_in = {}'.format(r_fraction_in)
			if r_fraction_in <= 0:
				break
		f = f3 = ma.array(f,mask = z.mask)


		#Bandwidth Cut: cut data that is more than N * FWHM_est away from zr_mag_est
		N = 8
		z = z4 = ma.masked_where((f > fr_est + N*FWHM_est) | (fr_est - N*FWHM_est > f),z,copy = True)
		f = f4 = ma.array(f,mask = z.mask)
		z_theta,z_theta_offset =self._angle(z, return_offset = True) # dont used self._angle(z)!


		#Angle jump cut : masks points where angle jumps to next branch of angle function, 
		mask = (f > fr_est + 0.5*FWHM_est) | (f < fr_est + -0.5*FWHM_est)
		f_in_FWHM = ma.masked_where(mask,f) # or alternatively: f_in_FWHM = f; f_in_FWHM[mask] = ma.masked 
		edge1,edge2 = ma.flatnotmasked_edges(f_in_FWHM)
		angle_slope = (z_theta[edge2]-z_theta[edge1])/(f[edge2]-f[edge1]) # angle is decreasing if negative slope
		upper_cond = ((f > fr_est +  0.5*FWHM_est) & ((z_theta[edge2]<z_theta) if (angle_slope<0) else (z_theta[edge2]>z_theta))) 
		lower_cond = ((f < fr_est + -0.5*FWHM_est) & ((z_theta[edge1]>z_theta) if (angle_slope<0) else (z_theta[edge1]<z_theta))) 
		z = z5 = ma.masked_where(lower_cond|upper_cond,z, copy = True)
		f = f5 = ma.array(f,mask = z.mask)
		z_theta = z_theta5 = ma.array(z_theta,mask = z.mask)
		


		#theta_est = np.extract(f==fr_est,z_theta)[0] # The old lication of theta_est computation 
		Q_est = fr_est/FWHM_est


		#consider reducing computation by extracting only the unmasked values of z,f, and z_theta of the minimization
		#These commands return a masked array where all the masked elements are removed.
		#z = z[~z.mask]
		#f = f[~f.mask]
		#z_theta = z_theta[~z_theta.mask]

		#These commands return np array
		z_c = ma.compressed(z)
		f_c = ma.compressed(f)
		z_theta_c  = ma.compressed(z_theta)
		

		if mysys.startswith('Windows'):
			dt = np.float64
		else:	
			dt = np.float128

		def hess(x, z_theta,f): #to avoid overflow try to re write hessian so that all numbers are of order 1
			theta,fr,Q = x	
			H = np.zeros((3,3), dtype = dt)
			ff = (1-(f/fr))
			denom = (1+4.0*np.square(ff*Q))
			numer = (theta+z_theta-2.0*np.arctan(2.0*ff*Q))
			H[0,0] = (2.0*np.ones_like(z_theta)).sum()
			H[0,1] = ((-8.0*f*Q)/(np.square(fr)*denom)).sum()
			H[0,2] = ((8.0*ff)/denom).sum()
			H[1,0] = H[0,1] #((8.0*f*Q)/(np.square(fr)*denom)).sum()
			H[1,1] = ((32.0*np.square(f*Q/(np.square(fr)*denom)))  +   (64.0*np.square(f/(np.square(fr)*denom))*ff*np.power(Q,3)*numer)   +  ((16.0*f*Q/np.power(fr,3))*(numer/denom))).sum()
			H[1,2] = (((32.0*f*Q*ff)/np.square(fr*denom))  +  ((64.0*f*np.square(ff*Q)*numer)/(np.square(fr*denom)))  - ((8.0*f*numer)/(np.square(fr)*denom))).sum()
			H[2,0] = H[0,2] #((8.0*ff)/denom).sum()
			H[2,1] = H[1,2] #(((32.0*f*ff*Q)/np.square(fr*denom))  +  ((64.0*f*np.square(ff*Q)*numer)/(np.square(fr*denom)))  -  ((8.0*f*numer)/(np.square(fr)*denom))).sum()
			H[2,2] = (((32.0*np.square(ff))/np.square(denom))  +  ((64.0*np.power(ff,3)*Q*numer)/np.square(denom))).sum()				
			return H

		def jac(x,z_theta,f):
			theta,fr,Q = x
			J = np.zeros((3,),dtype = dt)    #np.zeros_like(x)
			ff = (1-(f/fr))
			denom = (1+4.0*np.square(ff*Q))
			numer = (theta+z_theta-2.0*np.arctan(2.0*ff*Q))	
			J[0] = np.sum(2.0*numer)
			J[1] = np.sum(-8.0*f*Q*numer/(np.square(fr)*denom))
			J[2] = np.sum(-8.0*ff*numer/denom)
			return J


		def obj(x,z_theta,f):
			theta,fr,Q = x
			return np.square(z_theta + theta - 2.0*np.arctan(2.0*Q*(1-f/fr))).sum()	 #<--- Need hessian of this


		def obj_ls(x,z_theta,f):
			'''object fuctinon for least squares fit'''
			theta,fr,Q = x
			residual  = z_theta + theta - 2.0*np.arctan(2.0*Q*(1-f/fr))	
			return residual

		#p0 is the initial guess
		p0 = np.array([theta_est,fr_est ,Q_est])
		
		#Each fit method is saved as a lambda function in a dictionary called fit_func
		fit_func = {}
		fit_func['Powell'] = lambda : minimize(obj, p0, args=(z_theta_c,f_c), method='Powell', jac=None, hess=None, hessp=None, bounds=None, constraints=(), tol=1e-20, callback=None, options={'disp':False, 'maxiter': 70, 'maxfev': 50000, 'ftol':1e-20,'xtol':1e-20})#options={'disp':False})
		fit_func['Nelder-Mead']  = lambda : minimize(obj, p0, args=(z_theta_c,f_c), method='Nelder-Mead', jac=None, hess=None, hessp=None, bounds=None, constraints=(), tol=1e-18, callback=None, options={'disp':False, 'xtol' : 1e-6,'maxfev':1000})
		fit_func['Newton-CG'] = lambda : minimize(obj, p0, args=(z_theta_c,f_c), method='Newton-CG', jac=jac, hess=hess, hessp=None, bounds=None, constraints=(),tol=1e-18, callback=None, options={'maxiter' : 50,'xtol': 1e-4,'disp':False})

		fit = {}
		if isinstance(Fit_Method,set):      #All string inputs for Fit_Method were changed to sets at the begining of phase_fit
		   if Fit_Method == {'Multiple'}:
		      for method in fit_func.keys():
		         fit[method] = fit_func[method]() # Execute the fit lambda function
		   else:
		      for method in Fit_Method:
		         if method not in fit_func.keys():
		            print("Unrecognized fit method. Aborting fit. \n\t Must choose one of {0} or 'Multiple'".format(fit_func.keys()))
		            return
		         else:   
		            fit[method] = fit_func[method]()
		else:
		   print("Unrecognized fit method data type. Aborting fit. \n\t Please specify using a string or a set of strings from one of {0} or 'Multiple'".format(fit_func.keys()))
		   return	         	   
		               				
		
		#Does not work if the objective function is re-arranged as in the following
		# print('Nelder-Mead 2 ################# ')
		# def obj(x,z_theta,f):
		# 	theta,fr,Q = x
		# 	return np.square(np.tan((z_theta - theta)/2) - (2.0*Q*(1-f/fr))).sum()
		# res = minimize(obj, p0, args=(z_theta,f), method='Nelder-Mead', jac=None, hess=None, hessp=None, bounds=None, constraints=(), tol=1e-20, callback=None, options={'disp':True})
		# print(res)
	
		# Least square method does not find a good Q fit and the sum of the squares for solution is fairly high
		# print('Least Square ################# ')
		# print(fit['Least-Squares'])
		# print(np.square(fit['Least-Squares'][2]['fvec']).sum()) # this is the value of the sum of the squares for the solution
		# x = fit['Least-Squares'][0] 
		
		#x = res.x 
		bestfit = list(fit)[0]
		lowest = fit[bestfit].fun
		for key in fit.keys(): 
			if fit[key].fun < lowest:
				lowest = fit[key].fun
				bestfit = key
		

		theta0 = 2*np.pi - self._angle(np.exp(np.complex(0,fit[bestfit].x[0] - z_theta_offset)).conj())
		zc_m = np.abs(zc)
		R = np.sqrt(zc_m*zc_m + r*r -2.0*zc_m*r*np.cos(theta0) ) # Used in Qc

		alpha = self._angle(zc)#np.angle(zc)#
		z_pivot = zc + (np.complex(-r*np.cos(theta0), r*np.sin(theta0)))*np.complex(np.cos(alpha),np.sin(alpha))# vector for origin to pizot point
		theta = self._angle(z_pivot)
		phi  = np.angle(-(zc-z_pivot)*np.exp(-j*(self._angle(z_pivot)))) #not that domain is [-180, +180]

		self.loop.R = R
		self.loop.phase_fit_success = fit[bestfit].success
		self.loop.phase_fit_z = z5.data
		self.loop.phase_fit_mask = z5.mask
		self.loop.phase_fit_method = bestfit
		self.loop.Q = Q = fit[bestfit].x[2]
		self.loop.Qc = Qc = Q*R/(2*r)
		self.loop.Qi = Q*Qc/(Qc-Q)
		self.loop.fr = fr = fit[bestfit].x[1]
		self.loop.FWHM = fr/Q
		self.loop.phi = phi # radian
		self.loop.theta = theta # radian
		self.loop.chisquare, self.loop.pvalue = chisquare( z_theta_c,f_exp=fit[bestfit].x[0] + 2.0*np.arctan(2.0*Q*(1-f_c/fr)))
		self.loop.chisquare = self.loop.chisquare/ f_c.shape[0]
		#estimated quantities from MAG S21 
		self.loop.fr_est = fr_est
		self.loop.FWHM_est = FWHM_est
		self.loop.depth_est = depth_est
		self.loop.Q_est = Q_est
		


		# print 'phi + theta =  {0} deg'.format((phi+theta)*180/np.pi)
		# # abs_phi = np.arcsin(np.angle(z_pivot/zc)*(np.abs(zc)/r))
		# # #if theta > alpha:

		# # print 'theta is {} '.format(theta*180/np.pi)
		# # print 'phi is {}'.format( abs_phi*180/np.pi)
		# # #when -r*np.sin(theta0) is negative, phi is positive
		# # np.angle(-(zc-z_pivot)*np.exp(-j*(self._angle(z_pivot))))
		# # print 'phi is {}'.format(np.angle(-(zc-z_pivot)*np.exp(-j*(self._angle(z_pivot))))*180/np.pi)
		


		# alpha = self._angle(zc)#np.angle(zc)#
		# theta_f = -1.*(fit[bestfit].x[0] - z_theta_offset) # minus becasue of how "theta" is the objective function obj()
		# phi_theta = theta_f + alpha + np.pi #np.fmod(theta_f + alpha, np.pi) # return angle in th domain [+pi,-pi]
		# print 'phi + theta =  {0} deg, and alpha  = ang(zc) = {1} deg, theta_f is {2}, ztheta offset is {3} '.format((phi_theta)*180/np.pi,alpha*180/np.pi , theta_f*180/np.pi, z_theta_offset*180/np.pi)
		
		# def rectify_angle(ang, offset,alpha):
		# 	''' output correct angle in domain [-pi, pi]
		# 	'''

		# 	s = np.sign(offset)
		# 	if (alpha < np.pi/2. ) | (alpha > 3*np.pi/2.):
		# 		if s > 0:
		# 			return np.mod(ang, np.pi)
		# 		else:
		# 			return np.mod(ang, np.pi) - np.pi
		# 	else:
		# 		if s > 0:
		# 			return np.mod(ang, np.pi) - np.pi
		# 		else:
		# 			return np.mod(ang, np.pi) 

		# print 'guess algorith:  phi + theta =  {0} deg'.format(rectify_angle(phi_theta, z_theta_offset, alpha)*180/np.pi)
		# print 'theta is {0} deg, and zc is {1} deg, offset is {2}'.format((fit[bestfit].x[0] -z_theta_offset )*180/np.pi,self._angle(zc)*180/np.pi , z_theta_offset*180/np.pi)
		# print 'phi is {}'.format((180/np.pi)*phi)


		#self.loop.phi = rectify_angle(phi, z_theta_offset, alpha)
		#self.loop.theta = rectify_angle(theta, z_theta_offset, alpha) #theta#(self._angle(zc)-(fit[bestfit].x[0] - z_theta_offset )- 1*np.pi)

		# zc_m = np.abs(zc)
		# leg = np.sqrt(zc_m*zc_m + r*r -2.0*zc_m*r*np.cos(rectify_angle(theta_f, z_theta_offset, alpha)) )
		# phi = np.arcsin(zc_m*np.sin(theta_f)/leg)
		# theta =  rectify_angle(phi_theta, z_theta_offset, alpha)- phi
		# print  'theta =  {0} deg, and phi  = {1} deg'.format((theta)*180/np.pi,phi*180/np.pi)

		# def recify_offset(ang):
		# 	if ang<0:
		# 		ang = 2.*np.pi - ang
		# 	return ang
		# alpha = self._angle(zc)#np.angle(zc)#
		# theta_f = -1.*(fit[bestfit].x[0] + recify_offset(z_theta_offset)) # minus becasue of how "theta" is the objective function obj()
		# phi_theta = theta_f + alpha + np.pi #np.fmod(theta_f + alpha, np.pi) # return angle in th domain [+pi,-pi]
		# print 'phi + theta =  {0} deg, and alpha  = ang(zc) = {1} deg, theta_f is {2}, ztheta offset is {3} '.format((phi_theta)*180/np.pi,alpha*180/np.pi , theta_f*180/np.pi, z_theta_offset*180/np.pi)
		# print 'guess algorith:  phi + theta =  {0} deg'.format(rectify_angle(phi_theta, z_theta_offset, alpha)*180/np.pi)

		if Verbose: 
			print('Duplicates cuts:\n\t{0} duplicate frequencies removed from loop data, {1} remaining data points'.format(*self._points_removed(z0,z1)))
			print('Radius cut:\n\t{2} points < r_loop*{0} or > r_loop*{1} found and removed, {3} remaining data points'.format(r_fraction_in, r_fraction_out,*self._points_removed(z2,z3)))
			print('Bandwidth cut:\n\t{1} points outside of fr_est +/- {0}*FWHM_est removed, {2} remaining data points'.format(N, *self._points_removed(z3,z4)))
			print('Angle jump cut:\n\t{0} points with discontinuous jumps in loop angle removed, {1} remaining data points'.format(*self._points_removed(z4,z5)))
			print('Initial Guess:\n\tLoop rotation {0} deg, fr {1}, Q {2}'.format(p0[0]*180/np.pi,p0[1],p0[2] ))

			for method in fit.keys():
				print('\n{0} Minimzation Result:\n{1}\n'.format(method,fit[method]))



		if Show_Plot:
			total_removed, total_used_in_fit = self._points_removed(z0,z5)
			fig1 = plt.figure( facecolor = 'w',figsize = (10,10))
			ax = fig1.add_subplot(6,1,1)
			ax.set_title('Number of points used in fit = '+str(total_used_in_fit)+', Number of points removed = ' + str(total_removed) )
			#line = ax.plot(f1[~f5.mask], np.abs(z1[~z5.mask]),'g-', label = 'Used for Fit') #fails when no points are masked
			


			if f5.mask.size <= 1:#this is the case that there are no masked points, e.g. no mask. there will allways be 1 point in the mask due to adjacent distance
				line = ax.plot(ma.compressed(f1), np.abs(ma.compressed(z1)),'g-', label = 'Used for Fit')
			else:
				line = ax.plot(f1[~f5.mask], np.abs(z1[~z5.mask]),'g-', label = 'Used for Fit')
				line = ax.plot(f1[f5.mask], np.abs(z1[z5.mask]),'r.',markersize = 2,  alpha = 0.2, label = 'Excluded Data')
			line = ax.plot([f1[zr_est_index],f1[zr_est_index]] , [np.abs(z1[zr_est_index]),np.abs(zc)+r] ,'k.', label = 'Magitude Min and Max')
			line = ax.plot([f1[lower_index], f1[upper_index], f1[upper_index]], np.abs([z1[lower_index],z1[lower_index],z1[upper_index]]),'yo-', label = 'FWHM Estimate')
			ax.set_ylabel('Magnitude')
			## Find index of closet freq point to Fr
			a = np.square(np.abs(f1 - fr))
			fr_index = np.argmin(a)
			line = ax.plot(f1[fr_index], np.abs(z1[fr_index]),'gx', markersize = 7, markeredgewidth = 4, label = 'Fr (closest)')# this is the closest point in  the cut z1 to the true fr 
			ax.legend(loc = 'best', fontsize=10,scatterpoints =1, numpoints = 1, labelspacing = .1)
			
			
			ax = fig1.add_subplot(6,1,(2,4), aspect='equal')
			t = np.linspace(0, 2.0*np.pi, num=50, endpoint=True)
			line = ax.plot([0,zc.real],[0, zc.imag],'y*-', label = 'Center Vector')	
			line = ax.plot(zc.real + r*np.cos(t),zc.imag + r*np.sin(t),'y-', label = 'Circle Fit')		
			line = ax.plot(z1.real, z1.imag,'r:', label = 'Initial Location')
			line = ax.plot(z3.real, z3.imag,'r-', label = 'Aligned w/ Origin')
			lint = ax.plot([0,z_pivot.real],[0,z_pivot.imag],'yo-', label = 'Pivot point')
			lint = ax.plot([zc.real,z_pivot.real],[zc.imag,z_pivot.imag],'yo-', label = '_zc_to_zp')#zp is zpivot
			## Find index of closet freq point to Fr
			a = np.square(np.abs(f_c - fr))
			fr_index = np.argmin(a)

			line = ax.plot(z_c[fr_index].real, z_c[fr_index].imag,'gx', markersize = 7, markeredgewidth = 4, label = 'Fr (closest)')
			line = ax.plot([0,r*np.cos(theta0)],[0,-r*np.sin(theta0)], 'b',  label = 'Fr (True)') #vector to fr
			
			line = ax.plot(z4.real, z4.imag,'g:', linewidth = 3,label = 'Bandwidth Cut')
			##pt = ax.plot([z1[0].real,z[~z.mask][0].real], [z1[0].imag,z[~z.mask][0].imag],'ko', label = 'First Point') fails when no points are masked
			pt = ax.plot([z1[0].real,ma.compressed(z5)[0].real], [z1[0].imag,ma.compressed(z5)[0].imag],'ko', label = 'First Point') #--
			pt = ax.plot(z2[zr_est_index].real, z2[zr_est_index].imag,'k*', label = 'Magnitude Min')

			#line = ax.plot(z4[z4.mask].data.real, z4[z4.mask].data.imag,'r.', alpha = 0.2, label = 'Excluded Data')
			line = ax.plot(z5[ma.getmaskarray(z5)].data.real, z5[ma.getmaskarray(z5)].data.imag,'r.', alpha = 0.2,label = 'Excluded Data')
			ax.legend(loc = 'center left', bbox_to_anchor=(1.01, 0.5), fontsize=10, scatterpoints =1, numpoints = 1, labelspacing = .1)#,numpoints)
			
			text = ('$*Resonator Properties*$\n' + '$Q =$ ' + '{0:.2f}'.format(self.loop.Q) +'\nf$_0$ = ' + '{0:.6f}'.format(self.loop.fr/1e6) 
				+  ' MHz\n$Q_c$ = ' + '{0:.2f}'.format(self.loop.Qc) + '\n$Q_i$ = ' + '{0:.2f}'.format(self.loop.Qi) + '\n|S$_{21}$|$_{min}$ = ' 
				+ '{0:.3f}'.format(self.loop.depth_est) + ' dB' + '\nBW$_{FWHM}$ = ' + '{0:.3f}'.format(self.loop.FWHM/1e3) +  ' kHz' 
				+ '\n$\chi^{2}$ = ' + '{0:.4f}'.format(self.loop.chisquare) + '\n$\phi$ = ' + '{0:.3f}'.format(self.loop.phi*180/np.pi) +' deg' + '\n' + r'$\theta$ = ' 
				+ '{0:.3f}'.format(self.loop.theta*180/np.pi) +' deg' +'\n$- $'+self.loop.phase_fit_method 
				+ ' fit $-$') 
			bbox_args = dict(boxstyle="round", fc="0.8")        
			fig1.text(0.10,0.7,text,
					ha="center", va="top", visible = True,
					bbox=bbox_args, backgroundcolor = 'w')
			# ax.text(0.01, 0.01, text,
			# 	verticalalignment='bottom', horizontalalignment='left',
			# 	transform=ax.transAxes,
			# 	color='black', fontsize=4)


			ax = fig1.add_subplot(6,1,5)
			hline = ax.axhline(y = fit[bestfit].x[0],linewidth=2, color='y', linestyle = '-.',   label = r'$\theta_{r}$')
			vline = ax.axvline(x = fit[bestfit].x[1],linewidth=2, color='y', linestyle = ':',   label = r'$f_{r}$')
			line = ax.plot(f,z_theta,'g-',linewidth = 3,label = 'Data')
			line = ax.plot(f,(-fit[bestfit].x[0] + 2.0*np.arctan(2.0*fit[bestfit].x[2]*(1-f/fit[bestfit].x[1]))),'g:', linewidth = 1, label = 'Fit ')
			#line = ax.plot(f5[~f5.mask][0],z_theta5[~z_theta5.mask][0],'ko',linewidth = 3,label = 'First Point') #Failes when  no points are masked
			line = ax.plot(ma.compressed(f5)[0],ma.compressed(z_theta5)[0],'ko',linewidth = 3,label = 'First Point')

			ax.set_ylabel('Angle [rad]')
			ax.legend(loc = 'right', fontsize=10,scatterpoints =1, numpoints = 1, labelspacing = .1)
			
			ax = fig1.add_subplot(6,1,6)
			vline = ax.axvline(x = fit[bestfit].x[1],linewidth=2, color='y', linestyle = ':',   label = r'$f_{r}$')
			style  = ['-','--',':','-.','+','x']; s = 0 #Cyclic iterable?
			for key in fit.keys():
				line = ax.plot(f,(z_theta - fit[key].x[0] - 2.0*np.arctan(2.0*fit[key].x[2]*(1-f/fit[key].x[1]))),'b'+style[s], linewidth = 3, label = 'Data - Fit ' + key)
				s += 1
			ax.set_ylabel('Angle [rad]')
			ax.set_xlabel('Freq [Hz]')
			ax.legend(loc = 'right', fontsize=10,scatterpoints =1, numpoints = 1, labelspacing = .1)
			plt.show()

			# fig = plt.figure( figsize=(5, 5), dpi=150)
			# ax = {}
			# ax[1] = fig.add_subplot(1,1,1)
			# #dff = (f5 - fr)/fr
			# dff = f5 
			# curve = ax[1].plot(dff,np.abs(z5))
			# ax[1].ticklabel_format(axis='x', style='sci',scilimits = (0,0), useOffset=True)	

			# for k in ax.keys():
			# 	ax[k].tick_params(axis='y', labelsize=9)
			# 	ax[k].tick_params(axis='x', labelsize=5)
			# plt.show()

	def fill_sweep_array(self, Fit_Resonances = True, Compute_Preadout = False, Add_Temperatures = False, Complete_Fit = True, Remove_Gain_Compression = True ):
		

		if Compute_Preadout == True:
			needed = ('Atten_NA_Output', 'Atten_At_4K','Cable_Calibration')

			for quantities  in needed:				
				if  self.metadata.__dict__[quantities] == None:
					print('{0} metadate missing. Unable to compute Preadout. Setting to 0.'.format(quantities))
					Compute_Preadout = False

				Atten_NA_Output = self.metadata.Atten_NA_Output
				Atten_At_4K = self.metadata.Atten_At_4K
				Cable_Calibration_Key = 'One_Way_40mK'
				k = self.metadata.Cable_Calibration[Cable_Calibration_Key]

			if Fit_Resonances == False:
				print('Resonance fit not selected. Computation of Preadout_dB requires knowledge of resonance frequency and may not work.')


			if Compute_Preadout == True:
				Preadout = lambda f: k[0]*np.sqrt(f)+k[1]*f+k[2] - Atten_NA_Output - Atten_At_4K

		if Add_Temperatures == True:
			if self.metadata.Num_Temperatures < 1:
				Temperature_Calibration = self.metadata.Temperature_Calibration
				if (self.metadata.Fridge_Base_Temp != None) & (max(self.Sweep_Array['Heater_Voltage']) == min(self.Sweep_Array['Heater_Voltage'])): #& (self.Sweep_Array.size == 1):
					#This is usually the case of a survey or power sweep: done at base temp with no Heater power
					self.Sweep_Array['Temperature'][:] = self.metadata.Fridge_Base_Temp
					print('Setting Tempreature to metadata.Fridge_Base_Temp value.')
					Add_Temperatures = False

				elif type(Temperature_Calibration) == list: 
					Temperature_Calibration = np.array(Temperature_Calibration)
					# Temperature_Calibration[:,0] is heater voltages
					# Temperature_Calibration[:,1] is temperatures voltages
					
					# becasue ScanData heater voltages are read in as numbers like 0.24999999 and 0.2500001 instread of 0.25
					# as included in the Temperature_Calibration list/array, use this 'tol' to associate closest ScanData 
					# heater voltage to voltage in Temperature_Calibration list/array.
					tol =  0.0005 
				
				else:
					print('Temperature_Calibration metadata is not found or not of the correct type. Unable to add temperatures.')
					Add_Temperatures = False
			else:
				tol = None
				pass


			
		num_records = self.Sweep_Array.size
		for index in xrange(num_records): 
			sys.stdout.write('\r {0} of {1} '.format(index+1, num_records))
			sys.stdout.flush()

			#set current loop
			self.pick_loop(index)

			if Fit_Resonances == True:

				if Remove_Gain_Compression:
					# Remove Gain Compression
					self.decompress_gain(Compression_Calibration_Index = -1, Show_Plot = False, Verbose = False)

				# Normalize Loop
				#self.normalize_loop() 

				# Remove Cable Delay
				self.remove_cable_delay(Show_Plot = False, Verbose = False)	# should do nothing if a delay is defined in metadata

				# Fit loop to circle
				self.circle_fit(Show_Plot = False)
				if self.loop.circle_fit_exit_code != 0:
					self._define_sweep_array(index, Is_Valid = False)
				
				# Fit resonance parameters
				self.phase_fit(Fit_Method = 'Multiple',Verbose = False, Show_Plot = False)


				self._define_sweep_array(index, Q = self.loop.Q,
												Qc = self.loop.Qc,
												Fr = self.loop.fr,
												Mask = self.loop.phase_fit_mask,
												Chi_Squared = self.loop.chisquare,
												R = self.loop.R,
												r = self.loop.r,
												a = self.loop.a,
												b = self.loop.b,
												#Normalization  = self.loop.normalization,
												Theta = self.loop.theta,
												Phi = self.loop.phi,
												)

				if Complete_Fit:
					self.complete_fit(Use_Mask = True, Verbose = False , Show_Plot = False, Save_Fig = False, Sample_Size = 100, Use_Loop_Data = True)
					self._define_sweep_array(index, cQ = self.loop.cQ,
													cQc = self.loop.cQc,
													cFr = self.loop.cfr,
													cPhi = self.loop.cphi,
													cTheta = self.loop.ctheta,
													cR = self.loop.cR,
													cChi_Squared = self.loop.cchisquare,
													cIs_Valid = self.loop.cphase_fit_success if self.Sweep_Array['Is_Valid'][index] else self.Sweep_Array['Is_Valid'][index],

													sQ = self.loop.sQ,
													sQc = self.loop.sQc,
													sFr = self.loop.sfr,
													sPhi = self.loop.sphi,
													sTheta = self.loop.stheta,
													sR = self.loop.sR,
													sChi_Squared = self.loop.schisquare,
													sIs_Valid = self.loop.sphase_fit_success if self.Sweep_Array['Is_Valid'][index] else self.Sweep_Array['Is_Valid'][index]
													)

												


				# Only execute if phase_fit_success is False to avoid setting Is_Valid true when it was previously set fulse for a different reason, e.g bad Temp data
				if self.loop.phase_fit_success == False: 
					print('phase fit success is false')
					self._define_sweep_array(index, Is_Valid = False)

			if Compute_Preadout == True:
				if self.loop.fr != None:
					self._define_sweep_array(index, Preadout_dB = self.Sweep_Array['Pinput_dB'][index] + Preadout(self.loop.fr))
				elif np.abs(self.loop.freq[-1]-self.loop.freq[0]) > 1e9:
					print('Sweep bandwidth is {0} Hz. Sweep looks more like a survey. Preadout_dB is meaningless for a survey. Aborting Preadout computation... '.format(np.abs(self.loop.freq[-1]-self.loop.freq[0])))
					
				else:
					print('No resonance frquency (fr) on record for selected resonance. Estimating fr using sweep minimum.')
					fr = np.extract(np.abs(self.loop.z).min() == np.abs(self.loop.z),self.loop.freq)[0]
					self._define_sweep_array(index, Preadout_dB = self.Sweep_Array['Pinput_dB'][index] + fr)

			if Add_Temperatures == True:
				if self.metadata.Num_Temperatures < 1:
					condition = (self.Sweep_Array['Heater_Voltage'][index] + tol > Temperature_Calibration[:,0]) & (self.Sweep_Array['Heater_Voltage'][index] - tol < Temperature_Calibration[:,0])
					if condition.sum() >= 1:

						self.Sweep_Array['Temperature'][index] = Temperature_Calibration[condition,1][0] # <-- Needs to be updated so that duplicate voltages are handled correctly
					else:
						print('Unable to match unique temperature to heater voltage value for Sweep_Array[{0}]. {1} matches found.'.format(index,condition.sum() ))
				else:
					self._define_sweep_array(index, Temperature = 	self.Sweep_Array['Temperature_Readings'][index].mean()) 		
			# Clear out loop
			del(self.loop)
			self.loop = loop()
		print('\nSweep Array filled.')# Options selected Fit_Resonances = {0}, Compute_Preadout = {1}, Add_Temperatures = {2}'.format( Fit_Resonances,Compute_Preadout,Add_Temperatures))


	def _construct_readout_chain(self, F, Include_NA = True, Include_4K_to_40mK = False):
		'''
		F is a frequency array.
		Constructs gain, Tn_m (T noise magnitude), and Tn_p (phase)  lists.
		Each element of list corresponds to a component, e.g. primary amp, cable 1, second amp, attenator,.
		The order of the list correspondes to the order of components in the readout chain. First element is the first component (e.g. the primary amp)
		Each element of the list is an array the same shape as F. Each element of the arrays is the gain, Tn_m (T noise magnitude), and Tn_p (phase) at that frequency.

		This method does not use self.loop. data. It only uses self.metadata

		The System_Calibration and Cable_Calibration data are input into metadate at the time of data library creating (in the file Create_Lbrary.py)

		'''
		SC = self.metadata.System_Calibration # contains Noise powers, gains and P1dB of readout devices
		CC = self.metadata.Cable_Calibration # cable loss fit coefficients

		# Chain is the string of readout cables and amplifiers/devices
		chain  = []
		
		if Include_4K_to_40mK:
			chain.append('4K_to_40mK')

		if self.metadata.LNA['LNA'] is not None:
			chain.append(self.metadata.LNA['LNA'])

		chain.append('300K_to_4K')

		if self.metadata.RTAmp_In_Use:
			chain.append(self.metadata.RTAmp) 

		
		chain.append('One_Way_300K')

		if (self.metadata.Atten_NA_Input is not None) and (self.metadata.Atten_NA_Input>0):
			chain.append('Atten_NA_Input')

		if Include_NA:
			chain.append('NA')


		passive_device_temp = {'4K_to_40mK': (4. +.04)/2, '300K_to_4K' : (290.+4.)/2, 'One_Way_300K': 290., 'Atten_NA_Input':290.}
		Tn_p_s = []
		Tn_m_s = []
		g_s = []
		for i in xrange(len(chain)):
			device = chain[i]

			if device in CC.keys():
				g = CC[device][0]*np.sqrt(F)+CC[device][1]*F+CC[device][2]
				g = np.power(10.0,g/10.0)
				g_s.append(g)
				Tn = ((1.0/g)-1)*passive_device_temp[device]
				Tn_p_s.append(Tn)
				Tn_m_s.append(Tn)
				continue

			if device in SC.keys():
				g = np.polynomial.chebyshev.chebval(F,SC[device]['g_fit'])
				g = np.power(10.0,g/10.0)
				g_s.append(g)
				Tn_p_s.append(np.polynomial.chebyshev.chebval(F,SC[device]['Tn_p_fit']))
				Tn_m_s.append(np.polynomial.chebyshev.chebval(F,SC[device]['Tn_m_fit']))
				continue

			if device is 'Atten_NA_Input':
				g =  -np.abs(self.metadata.Atten_NA_Input)*np.ones_like(F)
				g = np.power(10.0,g/10.0)
				g_s.append(g)
				Tn = ((1.0/g)-1)*passive_device_temp[device]
				Tn_p_s.append(Tn)
				Tn_m_s.append(Tn)
				continue

			# warn me if the component is missing from calibration data
			print('Component in readout chain is not found in calibration data!! Aborting')
			return 
		return	g_s , Tn_m_s ,Tn_p_s
	
	def complete_fit(self, Use_Mask = True, Verbose = False , Show_Plot = False, Save_Fig = False, Sample_Size = 100, Use_Loop_Data = False):
		'''
		Sample_Size is the number of points used to extablish \sigma^2 for the gaussian noise model

		if Use_Loop_Data = True then values of Q, Qc, fr, phi are for initial guess are taken from curret loop object. If false, values come from self.Sweep_Array
		'''


		if self.loop.index == None:
			print 'Loop index is not specified. please pick_loop... Aborting'
			return

		Fit_Method = 'Multiple'
		if isinstance(Fit_Method,str): #Allow for single string input for Fit_Method
		   Fit_Method={Fit_Method}




		k = constants.value('Boltzmann constant') #unit is [J/k]
		BW = self.metadata.IFBW #unit is [Hz]	 
		# SC = self.metadata.System_Calibration # contains Noise powers, gains and P1dB of readout devices
		# CC = self.metadata.Cable_Calibration # cable loss fit coefficients
		R = 50 #system impedance

		#
		#
		# Implement Gain decompression on S21!
		#
		#
		if Use_Mask:
			F = ma.array(self.Sweep_Array[self.loop.index]['Frequencies'],mask = self.Sweep_Array[self.loop.index]['Mask'])
			F = F.compressed()
			S21 = ma.array(self.Sweep_Array[self.loop.index]['S21'],mask = self.Sweep_Array[self.loop.index]['Mask'])
			S21 = S21.compressed()
		else:
			F = self.Sweep_Array[self.loop.index]['Frequencies']
			S21 = self.Sweep_Array[self.loop.index]['S21']

		P_NA_out_dB = self.Sweep_Array[self.loop.index]['Pinput_dB'] #'out' as in our of NA, change of reference point
		P_NA_out_V2 = .001 * np.power(10,P_NA_out_dB/10) *2 *R 
		P_NA_in_V2 = np.square(np.abs(S21)) * P_NA_out_V2


		#Get chain and Noise temperatures for each element of readout chan and  at each frequency
		g_s , Tn_m_s ,Tn_p_s = self._construct_readout_chain(F)


		
		sigma_squared_m = np.zeros_like(F)
		sigma_squared_p = np.zeros_like(F)
		n = len(g_s)
		
		for i in xrange(n):	
			s2_m = 4*k*Tn_m_s[i]*R*BW # This sigma for the particular stage of the readout chain
			s2_p = 4*k*Tn_p_s[i]*R*BW
			#we assume s2_p * 4 * P_NA_in_V2 = s2_m ,  s2_p measured in radian^2
			sigma_squared_m = sigma_squared_m + s2_m*np.prod(g_s[i:], axis = 0) #rememebr g is a list of np vectors
			sigma_squared_p = sigma_squared_p + s2_p*np.square(np.prod(g_s[i:], axis = 0))/(4*P_NA_in_V2) #rememeber P_NA_in_V2 is a function of S21, see above definition




		if Use_Loop_Data == False:
			#a_0,b_0  = self.Sweep_Array[self.loop.index]['a'], self.Sweep_Array[self.loop.index]['b']
			R_0        = self.Sweep_Array[self.loop.index]['R']
			theta_0    = self.Sweep_Array[self.loop.index]['Theta']
			tau_0    = self.metadata.Electrical_Delay
			Q_0      = self.Sweep_Array[self.loop.index]['Q']
			Qc_0     = self.Sweep_Array[self.loop.index]['Qc']
			fr_0     = self.Sweep_Array[self.loop.index]['Fr'] 
			phi_0    = self.Sweep_Array[self.loop.index]['Phi']#(self.Sweep_Array[self.loop.index]['Phi'] * np.pi/180) + 0*np.pi
			
		else:
			#a_0,b_0  = self.loop.a, self.loop.b
			R_0      = self.loop.R
			theta_0  = self.loop.theta
			tau_0    = self.metadata.Electrical_Delay
			Q_0      = self.loop.Q 
			Qc_0     = self.loop.Qc 
			fr_0     = self.loop.fr 
			phi_0    = self.loop.phi# (self.loop.phi * np.pi/180) + 0*np.pi
			



		#p0 is the initial guess
		#p0 = np.array([a_0,b_0,tau_0,Q_0, Qc_0, fr_0, phi_0])
		p0 = np.array([R_0, theta_0,tau_0,Q_0, Qc_0, fr_0, phi_0])

		def obj(x,s21, sigma_squared_m,sigma_squared_p ,freq):# phase / magnitude fit
			# a,b,tau,Q, Qc, fr, phi= x
			# s21_fit  = norm * np.exp(np.complex(0.,np.angle(np.complex(a,b)))) * np.exp(np.complex(0,-2*np.pi*tau)*freq) * (1 - (Q/Qc)*np.exp(np.complex(0,phi)) / (1 + np.complex(0,2*Q)*(freq-fr)/fr ) )
			R,theta,tau,Q, Qc, fr, phi= x
			s21_fit  =  R * np.exp(np.complex(0.,theta)) * np.exp(np.complex(0,-2*np.pi*tau)*freq) * (1 - (Q/Qc)*np.exp(np.complex(0,phi)) / (1 + np.complex(0,2*Q)*(freq-fr)/fr ) )
	
			

			# diff = s21 - s21_fit
			# frac = (diff*diff.conj()).real/sigma_squared_m
			# #frac = (np.square(diff.real)/sigma_squared_m) + (np.square(diff.imag)/sigma_squared_m)
			frac = np.square(np.abs(s21) -  np.abs(s21_fit))*P_NA_out_V2/sigma_squared_m  + np.square(np.angle(s21/s21_fit))/sigma_squared_p  #(e^ia)/(e^ib) = e^i(a-b)
			N = freq.shape[0]*1.0 - x.shape[0]
			return  frac.sum()/N

		# Dont use masked data to sample points for Gaussian variance determination.
		if Use_Mask:
			S21_Sample = self.Sweep_Array[self.loop.index]['S21']
		else:
			S21_Sample = S21

		sigma_squared = 0 
		for i in xrange(Sample_Size):
			sigma_squared = sigma_squared + np.square(np.abs(S21_Sample[i] - S21_Sample[i+1]))
		sigma_squared = sigma_squared/(2.0*Sample_Size)
		
		def obj_s(x,s21, sigma_squared ,freq): # gaussian fit
			# a,b,tau,Q, Qc, fr, phi= x
			# s21_fit  = norm * np.exp(np.complex(0.,np.angle(np.complex(a,b)))) * np.exp(np.complex(0,-2*np.pi*tau)*freq) * (1 - (Q/Qc)*np.exp(np.complex(0,phi)) / (1 + np.complex(0,2*Q)*(freq-fr)/fr ) )
			R,theta,tau,Q, Qc, fr, phi= x
			s21_fit  =  R * np.exp(np.complex(0.,theta)) * np.exp(np.complex(0,-2*np.pi*tau)*freq) * (1 - (Q/Qc)*np.exp(np.complex(0,phi)) / (1 + np.complex(0,2*Q)*(freq-fr)/fr ) )


			# diff = s21 - s21_fit
			# frac = (diff*diff.conj()).real/sigma_squared_m
			# #frac = (np.square(diff.real)/sigma_squared_m) + (np.square(diff.imag)/sigma_squared_m)
			frac = np.square(np.abs(s21_fit-s21))/sigma_squared
			N = freq.shape[0]*1.0 - x.shape[0]
			return  frac.sum()/N
		
		
		#Each fit method is saved as a lambda function in a dictionary called fit_func
		fit_func = {}
		fit_func['cPowell'] = lambda : minimize(obj, p0, args=(S21,sigma_squared_m,sigma_squared_p ,F), method='Powell', jac=None, hess=None, hessp=None, bounds=None, constraints=(), tol=1e-15, callback=None, options={'disp':False})
		fit_func['sPowell'] = lambda : minimize(obj_s, p0, args=(S21,sigma_squared ,F), method='Powell', jac=None, hess=None, hessp=None, bounds=None, constraints=(), tol=1e-15, callback=None, options={'disp':False})

		#fit_func['Nelder-Mead']  = lambda : minimize(obj, p0, args=(S21,sigma_squared ,F), method='Nelder-Mead', jac=None, hess=None, hessp=None, bounds=None, constraints=(), tol=1e-15, callback=None, options={'disp':False, 'xtol' : 1e-6,'maxfev':1000})
		#fit_func['Newton-CG'] = lambda : minimize(obj, p0, args=(z_theta_c,f_c), method='Newton-CG', jac=jac, hess=hess, hessp=None, bounds=None, constraints=(),tol=1e-15, callback=None, options={'maxiter' : 50,'xtol': 1e-4,'disp':False})

		fit = {}
		if isinstance(Fit_Method,set):      #All string inputs for Fit_Method were changed to sets at the begining of phase_fit
		   if Fit_Method == {'Multiple'}:
		      for method in fit_func.keys():
		         fit[method] = fit_func[method]() # Execute the fit lambda function
		   else:
		      for method in Fit_Method:
		         if method not in fit_func.keys():
		            print("Unrecognized fit method. Aborting fit. \n\t Must choose one of {0} or 'Multiple'".format(fit_func.keys()))
		            return
		         else:   
		            fit[method] = fit_func[method]()
		else:
		   print("Unrecognized fit method data type. Aborting fit. \n\t Please specify using a string or a set of strings from one of {0} or 'Multiple'".format(fit_func.keys()))
		   return	         	   
		
		if Verbose:        				
			for method in fit.keys():
				print('\n{0} Minimzation Result:\n{1}\n'.format(method,fit[method]))

		
		
		# bestfit = list(fit)[0]
		# lowest = fit[bestfit].fun
		# for key in fit.keys(): 
		# 	if fit[key].fun < lowest:
		# 		lowest = fit[key].fun
		# 		bestfit = key

		cfit = 'cPowell' # phase / mag chi squared 
		#ca, cb, ctau = fit[cfit].x[0], fit[cfit].x[1], fit[cfit].x[2]
		self.loop.cR, self.loop.ctheta, ctau = cR, ctheta, ctau = fit[cfit].x[0], fit[cfit].x[1], fit[cfit].x[2]
		self.loop.cQ = cQ = fit[cfit].x[3]	
		self.loop.cQc = cQc = fit[cfit].x[4]	
		self.loop.cQi = cQi = 1.0/ ((1./self.loop.cQ ) - (1./self.loop.cQc )) 
		self.loop.cfr = cfr = fit[cfit].x[5]	
		self.loop.cphi = cphi =  fit[cfit].x[6]	
		self.loop.cchisquare = fit[cfit].fun
		self.loop.cphase_fit_success = fit[cfit].success

		sfit = 'sPowell' #gaussian chi squared
		#sa, sb, stau = fit[sfit].x[0], fit[sfit].x[1], fit[sfit].x[2]
		self.loop.sR, self.loop.stheta, stau  =sR, stheta, stau= fit[sfit].x[0], fit[sfit].x[1], fit[sfit].x[2]
		self.loop.sQ = sQ = fit[sfit].x[3]	
		self.loop.sQc = sQc = fit[sfit].x[4]	
		self.loop.sQi = sQi = 1.0/ ((1./self.loop.sQ ) - (1./self.loop.sQc )) 
		self.loop.sfr = sfr = fit[sfit].x[5]	
		self.loop.sphi = sphi =  fit[sfit].x[6]	
		self.loop.schisquare = fit[sfit].fun
		self.loop.sphase_fit_success = fit[sfit].success

		fit['sigma_squared_m'] = sigma_squared_m
		fit['sigma_squared_p'] = sigma_squared_p
		fit['sigma_squared'] = sigma_squared


		
		ax_dict = {}
		fig = plt.figure( figsize=(6.5, 6.5), dpi=100)
		fig_dict = {fig : ax_dict}
		ax = fig.add_subplot(111,aspect='equal')
		lines = []
		s21_concurrent_c = cR * np.exp(np.complex(0.,ctheta)) * np.exp(np.complex(0,-2*np.pi*ctau)*F) * (1 - (cQ/cQc)*np.exp(np.complex(0,cphi)) / ( 1 + np.complex(1, 2*cQ)*(F-cfr)/cfr  ))
		# s21_concurrent_c = norm * np.exp(np.complex(0.,np.angle(np.complex(ca,cb)))) * np.exp(np.complex(0,-2*np.pi*ctau)*F) * (1 - (cQ/cQc)*np.exp(np.complex(0,cphi)) / ( 1 + np.complex(1, 2*cQ)*(F-cfr)/cfr  ))
		lines.append(ax.plot(s21_concurrent_c.real,s21_concurrent_c.imag, markersize  = 3, linestyle = 'None',color = 'g', marker = 'o', markerfacecolor = 'g', markeredgecolor = 'g',  label = r'Concurrent Fit -  $\sigma_{V\theta}$')[0])

		s21_concurrent_s = sR * np.exp(np.complex(0.,stheta)) * np.exp(np.complex(0,-2*np.pi*stau)*F) * (1 - (sQ/sQc)*np.exp(np.complex(0,sphi)) / ( 1 + np.complex(1, 2*sQ)*(F-sfr)/sfr  ))
		#s21_concurrent_s = norm * np.exp(np.complex(0.,np.angle(np.complex(sa,sb)))) * np.exp(np.complex(0,-2*np.pi*stau)*F) * (1 - (sQ/sQc)*np.exp(np.complex(0,sphi)) / ( 1 + np.complex(1, 2*sQ)*(F-sfr)/sfr  ))
		lines.append(ax.plot(s21_concurrent_s.real,s21_concurrent_s.imag,markersize  = 3, color = 'm',linestyle = 'None', marker = 'o', markerfacecolor = 'm', markeredgecolor = 'm',  label = r'Concurrent Fit -  $\sigma_{G}$')[0])
		lines.append(ax.plot(s21_concurrent_s[0:Sample_Size:].real,s21_concurrent_s[0:Sample_Size:].imag,'m+', label = r'_Concurrent Fit -  $\sigma_{G}$')[0])

		lines.append(ax.plot(S21.real,S21.imag,markersize  = 3,color = 'b' ,marker = 'o',  linestyle = 'None',markerfacecolor = 'b', markeredgecolor = 'b', label = r'Raw Data - $S_{21}$')[0])


		s21_stepwise  =  R_0 * np.exp(np.complex(0.,theta_0)) * np.exp(np.complex(0,-2*np.pi*tau_0)*F) * (1 - (Q_0/Qc_0)*np.exp(np.complex(0,phi_0)) /( 1 + np.complex(1, 2*Q_0)*(F-fr_0)/fr_0  ))
		#s21_stepwise  = norm * np.exp(np.complex(0.,np.angle(np.complex(a_0,b_0)))) * np.exp(np.complex(0,-2*np.pi*tau_0)*F) * (1 - (Q_0/Qc_0)*np.exp(np.complex(0,phi_0)) /( 1 + np.complex(1, 2*Q_0)*(F-fr_0)/fr_0  ))
		lines.append(ax.plot(s21_stepwise.real,s21_stepwise.imag,markersize  = 3, color = 'r', linestyle = 'None',marker = 'o', markerfacecolor = 'r', markeredgecolor = 'r', label = r'Stepwise Fit - $\hat{S}_{21}$')[0])
		ax_dict.update({ax:lines})


		ax.set_xlabel(r'$\Re[S_{21}(f)]$')
		ax.set_ylabel(r'$\Im[S_{21}(f)]$')
		ax.yaxis.labelpad = -2
		ax.legend(loc = 'upper center', fontsize=5, bbox_to_anchor=(0.5, -0.1), ncol=2,scatterpoints =1, numpoints = 1, labelspacing = .02)
		#ax.legend(loc = 'best', fontsize=9,scatterpoints =1, numpoints = 1, labelspacing = .02) 

		plot_dict = fig_dict	

		if  Show_Plot:	
			plt.show()

		if Save_Fig == True:
			self._save_fig_dec(fig,'Concurrent_Fit_Index_{0}'.format(self.loop.index))
		


		return fit, plot_dict


	def _angle(self, z, deg = 0, return_offset = False):
		''' 
		IF Z IS A VECTOR, THEN ANGLE IS SHIFTED WRT FIRST ELEMENT!!!!

		If z is a masked array. angle(z) returns the angle of the elements of z
		within the branch [0,360] instead of [-180, 180], which is the branch used
		in np.angle(). The mask of angle(z) is set to be the mask of the input, z.

		If z is not a masked array, then angle(z) is the same as np.angle except 
		that range is [0,360] instead of [-180, 180]

		If z is a vector, then an angle shift is added to z  so the z[0] is 0 degrees
		If z is a number, then dont shift angle'''
		a = np.angle(z, deg = deg)
		
		try:
			offset = a[0] #if a is not a vector, then a[0] will throw an error
			a = a - offset  
		except:
			pass
		p = np.where(a<=0,1,0)
		n = 2
		units = n*np.pi if deg == 0 else n*180
		try:
			a = ma.array(a + p*units,mask =z.mask) 
		except:
			a = a + p*units #if z is not a masked array do this
		
		if return_offset:
			return a, offset
		else:
			return a


	def fit_system_calibration(self):
		'''compute chebyshev polynomial fits for  gain and noise values.
		save resulting polynomial coefficients list as:
		
		self.metadata.System_Calibration['device'][x + '_fit']

		where x is [gain, Tn_m ,Tn_p]... 

		use numpy.polynomial.chebyshev.chebval to evaluate polynomial

		'''
		max_degree = 9

		SC = self.metadata.System_Calibration
		#already_fit = [k + '_fit' for k in SC[key].keys()]
		
		# Dont fit 'freq' and 'P1dB' to 'freq'
		# dont fit specs which *are* fits already
		dont_fit  = set(['freq','P1dB'])
		for key in SC.keys():
			for spec in SC[key].keys():
				if spec.find('_fit') > -1:
					dont_fit.add(spec)

		for key in SC.keys():
			for spec in set(SC[key].keys()).difference(dont_fit): # everything in SC[key] except for dont_fit
				deg =  min(len(SC[key]['freq']) - 2,max_degree) if len(SC[key]['freq']) >2 else len(SC[key]['freq']) - 1
				coefs = np.polynomial.chebyshev.chebfit(SC[key]['freq'],  SC[key][spec], deg)
				#coefs = numpy.polynomial.polynomial.polyfit(SC[key]['freq'],  SC[key]['g'], deg)

				
				SC[key].update({spec + '_fit':list(coefs)})
			
	def fit_cable_loss(self, key_name, freq_range = [400e6, 1e9], Verbose = True, Show_Plot = True):
		'''produces fit to cable loss in the functional form:
		term1 + term2 + term3 = a * sqrt(f) + b * f + c
		term1 is the sum of inner and outer coaxial cable conductor losses
		term2 is due to coaxial cable dielectric loss
		term3 is a constant fudge factor
		The loss evaluates to units of dB.

		stores the  fit as dictionary
		(a,b,c,run,range_start,range_stop)= self.metadata.Cable_Calibration['One_Way_40mk']

		Two used this function load transmission for complete cable loop only (not amps or attens).
		Then call this function on that transmission data. This funciton creats the tuple (a,b,c,run,range_start,range_stop) in 
		metadata, where run is the name of the calibration run and range_start/stop is the frequency range over which the
		calibration is calculated.

		Create a function from a,b,c and it to the effect of attenuators on the input side of the cable loop.

		set freq_range = None to use full freq range	
		'''

		f   = self.loop.freq
		s21 = self.loop.z

		if freq_range == None:
			condition = f == f
		else:
			condition = (f>freq_range[0]) & (f<freq_range[1])
		
		f = np.extract(condition,f)
		s21 = np.extract(condition,s21)
		


		def obj(x,s21,f):
			a,b,c = x
			return np.square(20*np.log10(np.abs(s21)) - a*np.sqrt(f) - b*f - c).sum() #attenuation in dB/length goes as -a*sqrt(f)-b*f-c, where c has no theoretical basis.
		
		p0 = np.array([-3.0e-4,-1.0e-9 ,0.5])

		res = minimize(obj, p0, args=(s21,f), method='Nelder-Mead', jac=None, hess=None, hessp=None, bounds=None, constraints=(), tol=1e-15, callback=None, options={'disp':False, 'xtol' : 1e-6,'maxfev':1000})
		
		k = list(res.x/2.0) #devide by 2 to get one way loss
		k = k + [self.metadata.Run, f[0], f[-1]]

		if self.metadata.Cable_Calibration == None:
			cal = {}
			cal[key_name] = tuple(k)
			self.metadata.Cable_Calibration = self._Cable_Calibration = cal
		else:
			self.metadata.Cable_Calibration[key_name] =tuple(k)

		if Verbose == True:
			print(res)

		if Show_Plot == True:
			(fig,ax,) = self.plot_transmission(show = False)[:2]
			Cal  = lambda f: k[0]*np.sqrt(f)+k[1]*f+k[2]
			line = ax.plot(f, Cal(f)*2.0, 'r--', linewidth=3, label = 'fit - round trip')
			line = ax.plot(f, Cal(f), 'g-', linewidth=3, label = 'fit - one way')
			ax.set_xlim([freq_range[0]*0.75, freq_range[1]*1.25])
			leftvline = ax.axvline(x = freq_range[0],linewidth=2, color='k', linestyle = ':')
			rightvline = ax.axvline(x = freq_range[1],linewidth=2, color='k', linestyle = ':')
			ax.legend(loc = 'best', fontsize=10,scatterpoints =1, numpoints = 1, labelspacing = .1)
			plt.show()

	def nonlinear_fit(self, Fit_Method = 'Multiple', Verbose = True, Show_Plot = True, Save_Fig = False, Compute_Chi2 = False, Indexing = (None,None,None)):
		'''
		The indexing keyword allows for selection of the power sweep to be fit. 
		If P is the list of powers then Indexing = (Start,Stop,Step) is using only, P[Start,Stop, Step]
		'''

		from scipy.stats import chisquare
		import time
		R = 50 #System Impedance
		k = constants.value('Boltzmann constant') #unit is [J/k]
		BW = self.metadata.IFBW #unit is [Hz]	 

		if isinstance(Fit_Method,str): #Allow for single string input for Fit_Method
		   Fit_Method={Fit_Method}

		if self.loop.index == None:
			print('Loop index not chosen. Setting to 0.')
			index = 0
			self.pick_loop(index)

		Sweep_Array_Record_Index = self.loop.index 
		V = self.Sweep_Array['Heater_Voltage'][Sweep_Array_Record_Index]
		Fs = self.Sweep_Array['Fstart'][Sweep_Array_Record_Index]
		
		#### NOTE:  will need to fix for the case of sweeps with  duplicate V .... will involve using np.unique
		indices = np.where( (self.Sweep_Array['Heater_Voltage'] == V) & ( self.Sweep_Array['Fstart']==Fs))[0]
		P_min_index = np.where( (self.Sweep_Array['Heater_Voltage'] == V) & ( self.Sweep_Array['Fstart']==Fs) & (self.Sweep_Array['Pinput_dB'] == self.Sweep_Array['Pinput_dB'].min()))[0][0]

		##### Q, Qc, Qtl, fr  - used for initial guess in minimization
		##### Zfl, Zres - used in minimization, Zfl converts power to voltage			
		Q   = self.Sweep_Array['Q'][P_min_index]
		Qc  = self.Sweep_Array['Qc'][P_min_index]

		Qtl = np.power( (1./Q) - (1./Qc) , -1.)
		fr = self.Sweep_Array['Fr'][P_min_index]
		Zfl = self.metadata.Feedline_Impedance
		Zres = self.metadata.Resonator_Impedance


		power_sweep_list = []
		invalid_power_sweep_list = []
		start, stop, step = Indexing
		for index in indices[start:stop:step]: #
			# Clear out loop
			del(self.loop)
			self.loop = loop()
			
			# Pick new loop
			self.pick_loop(index)

	
			# Remove Gain Compression
			self.decompress_gain(Compression_Calibration_Index = -1, Show_Plot = False, Verbose = False)
			# Normalize Loop
			Outer_Radius = self.Sweep_Array['R'][index]
			if (Outer_Radius <= 0) or (Outer_Radius == None):
				print('Outer loop radius non valid. Using 1')
				Outer_Radius  = 1
			self.loop.z = self.loop.z/Outer_Radius
			#s21_mag = self.normalize_loop()

			# Remove Cable Delay
			self.remove_cable_delay(Show_Plot = False, Verbose = False)	
			# Fit loop to circle
			self.circle_fit(Show_Plot = False)

			Preadout = 0.001*np.power(10, self.Sweep_Array['Preadout_dB'][index]/10.0) #W, Readout power at device
			V1 = np.sqrt(Preadout*2*Zfl) #V, Readout amplitude at device
			mask = self.Sweep_Array['Mask'][index]
			f = ma.array(self.loop.freq,mask = mask)
			z = ma.array(self.loop.z,mask = mask)
			zc = np.complex(self.loop.a,self.loop.b)
			z = z*np.exp(np.complex(0,-np.angle(zc))) #rotate to real axis, but dont translate to origin 

			f_c = f.compressed()
			z_c = z.compressed()

			P_NA_out_dB = self.Sweep_Array[index]['Pinput_dB'] #Power out of the network analyzer, change of reference point
			P_NA_out_V2 = .001 * np.power(10,P_NA_out_dB/10) * 2 * R  #Voltage squared out of network analyzer

			if Compute_Chi2 is True: # Calculate variances for Chi2

				#z_c = z_c*Outer_Radius
				P_NA_in_V2 = np.square(np.abs(z_c)) * P_NA_out_V2
				g_s , Tn_m_s ,Tn_p_s = self._construct_readout_chain(f_c) # get the gain chain
				
			 	# g_i is the total gain between the device and readout digitizer (Network Analyzer) at the frequency f_i
				sigma_squared_m = np.zeros_like(f_c)
				sigma_squared_p = np.zeros_like(f_c)
				n = len(g_s)
				
				for i in xrange(n):
					s2_m = 4*k*Tn_m_s[i]*R*BW # This sigma for the particular stage of the readout chain
					s2_p = 4*k*Tn_p_s[i]*R*BW
					#we assume s2_p * 4 * P_NA_in_V2 = s2_m ,  s2_p measured in radian^2
					sigma_squared_m = sigma_squared_m + s2_m*np.prod(g_s[i:], axis = 0) #rememebr g is a list of np vectors
					sigma_squared_p = sigma_squared_p + s2_p*np.square(np.prod(g_s[i:], axis = 0))/(4*P_NA_in_V2) #rememeber P_NA_in_V2 is a function of S21, see above definition
			else:
				sigma_squared_m = np.ones_like(f_c)
				sigma_squared_p = np.ones_like(f_c)


			if self.Sweep_Array['Is_Valid'][index] == True: 
				power_sweep_list.append((V1,z_c,f_c,sigma_squared_m,sigma_squared_p,P_NA_out_V2,Outer_Radius))
			else:
				invalid_power_sweep_list.append((V1,z_c,f_c,sigma_squared_m,sigma_squared_p,P_NA_out_V2,Outer_Radius ))

		
		
		def progress(x):
			''' Add a dot to stdout at the end of each iteration without removing the dot from the previous iteration or 
			adding a new line.
			'''
			sys.stdout.write('.')
			sys.stdout.flush()
			

		V30V30 = fr #minimization will not converge if V30V30 is too small
		phiV1 = 0.0
		def obj(p):
			''' *** Objective function to be minimized for Chi2 and other fit ***
			'''
			parameter_dict = {'f_0':p[0], 'Qtl':p[1], 'Qc':p[2], 'phi31':p[3], 'eta':p[4], 'delta':p[5], 'Zfl':Zfl, 'Zres':Zres,  'phiV1':phiV1, 'V30V30':V30V30} 
			fd = self._nonlinear_formulae( parameter_dict, model = 2) # get the nonlinear formulae dict, fd 
			a,b,phi,tau = p[6:] # geometrical transformation parameters and tau - cable delay
			
			sumsq = 0
			N = 0 # total number of points in fit
			for sweep in power_sweep_list:
				V1_readout, S21_data, f,sigma_squared_m,sigma_squared_p,P_NA_out_V2 ,Outer_Radius= sweep 

				V3 = fd['V3'](S21_data,V1_readout)
				v1 = V3*V3.conjugate()
				

				#  Compute S21 and then Impose geometrical transformations to on it 
				S21_fit = (fd['S21'](v1,f) -  np.complex(a,b))/np.exp(np.complex(0,phi)+ np.complex(0,2.0*np.pi*tau)*f)
				
				if Compute_Chi2 is True:

					# Phase Mag approach doe not converge
					diff = np.square(( np.abs(S21_data) -  np.abs(S21_fit) ) * Outer_Radius)*P_NA_out_V2/sigma_squared_m  + np.square(np.angle(S21_data/S21_fit))/sigma_squared_p  #(e^ia)/(e^ib) = e^i(a-b)
					
					# Real Imaginary approach does not converge
					#diff = np.square(S21_data.real -  S21_fit.real)*P_NA_out_V2/sigma_squared_m  + np.square(S21_data.imag -  S21_fit.imag)*P_NA_out_V2/sigma_squared_m  
					
					# Real Imaginary approach does not *without# P_NA_out_V2 does converge!
					#diff = np.square(S21_data.real -  S21_fit.real)/sigma_squared_m  + np.square(S21_data.imag -  S21_fit.imag)/sigma_squared_m 
					
					sumsq = diff.sum()  + sumsq
					N = N + f.shape[0]*1.0
				else:
					diff = S21_data - S21_fit
					sumsq = (diff*diff.conjugate()).real.sum()  + sumsq
			
			if Compute_Chi2 is True:
				return sumsq/(N-p.shape[0])
			else:
				return sumsq
			



		phi31_est = np.pi/2
		eta_est = 0.001
		delta_est = 0.001
		a_est = 0.
		b_est = 0.
		phi_est = 0.
		tau_est = 0.0
		p0 = np.array([fr,Qtl,Qc,phi31_est,eta_est,delta_est,a_est,b_est, phi_est,tau_est ])
		#Each fit method is saved as a lambda function in a dictionary called fit_func
		fit_func = {}
		fit_func['Powell'] = lambda : minimize(obj, p0, method='Powell', jac=None, hess=None, hessp=None, bounds=None, constraints=None, tol=1e-20, callback=progress, options={'disp':False, 'maxiter': 100, 'maxfev': 50000, 'ftol':1e-14,'xtol':1e-14}) #maxfev: 11137 defaults: xtol=1e-4, ftol=1e-4,
		#fit_func['Nelder-Mead']  = lambda : minimize(obj, p0, args=(z_theta_c,f_c), method='Nelder-Mead', jac=None, hess=None, hessp=None, bounds=None, constraints=(), tol=1e-15, callback=None, options={'disp':False, 'xtol' : 1e-6,'maxfev':1000})
		#fit_func['Newton-CG'] = lambda : minimize(obj, p0, args=(z_theta_c,f_c), method='Newton-CG', jac=jac, hess=hess, hessp=None, bounds=None, constraints=(),tol=1e-15, callback=None, options={'maxiter' : 50,'xtol': 1e-4,'disp':False})


		fit = {}
		start = time.time()
	
		for method in fit_func.keys():
			sys.stdout.write('Iterating')
			sys.stdout.flush()
			fit[method] = fit_func[method]()
		
		finished = time.time()
		elapsed = (finished - start )/60.0 #minutes
		print 'Minimization took {:.2f} minutes'.format(elapsed)
		

		if fit.keys() != []: #if there is a fit object in the fit dictionary
			bestfit = list(fit)[0]
			lowest = fit[bestfit].fun # .fun is function value
			for key in fit.keys(): 
				if fit[key].fun < lowest:
					lowest = fit[key].fun
					bestfit = key
		else:
			bestfit = None



		if Verbose == True:
			print fit[bestfit]
		
		if Show_Plot == True:
			#Determine Sweep Direction
			direction = 'up'
			if direction == 'up':
				#min |--> up sweep (like at UCB)
				extreme = np.min 
			else:
				# max |--> down sweep
				extreme = np.max

			####### Set up plot objects
			fig = plt.figure( figsize=(5, 5), dpi=150)
			ax = {}
			gs = gridspec.GridSpec(2, 2)
			ax[1] = plt.subplot(gs[0, :])
			ax[2] = plt.subplot(gs[1, 0], aspect='equal' )
			ax[3] = plt.subplot(gs[1, 1])
			note = (r'Run {run}, Resonator width {width:.0f} $\mu m$'+'\n').format(run = self.metadata.Run, 
				width = (self.metadata.Resonator_Width if self.metadata.Resonator_Width is not None else 0)/1e-6)

			if bestfit != None:
				p = fit[bestfit].x
				parameter_dict = {'f_0':p[0], 'Qtl':p[1], 'Qc':p[2], 'phi31':p[3], 'eta':p[4], 'delta':p[5], 'Zfl':Zfl, 'Zres':Zres,  'phiV1':phiV1, 'V30V30':V30V30}
				fd = self._nonlinear_formulae( parameter_dict, model = 2) # get the nonlinear formulae dict, fd 
				a,b,phi,tau = p[6:]
				vline = ax[1].axvline(x = (parameter_dict['f_0']-fr)/fr,linewidth=1, color='y', linestyle = ':')#,   label = r'$f_{r}$')
				note = note + (r'$f_0$ = {f_0:3.2e} Hz, $Q_{sub1}$ = {Qtl:3.2e}, $Q_c$ = {Qc:3.2e}' +
					'\n' + r'$\phi_{sub2}$ = {ang:3.2f}$^\circ$, ${l1}$ = {et:3.2e}, ${l2}$ = {de:3.2e}').format(
					nl = '\n', et = parameter_dict['eta']/parameter_dict['V30V30'],
					de = parameter_dict['delta']/parameter_dict['V30V30'], 
					l1 = r'{\eta}/{V_{3,0}^2}',
					l2  = r'{\delta}/{V_{3,0}^2}',
					ang = parameter_dict['phi31']*180/np.pi, 
					sub1 = '{i}', sub2 = '{31}',**parameter_dict)
						

			for sweep in power_sweep_list:
				V1exp, S21exp, f ,sigma_squared_m,sigma_squared_p,P_NA_out_V2,Outer_Radius= sweep
				Pexp = 10*np.log10(V1exp*V1exp/(2 *Zfl*0.001))
				dff = (f - fr)/fr
				curve = ax[1].plot(dff,20*np.log10(np.abs(S21exp)), label = '$P_{probe}$ =' + ' {:3.2f} dBm'.format(Pexp)) # Pexp is Preadout
				curve = ax[2].plot(S21exp.real,S21exp.imag)

					
				if bestfit != None:
					#####Compute the experimental values of V3
					V3_exp = fd['V3'](S21exp,V1exp)

					#####Initialize arrays
					Number_of_Roots = 3
					V3V3 = np.ma.empty((f.shape[0],Number_of_Roots), dtype = np.complex128)
					V3V3_cubic = np.empty(f.shape)
					V3_cubic = np.empty(f.shape)
					S21_fit = np.empty_like(f,dtype = np.complex128)
					V3_fit = np.empty_like(f,dtype = np.complex128)

					for n in xrange(f.shape[0]):
						coefs = np.array([fd['z1z1'](f[n]), 2*fd['rez1z2c'](f[n]), fd['z2z2'](f[n]), -fd['z3z3'](V1exp)])
						V3V3[n] =np.ma.array(np.roots(coefs),mask= np.iscomplex(np.roots(coefs)),fill_value = 1)
						V3V3_cubic[n]    = extreme(np.extract(~V3V3[n].mask,V3V3[n])).real
						V3_cubic[n]    = np.sqrt(V3V3_cubic[n])
						# S21_fit is adjused to take into accout fit parameters a,b,phi,tau 
						S21_fit[n]  = (fd['S21'](V3V3_cubic[n],f[n]) - np.complex(a,b))*np.exp(np.complex(0,-phi)+ np.complex(0,-tau*2.0*np.pi)*f[n])
						
						# Note that V3_fit has the effect of a,b,phi,tau incorporated,  
						# So it should no be expected to equal V3_cubic
						V3_fit[n] = fd['V3'](S21_fit[n],V1exp)

					S21_cor = np.complex(a,b)+ np.exp(np.complex(0,phi)+ np.complex(0,2.0*np.pi*tau)*f)*S21exp
					V3_cor  = fd['V3'](S21_cor,V1exp)

					curve = ax[1].plot(dff,20*np.log10(np.abs(S21_fit)), linestyle = ':', color = 'c')
					curve = ax[2].plot(S21_fit.real,S21_fit.imag, linestyle = ':', color = 'c') 
					
					# curve = ax[3].plot(dff.real,V3_cor.real)
					# curve = ax[3].plot(dff.real,V3_cubic.real, linestyle = ':', color = 'g')
					

					# curve = ax[3].plot(dff,V3_exp.real)
					# curve = ax[3].plot(dff.real,V3_fit.real, linestyle = ':', color = 'c')#~np.iscomplex(V3fit)
					
					curve = ax[3].plot(dff,np.abs(V3_exp))
					curve = ax[3].plot(dff.real,np.abs(V3_fit), linestyle = ':', color = 'c')
				
			ax[1].set_title('Mag Transmission')
			ax[1].set_xlabel(r'$\delta f_0 / f_0$', color='k')
			ax[1].set_ylabel(r'$20 \cdot \log_{10}|S_{21}|$ [dB]', color='k') 
			ax[1].yaxis.labelpad = 0 #-6
			ax[1].xaxis.labelpad = 3
			ax[1].ticklabel_format(axis='x', style='sci',scilimits = (0,0), useOffset=True)
			ax[1].text(0.01, 0.01, note,
				verticalalignment='bottom', horizontalalignment='left',
				transform=ax[1].transAxes,
				color='black', fontsize=4)
			ax[1].legend(loc = 'upper center', fontsize=5, bbox_to_anchor=(.5, -1.5),  ncol=4,scatterpoints =1, numpoints = 1, labelspacing = .02)
			#bbox_to_anchor=(1.25, -0.1),bbox_transform = ax[2].transAxes, 



			ax[2].set_title('Resonance Loop')
			ax[2].set_xlabel(r'$\Re$[$S_{21}$]', color='k')
			ax[2].set_ylabel(r'$\Im$[$S_{21}$]', color='k')
			ax[2].yaxis.labelpad = -4
			ax[2].ticklabel_format(axis='x', style='sci',scilimits = (0,0),useOffset=False)

			ax[3].set_title('Resonator Amplitude')
			ax[3].set_xlabel(r'$\delta f_0 / f_0$', color='k')
			ax[3].ticklabel_format(axis='x', style='sci',scilimits = (0,0),useOffset=False)

			mpl.rcParams['axes.labelsize'] = 'small' # [size in points | 'xx-small' | 'x-small' | 'small' | 'medium....

			for k in ax.keys():
				ax[k].tick_params(axis='y', labelsize=5)
				ax[k].tick_params(axis='x', labelsize=5)

			plt.subplots_adjust(left=.1, bottom=.1, right=None ,wspace=.35, hspace=.3)
			
			if Save_Fig == True:
				name  = 'Nonlinear_Fit_' 
				if Compute_Chi2 is True:
					name = name  + 'Chi2_'
				self._save_fig_dec(fig, name + 'Start_Index_'+ str(Sweep_Array_Record_Index))
			plt.subplots_adjust(top =0.90)
			plt.suptitle('Fit to Nonlinear Resonator Data', fontweight='bold')
			plt.show()
 

 		fit.update(phiV1= phiV1, V30V30= V30V30)
		return fit, fig, ax #need to figure out a way to return all the curves too

	def _nonlinear_formulae(self, parameter_dict, model = 2):
		''' model 2 is paramterization based on input resonator amplitude V_3^-, e.g.: 
		parameter_dict = {'f_0':700e6, 'Qtl':300e3, 'Qc':80e3, 'eta':1e-1, 'delta':1e-6, 'Zfl':30, 'Zres':50, 'phi31': np.pi/2.03, 'phiV1':np.pi/10, 'V30V30':}
		'''
		d = parameter_dict
		k = {	'z1'     :  lambda f      : d['eta']/(d['Qtl']*d['V30V30']) + np.complex(0,1.0)*(2*d['delta']*f)/(d['V30V30']*d['f_0']),
				'z2'     :  lambda f      : (1.0/d['Qc']) + (1.0/d['Qtl']) + np.complex(0,2.0) *(f-d['f_0'])/d['f_0'],
				'z3'     :  lambda V1     : np.sqrt(np.complex(1,0)*d['Zres']/(np.pi * d['Qc'] *d['Zfl'])) * np.exp(np.complex(0,d['phi31'])) * V1 *  np.exp(np.complex(0,d['phiV1'])),
				'z1z1'   :  lambda f      : (k['z1'](f) * k['z1'](f).conjugate()).real,
				'z2z2'   :  lambda f      : (k['z2'](f) * k['z2'](f).conjugate()).real,
				'z3z3'   :  lambda V1     : (k['z3'](V1) * k['z3'](V1).conjugate()).real,
				'rez1z2c':  lambda f      : (k['z1'](f) * k['z2'](f).conjugate()).real,
				'imz1z2c':  lambda f      : (k['z1'](f) * k['z2'](f).conjugate()).imag,
				#'V3'     :  lambda S21,V1 : (S21 + (np.exp(np.complex(0,2.0*d['phi31'])) - 1.0)/2.0 )*V1*np.exp(np.complex(0,-1.0*d['phi31']))*np.sqrt(d['Zres']*d['Qc']/(d['Zfl']*np.pi)), # may have less rounding error 
				'V3'     :  lambda S21,V1 : (S21 + (np.exp(np.complex(0,2.0*d['phi31'])) - 1.0)/2.0 )*k['z3'](V1)*d['Qc']*np.exp(np.complex(0,-2.0*d['phi31'])),
				'S21'    :  lambda V3V3,f : ((1-np.exp(np.complex(0,2.0)*d['phi31']))/2 +( (1.0/d['Qc']) / (k['z2'](f) + k['z1'](f)*V3V3))*np.exp(np.complex(0,2.0)*d['phi31']))
			}
				#						   V3  = (S21 + (np.exp(np.complex(0,2.0*phi31)) - 1.0)/2.0 )*V1*np.exp(np.complex(0,-1.0*phi31))*np.sqrt(Z3*Qc/(Z1*np.pi))
				# #Now we use |V3V3|^2 = v1 to calculate the other two roots of the cubic, v2 and v3
				# v1 = V3*V3.conjugate()
				# term1 = -(z1z2c.real/z1z1) - v1/2.0
				# term2 = np.complex(0,1)*np.sqrt(4*z1z2c.imag*z1z2c.imag + 3*v1*v1*z1z1*z1z1 + 4*z1z1*z1z2c.real*v1)/(2*z1z1)
				# v2  = term1 + term2
				# v3  = term1 - term2

				# V3p = np.sqrt(v2)
				# V3m = np.sqrt(v3)

				# Note: Ztl can be removed from the calculation. In which case we use Pfl, (i.e. Vfl = sqrt(Pfl*Zfl*2)) 

		return k

	def generate_nonlinear_data(self,  Show_Plot = True, Phase_Noise_Variance = None, Amplitude_Noise_Variance = None, Like = None, Save_Fig = False,
		curve_parameter_dict = {'f_0':700e6, 'Qtl':300e3, 'Qc':80e3, 'eta':1e-1, 'delta':1e-6, 'Zfl':30, 'Zres':50, 'phi31': np.pi/2.03, 'phiV1':np.pi/10, 'V30V30':0.01},
		sweep_parameter_dict = {'Run': 'F1', 'Pprobe_dBm_Start' :-65.0,'Pprobe_dBm_Stop': -25.0, 'Pprobe_Num_Points':10, 'numBW':40,'num': 2000, 'Up_or_Down': 'Up', 'Freq_Spacing':'Linear'}):
		'''Creates and Loads Nonlinear Data
		eta -- Q nonlinearity
		delta --  freq nonlinearity	
		V30V30 -- V^2 normalization for nonlinearity

		If another KAM.sweep object is supplied in "Like" keyword, then its metadata will copied
		'''
		cd = curve_parameter_dict
		sd = sweep_parameter_dict


		#delete previous metadata object
		del(self.metadata)
		self.metadata = metadata()
		del(self.loop)
		self.loop = loop()

		# system_attenuation_before_device = -50 # dB,  Difference between Preadout and Pinput
		self.metadata.Electrical_Delay = 0.0
		self.metadata.Feedline_Impedance = cd['Zfl']
		self.metadata.Resonator_Impedance = cd['Zres']
		self.metadata.LNA = {}
		self.metadata.LNA['LNA'] =  'SiGe #1'
		self.metadata.RTAmp_In_Use = True
		self.metadata.Atten_At_4K = 40.
		self.metadata.Atten_NA_Output = 0. 
		self.metadata.Atten_NA_Input = 0.
		Cable_Calibration_Key = 'One_Way_40mK'
		self.metadata.Cable_Calibration = {}
		self.metadata.Cable_Calibration[Cable_Calibration_Key] = (0,0,0, 'False Cable', 0, 100e9)
		self.metadata.IFBW = 1.0
		
		if Like is not None: #would be better to confrim that Like is an instance of KAM.sweep
				self.metadata.__dict__.update(Like.metadata.__dict__)
			
		self.metadata.Electrical_Delay = 0
		self.metadata.Time_Created =   '05/01/2015 12:00:00' # or the current datetime datetime.datetime.now().strftime('%m/%d/%Y %H:%M:%S')
		self.metadata.Run = sd['Run']



		Q = 1.0/ ((1.0/cd['Qtl']) + (1.0/cd['Qc']))


		############################# Cable Calbration
		k = self.metadata.Cable_Calibration[Cable_Calibration_Key]
		Preadout = lambda f: k[0]*np.sqrt(f)+k[1]*f+k[2] - self.metadata.Atten_NA_Output - self.metadata.Atten_At_4K

		############################## Make Pprobe Array - This is power at the device.
		Pprobe_dBm = np.linspace(sd['Pprobe_dBm_Start'],sd['Pprobe_dBm_Stop'], sd['Pprobe_Num_Points'])
		Pprobe = 0.001* np.power(10.0,Pprobe_dBm/10.0)
		V1V1 = Pprobe *2*cd['Zfl']
		V1 = np.sqrt(V1V1) * np.exp(np.complex(0,1)*cd['phiV1'])
		# NOTE : V1 has a phase. Its a complex number

		################################# Create f array making sure it contains f_0
		BW = sd['numBW']*cd['f_0']/Q 

		if sd['Freq_Spacing'].lower() == 'triangular': #Triangular numbers - Denser around f_0
			T = np.linspace(1, sd['num'],  num=sd['num'], endpoint=True, retstep=False, dtype=None)
			T = T*(T+1.0)/2.0
			f_plus = (T*(BW/2)/T[-1]) + cd['f_0']
			f_minus = (-T[::-1]/T[-1])*(BW/2) + cd['f_0']
			f = np.hstack((f_minus,cd['f_0'],f_plus))

		if sd['Freq_Spacing'].lower() == 'linear': #linear
			f_plus = np.linspace(cd['f_0'], cd['f_0'] + BW/2,  num=sd['num'], endpoint=True, retstep=False, dtype=None)
			f_minus = np.linspace(cd['f_0']- BW/2,cd['f_0'],   num=sd['num']-1, endpoint=False, retstep=False, dtype=None)
			f = np.hstack((f_minus,f_plus))


		if sd['Freq_Spacing'].lower() == 'log': #logerithmic - Denser around f_0, for wide band sweeps
			f_plus = np.logspace(np.log10(cd['f_0']), np.log10(cd['f_0'] + BW/2),  num=sd['num'], endpoint=True, dtype=None)
			f_minus = -f_plus[:0:-1] + 2*cd['f_0']
			f = np.hstack((f_minus,f_plus))
		
		#################### Initialize Arrays		
		Number_of_Roots = 3
		V3V3 = np.ma.empty((f.shape[0],Number_of_Roots), dtype = np.complex128)

		V3V3_direction = np.empty(f.shape)
		S21_direction = np.empty_like(f,dtype = np.complex128)

		#################### Construct gain array
		if Like is not None: #would be better to confrim that Like is an instance of KAM.sweep
			g_s , Tn_m_s ,Tn_p_s = self._construct_readout_chain(f) # get the gain chain
			g = np.prod(g_s, axis = 0) # results in a numpy array  that is the same length as f...  a again for each frequency
		else:
			g = np.ones_like(f)
		# g_i is the total gain between the device and readout digitizer (Network Analyzer) at the frequency f_i

		
		#################### Find index of f_0
		try:
			f_0_index = np.where(f == curve_parameter_dict['f_0'])[0][0]
		except:
			d2 = np.square(f - curve_parameter_dict['f_0'])
			f_0_index = np.argmin(d2)



		#################### Initialize and Configure self.Sweep_Array
		tpoints = 0
		self._define_sweep_data_columns(f.size,tpoints)
		self.Sweep_Array = np.zeros(Pprobe_dBm.size, dtype = self.sweep_data_columns) #Sweep_Array holdes all sweep data. Its length is the number of sweeps


		fig = plt.figure( figsize=(5, 5), dpi=150)
		ax = {}
		ax[1] = fig.add_subplot(1,1,1)
		dff = (f - cd['f_0'])/cd['f_0']

		
		
		#Determine Sweep Direction
		if sd['Up_or_Down'].lower() == 'up':
			#min |--> up sweep (like at UCB)
			extreme = np.min 	
		else:
			# max |--> down sweep
			extreme = np.max


		print 'Generating False Data...'
		for index in xrange(Pprobe_dBm.size):
			sys.stdout.write('\r {0} of {1} '.format(index+1, Pprobe_dBm.size))
			sys.stdout.flush()
			Phase_Noise = np.zeros_like(f) if Phase_Noise_Variance is None else np.random.normal(scale = np.sqrt(Phase_Noise_Variance), size=f.shape)
			Amplitude_Noise = np.zeros_like(f) if Amplitude_Noise_Variance is None else np.random.normal(scale = np.sqrt(Amplitude_Noise_Variance), size=f.shape)

			for n in xrange(f.shape[0]):
				#################### Solve for Resonator amplitude using formulae from 
				fd = self._nonlinear_formulae(cd, model = 2) #get the nonlinear formulae dict, fd
				coefs = np.array([fd['z1z1'](f[n]), 2*fd['rez1z2c'](f[n]), fd['z2z2'](f[n]), -fd['z3z3'](V1[index])])


				V3V3[n] =np.ma.array(np.roots(coefs),mask= np.iscomplex(np.roots(coefs)),fill_value = 1)
				V3V3_direction[n]    = extreme(np.extract(~V3V3[n].mask,V3V3[n])).real
				S21_direction[n]  = fd['S21'](V3V3_direction[n],f[n])

			S21 = S21_direction + Amplitude_Noise + np.complex(0,1)*Phase_Noise
			
			Pin_dB = Pprobe_dBm[index] - Preadout(cd['f_0'])

			####################  Fill self.Sweep_Array
			self._define_sweep_array(index, Fstart = f[0], #Hz
										Fstop = f[-1], #Hz
										S21 = S21*np.sqrt(g), # g is power gain. so sqrt(g) is voltage gain #should be np.sqrt(g)*Rprobe_V/Pin_V  <-- _V meand voltage
										Frequencies = f, #Hz
										Preadout_dB = Pprobe_dBm[index],
										Pinput_dB = Pin_dB,
										Is_Valid = True,
										Mask = np.zeros(f.shape, dtype=np.bool),
										Chi_Squared = 0,
										Fr = cd['f_0'], #Note! we are using the resonance freq of the lowest power S21 for all 
										Q = Q,
										Qc = cd['Qc'],
										Heater_Voltage = 0.0,
										R = np.sqrt(g[f_0_index]) # remember,  V1 is the readout probe amplitude
										)

			curve = ax[1].plot(dff,20*np.log10(np.abs(S21)), linestyle = '-', label = '$P_{probe}$ = ' + '{0:.2f} dBm'.format(Pprobe_dBm[index]))
			

		################ Configure Plot
		
		ax[1].set_xlabel(r'$\delta f_0 / f_0$', color='k')
		ax[1].set_ylabel(r'Mag[$S_{21}$]', color='k')
		ax[1].yaxis.labelpad = -4
		ax[1].ticklabel_format(axis='x', style='sci',scilimits = (0,0), useOffset=True)
		ax[1].legend(loc = 'right', fontsize=4,scatterpoints =1, numpoints = 1, labelspacing = .1)
		for k in ax.keys():
			ax[k].tick_params(axis='y', labelsize=9)
			ax[k].tick_params(axis='x', labelsize=5)
		
		

		if Save_Fig:
			
			if Like is not  None:
				like = '_Like_' + Like.metadata.Run 
			else:
				like = ''
			self._save_fig_dec(fig,'Mock_Data' + like)
		#plt.subplots_adjust(left=.1, bottom=.1, right=None, top=.95 ,wspace=.4, hspace=.4)
		ax[1].set_title('Mag Transmission')
		plt.suptitle('Nonlinear Resonator Plots')
		plt.show()

		default_index = 0
		self.pick_loop(default_index)

		return fig, ax

	def _save_fig_dec(self, fig, name, Use_Date = False, Make_PGF = True):
			os.chdir(Plots_Dir)
			if self.metadata.Run is not None:
				name = self.metadata.Run+ '_'+ name  
			if Use_Date:
				name = name + '_'+ datetime.date.today().strftime("%Y%m%d")

			fig.savefig(name, dpi=300, transparency  = True, bbox_inches='tight')#Title.replace('\n','_').replace(' ','_')+date
			if Make_PGF:
				#cur_backend = mpl.get_backend()
				#plt.switch_backend('pgf')
				name = name + '.pgf'
				plt.savefig(name, bbox_inches = 'tight', transparancy = True) #
				#plt.switch_backend(cur_backend)
			os.chdir(Working_Dir)






	def old_phase_fit(self, Fit_Method = 'Multiple', Verbose = True, Show_Plot = True):
		from scipy.stats import chisquare
		
		if isinstance(Fit_Method,str): #Allow for single string input for Fit_Method
		   Fit_Method={Fit_Method}
		   
		def angle(z, deg = 0):
			''' If z is a masked array. angle(z) returns the angle of the elements of z
			within the branch [0,360] instead of [-180, 180], which is the branch used
			in np.angle(). The mask of angle(z) is set to be the mask of the input, z.

			If z is not a masked array, then angle(z) is the same as np.angle except 
			that range is [0,360] instead of [-180, 180]

			If z is a vector, then an angle shift is added to z  so the z[0] is 0 degrees
			If z is a number, then dont shift angle'''
			a = np.angle(z, deg = deg)
			try:
				a = a - a[0] #if a is not a vector, then a[0] will throw an error
			except:
				pass
			p = np.where(a<=0,1,0)
			n = 2
			units = n*np.pi if deg == 0 else n*180
			try:
				a = ma.array(a + p*units,mask =z.mask) 
			except:
				a = a + p*units #if z is not a masked array do this
			return a

		j = np.complex(0,1)
		try:
			zc = self.loop.a + j*self.loop.b
			r = self.loop.r
		except:
			print('Phase fit needs loop center and radius, which are not currently defined. Aborting phase fit.')
			return
		f = f0 = self.loop.freq
		z = z0 = self.loop.z
		

		# Remove duplicate frequency elements in z and f, e.g. places where f[n] = f[n+1]
		f_adjacent_distance =  np.hstack((np.abs(f[:-1]-f[1:]), [0.0]))
		z = z1 = ma.masked_where(f_adjacent_distance==0.0, z)
		f = f1 = ma.array(f,mask = z.mask) #Syncronize mask of f to match mask of z
		


		#Estimate Resonance frequency using minimum Dip or max adjacent distance
		Use_Dip = 1 
		if Use_Dip: #better for non-linear resonances with point near loop center
			zr_mag_est = np.abs(z).min()
			zr_est_index = np.where(np.abs(z)==zr_mag_est)[0][0]
		else:
			z_adjacent_distance = np.abs(z[:-1]-z[1:])
			zr_est_index = np.argmax(z_adjacent_distance) 
			zr_mag_est = np.abs(z[zr_est_index])


		#Transmission magnitude off resonance 
		Use_Fit = 1
		if Use_Fit:
			z_max_mag = np.abs(zc)+r
		else: #suspected to be better for non-linear resonances
			z_max_mag = np.abs(z).max()

		#Depth of resonance in dB
		depth_est = 20.0*np.log10(zr_mag_est/z_max_mag)

		#Magnitude of resonance dip at half max
		res_half_max_mag = (z_max_mag+zr_mag_est)/2

		#find the indices of the closest points to this magnitude along the loop, one below zr_mag_est and one above zr_mag_est
		a = np.square(np.abs(z[:zr_est_index+1]) - res_half_max_mag)
		lower_index = np.argmin(a)#np.where(a == a.min())[0][0]
		a = np.square(np.abs(z[zr_est_index:]) - res_half_max_mag)
		upper_index = np.argmin(a) + zr_est_index

		#estimate the FWHM bandwidth of the resonance
		f_upper_FWHM = f[upper_index]
		f_lower_FWHM = f[lower_index]
		FWHM_est = np.abs(f_upper_FWHM - f_lower_FWHM)
		fr_est = f[zr_est_index]
		theta_est = angle(z[zr_est_index])
		
		#consider refitting the circle here, or doing ellipse fit.

		#translate circle to origin, and rotate so that z[zr_est_index] has angle 0
		z = z2 = ma.array((z.data-zc)*np.exp(-j*(angle(zc)-np.pi)), mask = z.mask)
		
		#Compute theta_est before radious cut to prevent radius cut from removing z[f==fr_est]
		theta_est = angle(z[zr_est_index])	

		#Radius Cut: remove points that occur within r_cutoff of the origin of the centered data. 
		#(For non-linear resonances that have spurious point close to loop center)	
		r_fraction_in = 0.75
		r_fraction_out = 1.75
		r_cutoff_in  = r_fraction_in*r
		r_cutoff_out = r_fraction_out*r		
		z = z3 = ma.masked_where((np.abs(z2)<r_cutoff_in) | (np.abs(z2)>r_cutoff_out),z2, copy = True)
		# for substantially deformed loops we make sure that no more than Max_Removed_Radius_Cut points are removed from inner radious cut
		Max_Removed_Radius_Cut = 25
		while self._points_removed(z2, z3)[0] > Max_Removed_Radius_Cut:
			r_fraction_in = r_fraction_in - 0.02
			r_cutoff_in  = r_fraction_in*r
			z = z3 = ma.masked_where((np.abs(z2)<r_cutoff_in) | (np.abs(z2)>r_cutoff_out),z2, copy = True)
			print 'loosening inner radius cut: r_fraction_in = {}'.format(r_fraction_in)
			if r_fraction_in <= 0:
				break
		f = f3 = ma.array(f,mask = z.mask)


		#Bandwidth Cut: cut data that is more than N * FWHM_est away from zr_mag_est
		N = 8
		z = z4 = ma.masked_where((f > fr_est + N*FWHM_est) | (fr_est - N*FWHM_est > f),z,copy = True)
		f = f4 = ma.array(f,mask = z.mask)
		z_theta = angle(z)


		#Angle jump cut : masks points where angle jumps to next branch of angle function, 
		mask = (f > fr_est + 0.5*FWHM_est) | (f < fr_est + -0.5*FWHM_est)
		f_in_FWHM = ma.masked_where(mask,f) # or alternatively: f_in_FWHM = f; f_in_FWHM[mask] = ma.masked 
		edge1,edge2 = ma.flatnotmasked_edges(f_in_FWHM)
		angle_slope = (z_theta[edge2]-z_theta[edge1])/(f[edge2]-f[edge1]) # angle is decreasing if negative slope
		upper_cond = ((f > fr_est +  0.5*FWHM_est) & ((z_theta[edge2]<z_theta) if (angle_slope<0) else (z_theta[edge2]>z_theta))) 
		lower_cond = ((f < fr_est + -0.5*FWHM_est) & ((z_theta[edge1]>z_theta) if (angle_slope<0) else (z_theta[edge1]<z_theta))) 
		z = z5 = ma.masked_where(lower_cond|upper_cond,z, copy = True)
		f = f5 = ma.array(f,mask = z.mask)
		z_theta = z_theta5 = ma.array(z_theta,mask = z.mask)
		


		#theta_est = np.extract(f==fr_est,z_theta)[0] # The old lication of theta_est computation 
		Q_est = fr_est/FWHM_est


		#consider reducing computation by extracting only the unmasked values of z,f, and z_theta of the minimization
		#These commands return a masked array where all the masked elements are removed.
		#z = z[~z.mask]
		#f = f[~f.mask]
		#z_theta = z_theta[~z_theta.mask]

		#These commands return np array
		z_c = ma.compressed(z)
		f_c = ma.compressed(f)
		z_theta_c  = ma.compressed(z_theta)
		

		if mysys.startswith('Windows'):
			dt = np.float64
		else:	
			dt = np.float128

		def hess(x, z_theta,f): #to avoid overflow try to re write hessian so that all numbers are of order 1
			theta,fr,Q = x	
			H = np.zeros((3,3), dtype = dt)
			ff = (1-(f/fr))
			denom = (1+4.0*np.square(ff*Q))
			numer = (theta+z_theta-2.0*np.arctan(2.0*ff*Q))
			H[0,0] = (2.0*np.ones_like(z_theta)).sum()
			H[0,1] = ((-8.0*f*Q)/(np.square(fr)*denom)).sum()
			H[0,2] = ((8.0*ff)/denom).sum()
			H[1,0] = H[0,1] #((8.0*f*Q)/(np.square(fr)*denom)).sum()
			H[1,1] = ((32.0*np.square(f*Q/(np.square(fr)*denom)))  +   (64.0*np.square(f/(np.square(fr)*denom))*ff*np.power(Q,3)*numer)   +  ((16.0*f*Q/np.power(fr,3))*(numer/denom))).sum()
			H[1,2] = (((32.0*f*Q*ff)/np.square(fr*denom))  +  ((64.0*f*np.square(ff*Q)*numer)/(np.square(fr*denom)))  - ((8.0*f*numer)/(np.square(fr)*denom))).sum()
			H[2,0] = H[0,2] #((8.0*ff)/denom).sum()
			H[2,1] = H[1,2] #(((32.0*f*ff*Q)/np.square(fr*denom))  +  ((64.0*f*np.square(ff*Q)*numer)/(np.square(fr*denom)))  -  ((8.0*f*numer)/(np.square(fr)*denom))).sum()
			H[2,2] = (((32.0*np.square(ff))/np.square(denom))  +  ((64.0*np.power(ff,3)*Q*numer)/np.square(denom))).sum()				
			return H

		def jac(x,z_theta,f):
			theta,fr,Q = x
			J = np.zeros((3,),dtype = dt)    #np.zeros_like(x)
			ff = (1-(f/fr))
			denom = (1+4.0*np.square(ff*Q))
			numer = (theta+z_theta-2.0*np.arctan(2.0*ff*Q))	
			J[0] = np.sum(2.0*numer)
			J[1] = np.sum(-8.0*f*Q*numer/(np.square(fr)*denom))
			J[2] = np.sum(-8.0*ff*numer/denom)
			return J


		def obj(x,z_theta,f):
			theta,fr,Q = x
			return np.square(z_theta - theta - 2.0*np.arctan(2.0*Q*(1-f/fr))).sum()	 #<--- Need hessian of this


		def obj_ls(x,z_theta,f):
			'''object fuctinon for least squares fit'''
			theta,fr,Q = x
			residual  = z_theta - theta - 2.0*np.arctan(2.0*Q*(1-f/fr))	
			return residual

		#p0 is the initial guess
		p0 = np.array([theta_est,fr_est ,Q_est])
		
		#Each fit method is saved as a lambda function in a dictionary called fit_func
		fit_func = {}
		fit_func['Powell'] = lambda : minimize(obj, p0, args=(z_theta_c,f_c), method='Powell', jac=None, hess=None, hessp=None, bounds=None, constraints=(), tol=1e-15, callback=None, options={'disp':False})
		fit_func['Nelder-Mead']  = lambda : minimize(obj, p0, args=(z_theta_c,f_c), method='Nelder-Mead', jac=None, hess=None, hessp=None, bounds=None, constraints=(), tol=1e-15, callback=None, options={'disp':False, 'xtol' : 1e-6,'maxfev':1000})
		fit_func['Newton-CG'] = lambda : minimize(obj, p0, args=(z_theta_c,f_c), method='Newton-CG', jac=jac, hess=hess, hessp=None, bounds=None, constraints=(),tol=1e-15, callback=None, options={'maxiter' : 50,'xtol': 1e-4,'disp':False})

		fit = {}
		if isinstance(Fit_Method,set):      #All string inputs for Fit_Method were changed to sets at the begining of phase_fit
		   if Fit_Method == {'Multiple'}:
		      for method in fit_func.keys():
		         fit[method] = fit_func[method]() # Execute the fit lambda function
		   else:
		      for method in Fit_Method:
		         if method not in fit_func.keys():
		            print("Unrecognized fit method. Aborting fit. \n\t Must choose one of {0} or 'Multiple'".format(fit_func.keys()))
		            return
		         else:   
		            fit[method] = fit_func[method]()
		else:
		   print("Unrecognized fit method data type. Aborting fit. \n\t Please specify using a string or a set of strings from one of {0} or 'Multiple'".format(fit_func.keys()))
		   return	         	   
		               				
		
		#Does not work if the objective function is re-arranged as in the following
		# print('Nelder-Mead 2 ################# ')
		# def obj(x,z_theta,f):
		# 	theta,fr,Q = x
		# 	return np.square(np.tan((z_theta - theta)/2) - (2.0*Q*(1-f/fr))).sum()
		# res = minimize(obj, p0, args=(z_theta,f), method='Nelder-Mead', jac=None, hess=None, hessp=None, bounds=None, constraints=(), tol=1e-20, callback=None, options={'disp':True})
		# print(res)
	
		# Least square method does not find a good Q fit and the sum of the squares for solution is fairly high
		# print('Least Square ################# ')
		# print(fit['Least-Squares'])
		# print(np.square(fit['Least-Squares'][2]['fvec']).sum()) # this is the value of the sum of the squares for the solution
		# x = fit['Least-Squares'][0] 
		
		#x = res.x 
		bestfit = list(fit)[0]
		lowest = fit[bestfit].fun
		for key in fit.keys(): 
			if fit[key].fun < lowest:
				lowest = fit[key].fun
				bestfit = key
		
		self.loop.phase_fit_success = fit[bestfit].success
		self.loop.phase_fit_z = z5.data
		self.loop.phase_fit_mask = z5.mask
		self.loop.phase_fit_method = bestfit
		self.loop.Q = Q = fit[bestfit].x[2]
		self.loop.Qc = Qc = Q/(2*r)
		self.loop.Qi = Q*Qc/(Qc-Q)
		self.loop.fr = fr = fit[bestfit].x[1]
		self.loop.FWHM = fr/Q
		self.loop.phi = (fit[bestfit].x[0]-1*np.pi)*180/np.pi
		self.loop.chisquare, self.loop.pvalue = chisquare( z_theta_c,f_exp=fit[bestfit].x[0] + 2.0*np.arctan(2.0*Q*(1-f_c/fr)))
		self.loop.chisquare = self.loop.chisquare/ f_c.shape[0]
		#estimated quantities from MAG S21 
		self.loop.fr_est = fr_est
		self.loop.FWHM_est = FWHM_est
		self.loop.depth_est = depth_est
		self.loop.Q_est = Q_est

		if Verbose: 
			print('Duplicates cuts:\n\t{0} duplicate frequencies removed from loop data, {1} remaining data points'.format(*self._points_removed(z0,z1)))
			print('Radius cut:\n\t{2} points < r_loop*{0} or > r_loop*{1} found and removed, {3} remaining data points'.format(r_fraction_in, r_fraction_out,*self._points_removed(z2,z3)))
			print('Bandwidth cut:\n\t{1} points outside of fr_est +/- {0}*FWHM_est removed, {2} remaining data points'.format(N, *self._points_removed(z3,z4)))
			print('Angle jump cut:\n\t{0} points with discontinuous jumps in loop angle removed, {1} remaining data points'.format(*self._points_removed(z4,z5)))
			print('Initial Guess:\n\tLoop rotation {0}, fr {1}, Q {2}'.format(*p0))

			for method in fit.keys():
				print('\n{0} Minimzation Result:\n{1}\n'.format(method,fit[method]))



		if Show_Plot:
			total_removed, total_used_in_fit = self._points_removed(z0,z5)
			fig1 = plt.figure( facecolor = 'w',figsize = (10,10))
			ax = fig1.add_subplot(6,1,1)
			ax.set_title('Number of points used in fit = '+str(total_used_in_fit)+', Number of points removed = ' + str(total_removed) )
			#line = ax.plot(f1[~f5.mask], np.abs(z1[~z5.mask]),'g-', label = 'Used for Fit') #fails when no points are masked
			
			if f5.mask.size <= 1:#this is the case that there are no masked points, e.g. no mask. there will allways be 1 point in the mask due to adjacent distance
				line = ax.plot(ma.compressed(f1), np.abs(ma.compressed(z1)),'g-', label = 'Used for Fit')
			else:
				line = ax.plot(f1[~f5.mask], np.abs(z1[~z5.mask]),'g-', label = 'Used for Fit')
				line = ax.plot(f1[f5.mask], np.abs(z1[z5.mask]),'r.',markersize = 2,  alpha = 0.2, label = 'Excluded Data')
			line = ax.plot([f1[zr_est_index],f1[zr_est_index]] , [np.abs(z1[zr_est_index]),np.abs(zc)+r] ,'k.', label = 'Magitude Min and Max')
			line = ax.plot([f1[lower_index], f1[upper_index], f1[upper_index]], np.abs([z1[lower_index],z1[lower_index],z1[upper_index]]),'yo-', label = 'FWHM Estimate')
			ax.set_ylabel('Magnitude')
			ax.legend(loc = 'best', fontsize=10,scatterpoints =1, numpoints = 1, labelspacing = .1)
			
			
			ax = fig1.add_subplot(6,1,(2,4), aspect='equal')
			t = np.linspace(0, 2.0*np.pi, num=50, endpoint=True)
			line = ax.plot([0,zc.real],[0, zc.imag],'y*-', label = 'Center Vector')	
			line = ax.plot(zc.real + r*np.cos(t),zc.imag + r*np.sin(t),'y-', label = 'Circle Fit')		
			line = ax.plot(z1.real, z1.imag,'r:', label = 'Initial Location')
			line = ax.plot(z3.real, z3.imag,'r-', label = 'Aligned w/ Origin')
			line = ax.plot(z4.real, z4.imag,'g:', linewidth = 3,label = 'Bandwidth Cut')
			##pt = ax.plot([z1[0].real,z[~z.mask][0].real], [z1[0].imag,z[~z.mask][0].imag],'ko', label = 'First Point') fails when no points are masked
			pt = ax.plot([z1[0].real,ma.compressed(z5)[0].real], [z1[0].imag,ma.compressed(z5)[0].imag],'ko', label = 'First Point') #--
			pt = ax.plot(z2[zr_est_index].real, z2[zr_est_index].imag,'k*', label = 'Magnitude Min')

			#line = ax.plot(z4[z4.mask].data.real, z4[z4.mask].data.imag,'r.', alpha = 0.2, label = 'Excluded Data')
			line = ax.plot(z5[ma.getmaskarray(z5)].data.real, z5[ma.getmaskarray(z5)].data.imag,'r.', alpha = 0.2,label = 'Excluded Data')
			ax.legend(loc = 'best', fontsize=10, scatterpoints =1, numpoints = 1, labelspacing = .1)#,numpoints)
			
			text = ('$*Resonator Properties*$\n' + '$Q =$ ' + '{0:.2f}'.format(self.loop.Q) +'\nf$_0$ = ' + '{0:.6f}'.format(self.loop.fr/1e6) 
				+  ' MHz\n$Q_c$ = ' + '{0:.2f}'.format(self.loop.Qc) + '\n$Q_i$ = ' + '{0:.2f}'.format(self.loop.Qi) + '\n|S$_{21}$|$_{min}$ = ' 
				+ '{0:.3f}'.format(self.loop.depth_est) + ' dB' + '\nBW$_{FWHM}$ = ' + '{0:.3f}'.format(self.loop.FWHM/1e3) +  ' kHz' 
				+ '\n$\chi^{2}$ = ' + '{0:.4f}'.format(self.loop.chisquare) + '\n$\phi$ = ' + '{0:.3f}'.format(self.loop.phi) +' deg' + '\n$- $'+self.loop.phase_fit_method 
				+ ' fit $-$') 
			bbox_args = dict(boxstyle="round", fc="0.8")        
			fig1.text(0.10,0.7,text,
					ha="center", va="top", visible = True,
					bbox=bbox_args, backgroundcolor = 'w')


			ax = fig1.add_subplot(6,1,5)
			hline = ax.axhline(y = fit[bestfit].x[0],linewidth=2, color='y', linestyle = '-.',   label = r'$\theta_{r}$')
			vline = ax.axvline(x = fit[bestfit].x[1],linewidth=2, color='y', linestyle = ':',   label = r'$f_{r}$')
			line = ax.plot(f,z_theta,'g-',linewidth = 3,label = 'Data')
			line = ax.plot(f,(fit[bestfit].x[0] + 2.0*np.arctan(2.0*fit[bestfit].x[2]*(1-f/fit[bestfit].x[1]))),'g:', linewidth = 1, label = 'Fit ')
			#line = ax.plot(f5[~f5.mask][0],z_theta5[~z_theta5.mask][0],'ko',linewidth = 3,label = 'First Point') #Failes when  no points are masked
			line = ax.plot(ma.compressed(f5)[0],ma.compressed(z_theta5)[0],'ko',linewidth = 3,label = 'First Point')

			ax.set_ylabel('Angle [rad]')
			ax.legend(loc = 'right', fontsize=10,scatterpoints =1, numpoints = 1, labelspacing = .1)
			
			ax = fig1.add_subplot(6,1,6)
			vline = ax.axvline(x = fit[bestfit].x[1],linewidth=2, color='y', linestyle = ':',   label = r'$f_{r}$')
			style  = ['-','--',':','-.','+','x']; s = 0 #Cyclic iterable?
			for key in fit.keys():
				line = ax.plot(f,(z_theta - fit[key].x[0] - 2.0*np.arctan(2.0*fit[key].x[2]*(1-f/fit[key].x[1]))),'b'+style[s], linewidth = 3, label = 'Data - Fit ' + key)
				s += 1
			ax.set_ylabel('Angle [rad]')
			ax.set_xlabel('Freq [Hz]')
			ax.legend(loc = 'right', fontsize=10,scatterpoints =1, numpoints = 1, labelspacing = .1)
			plt.show()

			# fig = plt.figure( figsize=(5, 5), dpi=150)
			# ax = {}
			# ax[1] = fig.add_subplot(1,1,1)
			# #dff = (f5 - fr)/fr
			# dff = f5 
			# curve = ax[1].plot(dff,np.abs(z5))
			# ax[1].ticklabel_format(axis='x', style='sci',scilimits = (0,0), useOffset=True)	

			# for k in ax.keys():
			# 	ax[k].tick_params(axis='y', labelsize=9)
			# 	ax[k].tick_params(axis='x', labelsize=5)
			# plt.show()

	def old_nonlinear_fit(self, Fit_Method = 'Multiple', Verbose = True, Show_Plot = True, Save_Fig = False, Indexing = (None,None,None)):
		'''
		The indexing keyword allows for selection of the power sweep to be fit. 
		If P is the list of powers then Indexing = (Start,Stop,Step) is using only, P[Start,Stop, Step]
		'''

		from scipy.stats import chisquare
		import time
		
		
		if isinstance(Fit_Method,str): #Allow for single string input for Fit_Method
		   Fit_Method={Fit_Method}

		if self.loop.index == None:
			print('Loop index not chosen. Setting to 0.')
			index = 0
			self.pick_loop(index)

		Sweep_Array_Record_Index = self.loop.index 
		V = self.Sweep_Array['Heater_Voltage'][Sweep_Array_Record_Index]
		Fs = self.Sweep_Array['Fstart'][Sweep_Array_Record_Index]
		
		#### NOTE:  will need to fix for the case of sweeps with  duplicate V .... will involve using np.unique
		indices = np.where( (self.Sweep_Array['Heater_Voltage'] == V) & ( self.Sweep_Array['Fstart']==Fs))[0]
		P_min_index = np.where( (self.Sweep_Array['Heater_Voltage'] == V) & ( self.Sweep_Array['Fstart']==Fs) & (self.Sweep_Array['Pinput_dB'] == self.Sweep_Array['Pinput_dB'].min()))[0][0]

		##### Q, Qc, Qtl, fr  - used for initial guess in minimization
		##### Zfl, Zres - used in minimization, Zfl converts power to voltage			
		Q   = self.Sweep_Array['Q'][P_min_index]
		Qc  = self.Sweep_Array['Qc'][P_min_index]

		Qtl = np.power( (1./Q) - (1./Qc) , -1.)
		fr = self.Sweep_Array['Fr'][P_min_index]
		Zfl = self.metadata.Feedline_Impedance
		Zres = self.metadata.Resonator_Impedance


		power_sweep_list = []
		invalid_power_sweep_list = []
		start, stop, step = Indexing
		for index in indices[start:stop:step]: #
			# Clear out loop
			del(self.loop)
			self.loop = loop()
			
			# Pick new loop
			self.pick_loop(index)

	
			# Remove Gain Compression
			self.decompress_gain(Compression_Calibration_Index = -1, Show_Plot = False, Verbose = False)
			# Normalize Loop
			norm = self.Sweep_Array['R'][index]
			if (norm <= 0) or (norm == None):
				print('Outer loop radius non valid. Using using 1')
				norm  = 1
			self.loop.z = self.loop.z/norm
			#s21_mag = self.normalize_loop()
			# Remove Cable Delay
			self.remove_cable_delay(Show_Plot = False, Verbose = False)	
			# Fit loop to circle
			self.circle_fit(Show_Plot = False)

			Preadout = 0.001*np.power(10, self.Sweep_Array['Preadout_dB'][index]/10.0) #W, Power out of NA
			V1 = np.sqrt(Preadout*2*Zfl)
			mask = self.Sweep_Array['Mask'][index]
			f = ma.array(self.loop.freq,mask = mask)
			z = ma.array(self.loop.z,mask = mask)
			zc = np.complex(self.loop.a,self.loop.b)
			z = z*np.exp(np.complex(0,-np.angle(zc))) #rotate to real axis, but dont translate to origin 

			if self.Sweep_Array['Is_Valid'][index] == True: 
				power_sweep_list.append((V1,z.compressed(),f.compressed()))
			else:
				invalid_power_sweep_list.append((V1,z.compressed(),f.compressed()))

		
		
		def progress(x):
			''' Add a dot to stdout at the end of each iteration without removing the dot from the previous iteration or 
			adding a new line.
			'''
			sys.stdout.write('.')
			sys.stdout.flush()
			

		V30V30 = fr #minimization will not converge if V30V30 is too small
		phiV1 = 0.0
		def obj(p):
			''' Objective function to be minimized
			'''
			parameter_dict = {'f_0':p[0], 'Qtl':p[1], 'Qc':p[2], 'phi31':p[3], 'eta':p[4], 'delta':p[5], 'Zfl':Zfl, 'Zres':Zres,  'phiV1':phiV1, 'V30V30':V30V30} 
			fd = self._nonlinear_formulae( parameter_dict, model = 2) # get the nonlinear formulae dict, fd 
			a,b,phi,tau = p[6:]
			
			sumsq = 0
			for sweep in power_sweep_list:
				V1e, S21e, f = sweep #V1e, S21e -- experimental values of these quantities
				V1  = V1e
				
				# Impose geometrical transformations to S21
				S21 = np.complex(a,b)+ np.exp(np.complex(0,phi)+ np.complex(0,2.0*np.pi*tau)*f)*S21e
				
				V3 = fd['V3'](S21,V1)
				v1 = V3*V3.conjugate()
				s21 = fd['S21'](v1,f)

				##### Old way by means of direct calculations rather then centralized nonlinear funct dict  - Probably faster
				#V3  = (S21 + (np.exp(np.complex(0,2.0*phi31)) - 1.0)/2.0 )*V1*np.exp(np.complex(0,-1.0*phi31))*np.sqrt(Z3*Qc/(Z1*np.pi))
				#v1 = V3*V3.conjugate()		
				#s21 = ((1-np.exp(np.complex(0,2.0)*phi31))/2 +( (1/Qc) / ((1/Qc) + (1/Qtl)*(1+eta*v1/V30V30) + np.complex(0,2)* (((f-f_0)/f_0) + delta*(v1/V30V30)*(f/f_0))))*np.exp(np.complex(0,2.0)*phi31))
				diff = S21 - s21
				sumsq = (diff*diff.conjugate()).real.sum()  + sumsq
			return sumsq
			
	
		phi31_est = np.pi/2
		eta_est = 0.001
		delta_est = 0.001
		a_est = 0.
		b_est = 0.
		phi_est = 0.
		tau_est = 0.0
		p0 = np.array([fr,Qtl,Qc,phi31_est,eta_est,delta_est,a_est,b_est, phi_est,tau_est ])
		#Each fit method is saved as a lambda function in a dictionary called fit_func
		fit_func = {}
		fit_func['Powell'] = lambda : minimize(obj, p0, method='Powell', jac=None, hess=None, hessp=None, bounds=None, constraints=None, tol=1e-20, callback=progress, options={'disp':False, 'maxiter': 70, 'maxfev': 50000, 'ftol':1e-14,'xtol':1e-14}) #maxfev: 11137 defaults: xtol=1e-4, ftol=1e-4,
		#fit_func['Nelder-Mead']  = lambda : minimize(obj, p0, args=(z_theta_c,f_c), method='Nelder-Mead', jac=None, hess=None, hessp=None, bounds=None, constraints=(), tol=1e-15, callback=None, options={'disp':False, 'xtol' : 1e-6,'maxfev':1000})
		#fit_func['Newton-CG'] = lambda : minimize(obj, p0, args=(z_theta_c,f_c), method='Newton-CG', jac=jac, hess=hess, hessp=None, bounds=None, constraints=(),tol=1e-15, callback=None, options={'maxiter' : 50,'xtol': 1e-4,'disp':False})


		fit = {}
		start = time.time()
	
		for method in fit_func.keys():
			sys.stdout.write('Iterating')
			sys.stdout.flush()
			fit[method] = fit_func[method]()
		
		finished = time.time()
		elapsed = (finished - start )/60.0 #minutes
		print 'Minimization took {:.2f} minutes'.format(elapsed)
		

		if fit.keys() != []: #if there is a fit object in the fit dictionary
			bestfit = list(fit)[0]
			lowest = fit[bestfit].fun # .fun is function value
			for key in fit.keys(): 
				if fit[key].fun < lowest:
					lowest = fit[key].fun
					bestfit = key
		else:
			bestfit = None

		if Verbose == True:
			print fit[bestfit]
		
		if Show_Plot == True:
			#Determine Sweep Direction
			direction = 'up'
			if direction == 'up':
				#min |--> up sweep (like at UCB)
				extreme = np.min 
			else:
				# max |--> down sweep
				extreme = np.max

			####### Set up plot objects
			fig = plt.figure( figsize=(5, 5), dpi=150)
			ax = {}
			gs = gridspec.GridSpec(2, 2)
			ax[1] = plt.subplot(gs[0, :])
			ax[2] = plt.subplot(gs[1, 0], aspect='equal' )
			ax[3] = plt.subplot(gs[1, 1])
			note = (r'Run {run}, Resonator width {width:.0f} $\mu m$'+'\n').format(run = self.metadata.Run, 
				width = (self.metadata.Resonator_Width if self.metadata.Resonator_Width is not None else 0)/1e-6)

			if bestfit != None:
				p = fit[bestfit].x
				parameter_dict = {'f_0':p[0], 'Qtl':p[1], 'Qc':p[2], 'phi31':p[3], 'eta':p[4], 'delta':p[5], 'Zfl':Zfl, 'Zres':Zres,  'phiV1':phiV1, 'V30V30':V30V30}
				fd = self._nonlinear_formulae( parameter_dict, model = 2) # get the nonlinear formulae dict, fd 
				a,b,phi,tau = p[6:]
				vline = ax[1].axvline(x = (parameter_dict['f_0']-fr)/fr,linewidth=1, color='y', linestyle = ':')#,   label = r'$f_{r}$')
				note = note + (r'$f_0$ = {f_0:3.2e} Hz, $Q_{sub1}$ = {Qtl:3.2e}, $Q_c$ = {Qc:3.2e}' +
					'\n' + r'$\phi_{sub2}$ = {ang:3.2f}$^\circ$, ${l1}$ = {et:3.2e}, ${l2}$ = {de:3.2e}').format(
					nl = '\n', et = parameter_dict['eta']/parameter_dict['V30V30'],
					de = parameter_dict['delta']/parameter_dict['V30V30'], 
					l1 = r'{\eta}/{V_{3,0}^2}',
					l2  = r'{\delta}/{V_{3,0}^2}',
					ang = parameter_dict['phi31']*180/np.pi, 
					sub1 = '{i}', sub2 = '{31}',**parameter_dict)
						

			for sweep in power_sweep_list:
				V1exp, S21exp, f = sweep
				Pexp = 10*np.log10(V1exp*V1exp/(2 *Zfl*0.001))
				dff = (f - fr)/fr
				curve = ax[1].plot(dff,20*np.log10(np.abs(S21exp)), label = '$P_{probe}$ =' + ' {:3.2f} dBm'.format(Pexp)) # Pexp is Preadout
				curve = ax[2].plot(S21exp.real,S21exp.imag)

					
				if bestfit != None:
					#####Compute the experimental values of V3
					V3_exp = fd['V3'](S21exp,V1exp)

					#####Initialize arrays
					Number_of_Roots = 3
					V3V3 = np.ma.empty((f.shape[0],Number_of_Roots), dtype = np.complex128)
					V3V3_cubic = np.empty(f.shape)
					V3_cubic = np.empty(f.shape)
					S21_fit = np.empty_like(f,dtype = np.complex128)
					V3_fit = np.empty_like(f,dtype = np.complex128)

					for n in xrange(f.shape[0]):
						coefs = np.array([fd['z1z1'](f[n]), 2*fd['rez1z2c'](f[n]), fd['z2z2'](f[n]), -fd['z3z3'](V1exp)])
						V3V3[n] =np.ma.array(np.roots(coefs),mask= np.iscomplex(np.roots(coefs)),fill_value = 1)
						V3V3_cubic[n]    = extreme(np.extract(~V3V3[n].mask,V3V3[n])).real
						V3_cubic[n]    = np.sqrt(V3V3_cubic[n])
						# S21_fit is adjused to take into accout fit parameters a,b,phi,tau 
						S21_fit[n]  = (fd['S21'](V3V3_cubic[n],f[n]) - np.complex(a,b))*np.exp(np.complex(0,-phi)+ np.complex(0,-tau*2.0*np.pi)*f[n])
						# Note that V3_fit has the effect of a,b,phi,tau incorporated,  
						# So it should no be expected to equal V3_cubic
						V3_fit[n] = fd['V3'](S21_fit[n],V1exp)

					S21_cor = np.complex(a,b)+ np.exp(np.complex(0,phi)+ np.complex(0,2.0*np.pi*tau)*f)*S21exp
					V3_cor  = fd['V3'](S21_cor,V1exp)

					curve = ax[1].plot(dff,20*np.log10(np.abs(S21_fit)), linestyle = ':', color = 'c')
					curve = ax[2].plot(S21_fit.real,S21_fit.imag, linestyle = ':', color = 'c') 
					
					# curve = ax[3].plot(dff.real,V3_cor.real)
					# curve = ax[3].plot(dff.real,V3_cubic.real, linestyle = ':', color = 'g')
					

					# curve = ax[3].plot(dff,V3_exp.real)
					# curve = ax[3].plot(dff.real,V3_fit.real, linestyle = ':', color = 'c')#~np.iscomplex(V3fit)
					
					curve = ax[3].plot(dff,np.abs(V3_exp))
					curve = ax[3].plot(dff.real,np.abs(V3_fit), linestyle = ':', color = 'c')
				
			ax[1].set_title('Mag Transmission')
			ax[1].set_xlabel(r'$\delta f_0 / f_0$', color='k')
			ax[1].set_ylabel(r'$20 \cdot \log_{10}|S_{21}|$ [dB]', color='k') 
			ax[1].yaxis.labelpad = 0 #-6
			ax[1].xaxis.labelpad = 3
			ax[1].ticklabel_format(axis='x', style='sci',scilimits = (0,0), useOffset=True)
			ax[1].text(0.01, 0.01, note,
				verticalalignment='bottom', horizontalalignment='left',
				transform=ax[1].transAxes,
				color='black', fontsize=4)
			ax[1].legend(loc = 'upper center', fontsize=5, bbox_to_anchor=(.5, -1.5),  ncol=4,scatterpoints =1, numpoints = 1, labelspacing = .02)
			#bbox_to_anchor=(1.25, -0.1),bbox_transform = ax[2].transAxes, 



			ax[2].set_title('Resonance Loop')
			ax[2].set_xlabel(r'$\Re$[$S_{21}$]', color='k')
			ax[2].set_ylabel(r'$\Im$[$S_{21}$]', color='k')
			ax[2].yaxis.labelpad = -4
			ax[2].ticklabel_format(axis='x', style='sci',scilimits = (0,0),useOffset=False)

			ax[3].set_title('Resonator Amplitude')
			ax[3].set_xlabel(r'$\delta f_0 / f_0$', color='k')
			ax[3].ticklabel_format(axis='x', style='sci',scilimits = (0,0),useOffset=False)

			mpl.rcParams['axes.labelsize'] = 'small' # [size in points | 'xx-small' | 'x-small' | 'small' | 'medium....

			for k in ax.keys():
				ax[k].tick_params(axis='y', labelsize=5)
				ax[k].tick_params(axis='x', labelsize=5)

			plt.subplots_adjust(left=.1, bottom=.1, right=None ,wspace=.35, hspace=.3)
			
			if Save_Fig == True:
				self._save_fig_dec(fig, 'Nonlinear_Fit_Start_Index_' + str(Sweep_Array_Record_Index))
			plt.subplots_adjust(top =0.90)
			plt.suptitle('Fit to Nonlinear Resonator Data', fontweight='bold')
			plt.show()
 

 		fit.update(phiV1= phiV1, V30V30= V30V30)
		return fit, fig, ax #need to figure out a way to return all the curves too
		 		













	