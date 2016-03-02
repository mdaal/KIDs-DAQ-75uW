import Instruments
import Fridge_Interfaces
import os, io, datetime, time
import logging
import numpy as np
import tables
import matplotlib as mpl
import matplotlib.pyplot as plt
import contextlib
import fractions

data_directory_prefix = 'Z:\\user\\miguel\\KIPS_Data'

class measurement_manager:

	def __init__(self, devices, data_dir):
		'''
		data_dir is the run name
		NOTE: Change Thermometer_Configuration in KAM to Devices,  add measurement_start_time in KAM
		'''
		
		self.Verbose = True
		self.Show_Plots = True
		self.devices = devices
		self.inl = Instruments.Instrument_Name_List
		
		
		self.start_directory = os.getcwd()
		self.data_directory_path = data_directory_prefix + os.sep + data_dir
		
		if os.path.exists(self.data_directory_path) == False:
			logging.error('data directory path does no exist.')

		self.measurement_metadata = {'Devices': devices}
		self._read_about_file()

		self.num_aux_sweep_pts   = 1
		self.num_syn_freq_points =  1
		self.num_power_sweep_pts = 1
		self.num_temp_sweep_pts =  1
		self.measurement_start_time = None  #datetime.datetime.now().strftime('%Y%m%d%H%M')
		self.current_sweep_data_table = None #path within hdf5 file to current data table 

	def __del__(self):
		pass

	def _print(self, mssg):
		if self.Verbose:
			print(mssg)

	def _read_about_file(self):
		'''
		Find About_XXX file, read it, and update measurement metadata with its contents
		'''

		# List the files at data dir path
		files_in_dir = os.listdir(self.data_directory_path)

		#find the about file
		self.about_file_path = None
		for file_in_dir in files_in_dir:
			if  file_in_dir.lower().startswith('about_'):
				self.about_file_path = self.data_directory_path + os.sep + file_in_dir
				break

		#tokens = ['Run:', 'Start Date:','Location:', 'Sensor:', 'Ground Plane:', 'Box:', 'Press:', 'HEMT:', 'Atten 4K:', 'Notes:']
		tokens = [('Run:','Run', str), ('Start Date:','Fridge_Run_Start_Date',str), ('Location:','Test_Location', str), 
				('Sensor:','Sensor',str), ('Ground Plane:','Ground_Plane',str), ('Box:','Box',str), ('Press:','Press',str), 
				('Notes:','Notes',str),('Atten 4K:', 'Atten_At_4K', np.float32), ('HEMT:', 'LNA', lambda x: {'LNA': str(x)}),
				('Base Temperature:','Fridge_Base_Temp',np.float32)]
		self._read_tokens_in_file(self.about_file_path, tokens)

	def _update_file(self, file_path, token, update):
		sep_pos = file_path.rfind(os.sep)
		if sep_pos == -1:
			tmp_file_path = 'tmp.txt'
		else:
			tmp_file_path = file_path[:sep_pos +1] + 'tmp.txt'

	
		with io.open(file_path, mode = 'r') as file:
			with io.open(tmp_file_path, mode = 'w') as tmp_file:
				proceed = 1
				line = ''
				while proceed:
					if line.startswith(token):
						tmp_file.write(unicode(line))
						tmp_file.write(unicode( update + ' |\n'))
					else:
						tmp_file.write(unicode(line))
					try:
						line = file.next()
					except:
						proceed = 0
		os.remove(file_path)
		os.renames(tmp_file_path, file_path)


	def _read_tokens_in_file(self, file_path, tokens):
		'''
		Update measurement metadata dict with contents of file.
		tokens: Is a list. function looks for these tokens and extracts their txt. If list elements are strings....  
			Updates  measurement_metadata dict as measurement_metadata[token] = str(token_text_in_file)
			If list elements are tuples (token, measurement_metadata_key, format_function), then  updates 
			measurement_metadata dict as  measurement_metadata[measurement_metadata_key] = format_function(token_text_in_file) 
		'''
		#read the file
		file_contents = None
		with io.open(file_path, mode = 'r') as file:
			file_contents = file.read()

		token_loc = []
		token_text_EOL = '|'
		is_tuple = isinstance(tokens[0], tuple)
		if is_tuple:
			for token in tokens:
				pos = file_contents.lower().find(token[0].lower())
				if pos == -1:
					logging.error('Unable to find "{}" in file:{}'.format(token[0], file_path))
					continue
				token_loc.append((token[0], file_contents.lower().find(token[0].lower()), token[1], len(token[0]), token[2]))

		else:	
			for token in tokens:
				pos = file_contents.lower().find(token.lower())
				if pos == -1:
					logging.error('Unable to find "{}" in file:{}'.format(token, file_path))
					continue
				token_loc.append((token, file_contents.lower().find(token.lower()), token.replace(':','').replace(' ', '_'), len(token),str))
		
		token_loc.sort(cmp = lambda x,y: cmp(x[1], y[1]))
		token_loc_len = len(token_loc)


		for i in xrange(token_loc_len):
			end_index = None
			next_EOL_loc = file_contents[token_loc[i][1] + token_loc[i][3]:].find(token_text_EOL) 
			
			if i+1 ==  token_loc_len:
				end_index = next_EOL_loc + token_loc[i][1] + token_loc[i][3] if  (next_EOL_loc > -1) else None
			elif  (next_EOL_loc + token_loc[i][1] + token_loc[i][3] <  token_loc[i+1][1]) & (next_EOL_loc > -1):
				end_index = next_EOL_loc + token_loc[i][1] + token_loc[i][3]
			else:
				end_index = token_loc[i+1][1]
			#token_text  = file_contents[token_loc[i][1] + token_loc[i][3]:None if i+1 ==  token_loc_len else token_loc[i+1][1]]
			token_text  = file_contents[token_loc[i][1] + token_loc[i][3]:end_index]
			token_text  = token_text.strip('\n').strip()
			if token_text == '':
				logging.error('empty token text for "{}" in file {}'.format(token_loc[i][0],file_path ))
			self.measurement_metadata[token_loc[i][2]] = str(token_text) if token_loc[i][4] == None else  token_loc[i][4](token_text) 

	def _define_sweep_array(self,index,**field_names):
		#for field_name in self.sweep_data_columns.fields.keys():
		for field_name in field_names:
			self.Sweep_Array[field_name][index] = field_names[field_name]

	def _define_sweep_data_columns(self, fsteps = 1, tpoints = 1, fsteps_syn = 1, noise_spectrum_length = 1):
		self.measurement_metadata['Fsteps'] = fsteps
		self.measurement_metadata['Num_Temperatures']  = tpoints
		self.measurement_metadata['Num_Syn_Freq_Steps']  = fsteps_syn
		self.measurement_metadata['Noise_Spectrum_Length'] = noise_spectrum_length

		if tpoints < 1: # we dont want a shape = (0,) array. We want at least (1,)
			tpoints = 1

		if fsteps_syn < 1:
			fsteps_syn = 1

		if noise_spectrum_length < 1:
			noise_spectrum_length = 1

		self.sweep_data_columns = np.dtype([
			("Fstart"         			, np.float64), # in Hz
			("Fstop"          			, np.float64), # in Hz
			("Heater_Voltage" 			, np.float64), # in Volts
			("Pinput_dB"      			, np.float64), # in dB
			#("Preadout_dB"     			, np.float64), # in dB  - The power at the input of the resonator, not inside the resonator
			("Thermometer_Voltage_Bias"	, np.float64), # in Volts
			("Temperature_Readings"    	, np.float64,(tpoints,)), # in Kelvin
			("Temperature"		    	, np.float64), # in Kelvin
			("S21"            			, np.complex128, (fsteps,)), # in complex numbers, experimental values.
			("Frequencies"    			, np.float64,(fsteps,)), # in Hz
			("Is_Valid"					, np.bool),
			("Aux_Voltage"				, np.float64), # in Volts
			("Aux_Value"				, np.float64), # value of aux signal due to Aux_Voltage, e.g. B-field in gauss if Helmholtz coil is on aux channel
			("S21_Syn"            		, np.complex128, (fsteps_syn,)), # in complex numbers
			("Frequencies_Syn"    		, np.float64,(fsteps_syn,)), # in Hz
			("Noise_Spectrum"			, np.float64, (noise_spectrum_length,)), # V/sqrt(hz)
			("Frequencies_Noise"    	, np.float64,(noise_spectrum_length,)), # in Hz, the  frequencies of the noise spectrum
			("Power_Baseline_Noise"    	, np.float64), # dBm -  off resonnce power entering digitizer for noise spectrum measurement
			("On_Res_Noise_Freq"    	, np.float64), # Hz - freq at which on res noise was taken
			("Off_Res_Noise_Freq"    	, np.float64), # Hz - freq at which off res noise was taken
			#auto and cross correlated noise
			#phase cancellation value for carrier cuppresion - np.float64,(fsteps_syn,)
			#ampl cancellation value for carrir suppression -  np.float64,(fsteps_syn,)
			("Scan_Timestamp"			, np.uint64),
			])
	
	def save_hf5(self, filename , overwrite = False):
		'''Saves current self.Sweep_Array into table contained in the hdf5 file speficied by filename.
		If overwite = True, self.Sweep_Array will overwright whatever is previous table data there is.

		group_name : is the type of sweep. can be something like SweepANPT, SweepPT, SweepA, etc
			A - Aux channel is swept
			N - Noise is taken
			P - Power is swept
			T - Temperature is swept

		Tables are then named T[Date]_[groupname] where date is in the form '%Y%m%d%H%M'

		'''
		
		if not os.path.isfile(filename):
			self._print('Speficied h5 database does not exist. Creating new one.')
			pos = filename.find('/')
			if pos >= 0:
				try:
					os.makedirs(filename[0:pos+1])
				except OSError:
					self._print('{0} exists...'.format(filename[0:pos+1]))
			wmode = 'w'
		else:
			self._print('Speficied h5 database exists and will be updated.')
			wmode = 'a'
			
		db_title = 'Run {} Data'.format(self.measurement_metadata['Run'])
		
		group_name = 'Sweep{0}{1}{2}{3}'.format('A' if self.num_aux_sweep_pts > 1 else '',
												'N' if self.num_syn_freq_points > 1 else '',
												'P' if self.num_power_sweep_pts > 1 else '',
												'T' if self.num_temp_sweep_pts > 1 else '')
		group_title = 'Sweep of {0}{1}{2}{3}'.format(	'Aux input,' if self.num_aux_sweep_pts > 1 else '',
														' Noise,' if self.num_syn_freq_points > 1 else '',
														' Power,' if self.num_power_sweep_pts > 1 else '',
														' and Temperature' if self.num_temp_sweep_pts > 1 else '')
		group_title = 'Survey' if group_title == 'Sweep of ' else  group_title


		sweep_data_table_name = 'T' + self.measurement_start_time + '_' + group_name
		if self.current_sweep_data_table == None:
			table_path = '/' + group_name + '/' + sweep_data_table_name
			self.current_sweep_data_table = table_path
		table_path = self.current_sweep_data_table
		
		with tables.open_file(filename, mode = wmode, title = db_title ) as fileh:
			try:
				sweep_data_table = fileh.get_node(table_path)
				self._print('Adding data to existing table {0} exists.'.format(table_path))
			except:
				self._print('Creating new table {0} and adding data.'.format(table_path))
				sweep_data_table = fileh.create_table('/'+ group_name,sweep_data_table_name,description=self.sweep_data_columns,title = 'Sweep Data Table',filters=tables.Filters(0), createparents=True)
			
			# copy Sweep_Array to sweep_data_table
			sweep_data_table.append(self.Sweep_Array)

			# Save metadata
			for data in self.measurement_metadata.keys(): 
				exec('sweep_data_table.attrs.{0} = self.measurement_metadata["{0}"]'.format(data))
				#should  use get attrs function but cant make it work... sweep_data_table.__setattr__(data, self.measurement_metadata[data])
				if self.measurement_metadata[data] == None:
					self._print('table metadata {0} not defined and is set to None'.format(data))
			sweep_data_table.attrs.keys =  self.measurement_metadata.keys() #NOTE: we record the metadata keys in the metadate
			sweep_data_table.flush()
		
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

	def _record_LNA_settings(self):
		self.psu = Instruments.power_supply(self.inl.POWER_SUPPLY_HPE3631A)
		try:
			self.measurement_metadata['LNA'].update(Vg = self.psu.get_Vg(),
													Vd = self.psu.get_Vd(), 
													Id = self.psu.get_Id()) 
		finally:
			del(self.psu)

	def _perform_subscans(self):
		'''
		Must be called after self.na is instantiated
		and after self.measurement_metadata has been updated to incorporate scan parameters. 
		'''
		num_points_per_na_scan = self.measurement_metadata['Num_Points_Per_NA_Scan'] 
		num_f_steps = self.measurement_metadata['Fsteps']
		scan_bw = self.scan_bw
		num_sub_scans = np.int(num_f_steps/np.float64(num_points_per_na_scan))
		sub_scan_width = (scan_bw/num_f_steps)*num_points_per_na_scan
		model_freq_array = np.linspace(self.measurement_metadata['Frequency_Range'][0],self.measurement_metadata['Frequency_Range'][1], num = num_f_steps,dtype=np.float64)
		
		for i in xrange(num_sub_scans):
			self.na.set_start_freq(model_freq_array[num_points_per_na_scan*i])
			self.na.set_stop_freq(model_freq_array[num_points_per_na_scan*(i+1)-1])			
			self.na.trigger_single_scan()
			frequencies, s_parameters = self.na.read_scan_data()
			self.Sweep_Array[0]['Frequencies'][num_points_per_na_scan*i: num_points_per_na_scan*(i+1)] = frequencies
			self.Sweep_Array[0]['S21'][num_points_per_na_scan*i: num_points_per_na_scan*(i+1)] = s_parameters
		del(model_freq_array)

	@contextlib.contextmanager
	def _setup_na_for_scans(self):
		'''
		a context manager for creating and deleting the network analyzer object, modifying its state for the  scan and then returnig it to continuous scan mode.
		'''
		min_number_of_na_scan_pts = np.float64(100) #use a nice rounds number
		avg_fac = self.measurement_metadata['NA_Average_Factor']
		IFBW = self.measurement_metadata['IFBW']
		min_resolution = self.measurement_metadata['Min_Freq_Resolution']
		scan_bw = self.scan_bw
		min_num_scan_pts_req = np.ceil(scan_bw/min_resolution)

		
		def round_up(x):
			return np.int(np.ceil(x/min_number_of_na_scan_pts)) * min_number_of_na_scan_pts

		def round_down(x):
			return np.int(np.floor(x/min_number_of_na_scan_pts)) * min_number_of_na_scan_pts

		self.na = Instruments.network_analyzer(self.inl.NETWORK_ANALYZER_E5071B)
		max_na_points_per_scan = np.float64(self.na.max_na_scan_points)
		if  min_number_of_na_scan_pts >= min_num_scan_pts_req: 
			num_points_per_na_scan = min_number_of_na_scan_pts
			num_f_steps = num_points_per_na_scan
			num_sub_scans = 1
		elif (min_num_scan_pts_req > min_number_of_na_scan_pts) & (min_num_scan_pts_req <= max_na_points_per_scan): # one scan needed
			num_points_per_na_scan = min_num_scan_pts_req
			num_f_steps = num_points_per_na_scan
			num_sub_scans = 1
		else: 
			num_f_steps = round_up(min_num_scan_pts_req)
			num_points_per_na_scan = fractions.gcd(round_down(max_na_points_per_scan), num_f_steps)
			num_sub_scans = np.int(num_f_steps/np.float64(num_points_per_na_scan))
			print('Total num freq pts: {}. Number pts per scan {}. Number of scans {}'.format(num_f_steps,num_points_per_na_scan,num_sub_scans ))

		sub_scan_width = (scan_bw/num_f_steps)*num_points_per_na_scan
		self.measurement_metadata['Num_Points_Per_NA_Scan'] = num_points_per_na_scan
		

		self._define_sweep_data_columns( fsteps = num_f_steps, 
									tpoints = self.measurement_metadata['Num_Temp_Readings'],
									fsteps_syn = self.num_syn_freq_points if self.num_syn_freq_points >0 else 1, 
									noise_spectrum_length = self.num_noise_spec_freq_points)


		try:
			self.na.set_num_points_per_scan(num_points_per_na_scan)
			self.na.set_IFBW(IFBW)
			if avg_fac  > 0:
				self.na.turn_on_averaging(avg_fac, channel = 1)
			self.na.query_sweep_time( update_timout = True)
			#self.na.set_start_freq(self.Frequency_Range[0])
			#self.na.set_stop_freq(self.Frequency_Range[1])
			self.na.setup_single_scan_mode()
			yield self.na 
			if avg_fac  > 0:
				self.na.turn_off_averaging(channel = 1)
			self.na.setup_continuous_scan_mode( channel = 1)
		finally:
			del(self.na)



	def execute_sweep(self, sweep_parameter_file, database_filename, use_table = None, End_Heater_Voltage = None):
		'''
		Reads the sweep_parameter_file located in data_directory_path and reads in swee parameters.

		The sweep paramater file is a txt with the following information
		Define the following 
		Frequency_Range: | [Start_Hz, Stop_Hz] <- Brackets mean list
		Min_Resolution: | Hz
		IFBW: | Hz
		Average_Factor:
		Heater_Voltage:  | [Volts]
		Thermometer_Bias: | [Volts]
		Aux_Voltage: | [Volts]
		Delay_Time: 0 | Seconds - Time to wait before starting measurement. 0 if no delay..
		Pause:	0 | Seconds - Time to wait between changing heater voltage
		Powers:  | [dBm]
		Atten_NA_Output: | dB
		Atten_NA_Input: | dB
		Atten_RTAmp_Input: | dB
		RTAmp_In_Use:  | 0 or 1
		Device_Themometer_Channel: 'MF3' | Can be 'MF1', 'MF2', or 'MF3'
		Num_Temp_Readings: 10 |
		Noise_Sweep_Num_Points: 0 |
		Sweep_Data_Table: |

		'''
		if use_table != None:
			self.current_sweep_data_table = use_table

		sweep_parameter_file = self.data_directory_path + os.sep + sweep_parameter_file
		database_filename = self.data_directory_path + os.sep + database_filename

		self.fi = Fridge_Interfaces.fridge_interface(self.devices)
		self.fi.suspend()

		update_token = 'Sweep_Data_Table' #line to updated in sweep_parameter_file
	
		
		self.num_noise_spec_freq_points = 1

		tokens = 	[('Powers:','Powers', eval), ('Min_Resolution:','Min_Freq_Resolution', np.float), ('IFBW:','IFBW', np.float),
					('Heater_Voltage:','Heater_Voltage',eval), ('Aux_Voltage:','Aux_Voltage',eval), ('Average_Factor:','NA_Average_Factor', np.int), 
					('Pause:','Wait_Time',np.float), ('RTAmp_In_Use:', 'RTAmp_In_Use', np.int),  ('Noise_Sweep_Num_Points:','Noise_Sweep_Num_Points',np.int),
					('Atten_NA_Output:', 'Atten_NA_Output',np.float32), ('Atten_NA_Input:','Atten_NA_Input',np.float32),
					('Atten_RTAmp_Input:','Atten_RTAmp_Input',np.float32),  ('Thermometer_Bias:','Thermometer_Voltage_Bias', eval),
					('Frequency_Range:','Frequency_Range',eval), ('Delay_Time:','Delay_Time',np.float),
					('Device_Themometer_Channel:', 'Device_Themometer_Channel',str), ('Num_Temp_Readings:','Num_Temp_Readings',np.int)]
		
		self._read_tokens_in_file(sweep_parameter_file, tokens)

		thermometer_channel_name = self.measurement_metadata['Device_Themometer_Channel']
		num_temp_readings = self.measurement_metadata['Num_Temp_Readings'] 
		self.Delay_Time = self.measurement_metadata.pop('Delay_Time')
		self.Aux_Voltage = self.measurement_metadata.pop('Aux_Voltage')
		self.Powers = self.measurement_metadata.pop('Powers')
		self.Heater_Voltage = self.measurement_metadata.pop('Heater_Voltage')
		self.Thermometer_Voltage_Bias = self.measurement_metadata.pop('Thermometer_Voltage_Bias')
		#self.Frequency_Range  =  self.measurement_metadata.pop('Frequency_Range')
		self.num_syn_freq_points = self.measurement_metadata['Noise_Sweep_Num_Points']
		self.num_aux_sweep_pts   = len(self.Aux_Voltage)
		self.num_power_sweep_pts = len(self.Powers) 
		self.num_temp_sweep_pts =  len(self.Heater_Voltage) 
		self.scan_bw = np.float64(self.measurement_metadata['Frequency_Range'][1]) - np.float64(self.measurement_metadata['Frequency_Range'][0])

		self.measurement_start_time = datetime.datetime.now().strftime('%Y%m%d%H%M')
		self.measurement_metadata['Measurement_Start_Time'] = self.measurement_start_time
		self._record_LNA_settings()






		
		if self.Delay_Time != np.float(0):
			time.sleep(self.Delay_Time)
			self._print('Executing delay time before taking data. first scan begins at {}'.format((datetime.datetime.now() + datetime.timedelta(seconds = self.Delay_Time)).strftime( '%m/%d/%Y %H:%M:%S')))



		if self.Show_Plots:
			plt.rcParams["axes.titlesize"] = 10
			fig = plt.figure( figsize=(8, 6), dpi=100)
			ax = fig.add_subplot(111)


		with self._setup_na_for_scans():
			t_index = 0
			for t in self.Heater_Voltage:
				self.fi.resume()
				self.fi.set_stack_heater_voltage(t)
				self.fi.set_thermometer_bias_voltage(self.Thermometer_Voltage_Bias[t_index])
				self.fi.suspend()

				if (t_index != 0) & (self.measurement_metadata['Wait_Time'] != np.float(0)):
					self.na.off()
					time.sleep(self.measurement_metadata['Wait_Time'])
					self._print('Temp Stability Pause. Network analyzer stimulus off. Next scan at {}'.format((datetime.datetime.now() + datetime.timedelta(seconds = self.measurement_metadata['Wait_Time'])).strftime( '%m/%d/%Y %H:%M:%S')))
					self.na.on()

				p_index = 0	
				for p in self.Powers:
					self.na.set_power(p)

					a_index = 0
					for a in self.Aux_Voltage:
						self.fi.resume()
						self.fi.set_aux_channel_voltage(a)
						self.fi.suspend()


						self._print('Executing scan: Temp {0} of {1}, Pow {2} of {3}, Aux {4} of {5}'.format(t_index+1,self.num_temp_sweep_pts,p_index+1,self.num_power_sweep_pts, a_index+1,self.num_aux_sweep_pts))					
						
						self.Sweep_Array = np.zeros(1, dtype = self.sweep_data_columns)

						######
						#
						# Perform sub scans: and add data to Sweep_Array
						#
						######
						self._perform_subscans()
						
						

						######
						#
						# Switch to noise measurement and execute
						#
						######
						
						self.fi.resume()
						temperature_readings =  np.empty((num_temp_readings,), np.float64)
						for i in xrange( temperature_readings.size):
							temperature_readings[i] = self.fi.read_temp(thermometer_channel_name)
						self.fi.suspend()

						self._define_sweep_array(0,  #The  array  is only 1 recode long and this record is index 0
												Fstart = self.Sweep_Array[0]['Frequencies'][0],
												Fstop = self.Sweep_Array[0]['Frequencies'][-1],
												Heater_Voltage = t,
												Temperature_Readings = temperature_readings,
												Temperature = np.median(temperature_readings),
												Thermometer_Voltage_Bias = self.Thermometer_Voltage_Bias[t_index],#set to zero unless there is an array of temps in the ScanData
												Pinput_dB = p, #we only want the power coming out of the source, i.e. the NA
												#S21 = s_parameters,
												#Frequencies = frequencies,
												Is_Valid = True,
												Aux_Voltage = a,
												Scan_Timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M')
												)


											# ("Aux_Value"				, np.float64), # value of aux signal due to Aux_Voltage, e.g. B-field in gauss if Helmholtz coil is on aux channel
											# ("S21_Syn"            		, np.complex128, (fsteps_syn,)), # in complex numbers
											# ("Frequencies_Syn"    		, np.float64,(fsteps_syn,)), # in Hz
											# ("Noise_Spectrum"			, np.float64, (noise_spectrum_length,)), # V/sqrt(hz)
											# ("Frequencies_Noise"    	, np.float64,(noise_spectrum_length,)), # in Hz, the  frequencies of the noise spectrum
											# ("Power_Baseline_Noise"    	, np.float64), # dBm -  off resonnce power entering digitizer for noise spectrum measurement
											# ("On_Res_Noise_Freq"    	, np.float64), # Hz - freq at which on res noise was taken
											# ("Off_Res_Noise_Freq"    	, np.float64), # Hz - freq at which off res noise was taken

						if self.Show_Plots: # issue '%matplotlib qt' in spyder?
							#plt.rcParams["axes.titlesize"] = 10
							#fig = plt.figure( figsize=(8, 6), dpi=100)
							ax.clear()
							
							line = ax.plot(self.Sweep_Array[0]['Frequencies'],20*np.log10(np.abs(self.Sweep_Array[0]['S21'])),'b-',)
							ax.set_xlabel('Frequency [Hz]')
							ax.set_ylabel('$20*Log_{10}[|S_{21}|]$ [dB]')

							ax.set_title('Run: {0}; Sensor: {1}; Ground: {2}; Power {3} dBm'.format(self.measurement_metadata['Run'], 
																									self.measurement_metadata['Sensor'], 
																									self.measurement_metadata['Ground_Plane'], 
																									p))		
							#ax.legend(loc = 'best', fontsize = 9)
							ax.grid()
							#plt.draw()
							plt.show()

						
						self.save_hf5(database_filename , overwrite = False)
						a_index = a_index +1
					p_index = p_index +1		
				t_index = t_index + 1
		self._update_file(sweep_parameter_file, update_token, self.current_sweep_data_table)
		self.current_sweep_data_table = None
		
		if  End_Heater_Voltage != None:
			self.fi.resume()
			self.fi.set_stack_heater_voltage(End_Heater_Voltage)
			self.fi.suspend()
		del(self.fi)
		#na.setup_continuous_scan_mode()
		#del(self.na) # handeled in context manager

	def get_current_na_frequency_range(self):
		self.na = Instruments.network_analyzer(self.inl.NETWORK_ANALYZER_E5071B)
		start = self.na.get_start_freq()
		stop = self.na.get_stop_freq()
		num_pts = np.float64(self.na.get_num_points_per_scan())
		del(self.na)
		freq_resolution = (stop - start)/num_pts
		print('Start: {}    Stop: {}    Freq Resolution: {}'.format(start,stop, freq_resolution))

[-55., -54., -53., -52., -51., -50., -49., -48., -47., -46., -45., -44., -43., -42., -41., -40., -39., -38., -37., -36., -35., -34.,-33., -32., -31., -30., -29., -28., -27., -26., -25., -24., -23., -22., -21., -20., -19., -18., -17., -16., -15., -14., -13., -12.,
       -11., -10.,  -9.,  -8.,  -7.,  -6.,  -5.,  -4.,  -3.,  -2.,  -1.,
         0.]