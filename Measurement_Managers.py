import Instruments
import Fridge_Interfaces
import os, io, datetime
import logging
import numpy as np
import tables

data_directory_prefix = 'Z:\\user\\miguel\\KIPS_Data'

class measurement_manager:

	def __init__(self, devices, data_dir):
		'''
		data_dir is the run name
		NOTE: Change Thermometer_Configuration in KAM to Devices,  add measurement_start_time in KAM
		'''
		#self.fi = Fridge_Interfaces.fridge_interface(devices)
		self.inl = Instruments.Instrument_Name_List
		
		
		self.start_directory = os.getcwd()
		self.data_directory_path = data_directory_prefix + os.sep + data_dir
		
		if os.path.exists(self.data_directory_path) == False:
			logging.error('data directory path does no exist.')

		self.measurement_metadata = {'Devices': devices}
		self._read_about_file()

		self.is_aux_sweep = False
		self.is_noise_sweep = False
		self.is_power_sweep = False
		self.is_temp_sweep =  False
		self.measurement_start_time = None  #datetime.datetime.now().strftime('%Y%m%d%H%M')
		self.current_sweep_data_table = None

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



	def _read_tokens_in_file(self, file_path, tokens):
		'''
		Update measurement metadata dict with contents of file.
		tokens: Is a list. function looks for these tokens and extracts their txt. If list elements are strings....  
			Updates  measurement_metadata dict as measurement_metadata[token] = str(token_text_in_file)
			If list elements are tuples (token, measurement_metadata_key, format_function), then  updates 
			measurement_metadata dict as  measurement_metadata[measurement_metadata_key] = format_function(token_text_in_file) 
		'''
		#read the about file
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
			
		db_title = 'Run {} Data'.format(self.measurement_metadata['Run'])
		
		group_name = 'Sweep{0}{1}{2}{3}'.format('A' if self.is_aux_sweep == True else '',
												'N' if self.is_noise_sweep == True else '',
												'P' if self.is_power_sweep == True else '',
												'T' if self.is_temp_sweep == True else '')
		group_title = 'Sweep of {0}{1}{2}{3}'.format(	'Aux input,' if self.is_aux_sweep == True else '',
														' Noise,' if self.is_noise_sweep == True else '',
														' Power,' if self.is_power_sweep == True else '',
														' and Temperature' if self.is_temp_sweep == True else '')
		group_title = 'Survey' if group_title == 'Sweep of ' else  group_title


		sweep_data_table_name = 'T' + self.measurement_start_time + '_' + group_name
		table_path = '/' + group_name + '/' + sweep_data_table_name
		self.current_sweep_data_table = table_path

		
		with tables.open_file(filename, mode = wmode, title = db_title ) as fileh:
			try:
				sweep_data_table = fileh.get_node(table_path)
				print('Adding data to existing table {0} exists.'.format(table_path))
			except:
				print('Creating new table {0} and adding data.'.format(table_path))
				sweep_data_table = fileh.create_table('/'+ group_name,sweep_data_table_name,description=self.sweep_data_columns,title = 'Sweep Data Table',filters=tables.Filters(0), createparents=True)
			
			# copy Sweep_Array to sweep_data_table
			sweep_data_table.append(self.Sweep_Array)

			# Save metadata
			for data in self.measurement_metadata.keys(): 
				exec('sweep_data_table.attrs.{0} = self.measurement_metadata["{0}"]'.format(data))
				#should  use get attrs function but cant make it work... sweep_data_table.__setattr__(data, self.measurement_metadata[data])
				if self.measurement_metadata[data] == None:
					print('table metadata {0} not defined and is set to None'.format(data))	
			sweep_data_table.flush()	

	def _record_LNA_settings(self):
		self.psu = Instruments.power_supply('GPIB0::14::INSTR')

		self.measurement_metadata['LNA'].update(Vg = self.psu.get_Vg(),
												Vd = self.psu.get_Vd(), 
												Id = self.psu.get_Id()) 
		self.psu.close()


	def single_scan(self, parameter_filename, database_filename):
		'''

		Define the following.....
		Frequency_Range: | [Hz Hz]
		Min_Resolution: | Hz
		IFBW: | Hz
		Average_Factor:
		Heater_Voltage:  | Volts
		Thermometer_Bias: | Volts
		Aux_Voltage: | Volts 
		Delay_Time: 0 | Seconds - Time to wait before starting measurement. 0 if no delay..
		Pause:	0 | Seconds - Time to wait between changing heater voltage
		Powers: [] | dBm
		Atten_NA_Output: | dB
		Atten_NA_Input: | dB
		Atten_RTAmp_Input: | dB
		RTAmp_In_Use:  | 0 or 1
		Sweep_Data_Table: |

		'''
		parameter_filename = self.data_directory_path + os.sep+ parameter_filename
		database_filename = self.data_directory_path + os.sep+ database_filename
		self.na = na = Instruments.network_analyzer(self.inl.NETWORK_ANALYZER_E5071B)
		num_points_per_scan = 800
		tokens = 	[('Powers:','Powers', eval), ('Min_Resolution:','Min_Freq_Resolution', np.float), ('IFBW:','IFBW', np.float),
					('Heater_Voltage:','Heater_Voltage',eval), ('Aux_Voltage:','Aux_Voltage',eval), ('Average_Factor:','NA_Average_Factor', np.int), 
					('Pause:','Wait_Time',np.float), ('RTAmp_In_Use:', 'RTAmp_In_Use', np.int),  ('Noise_Sweep_Num_Points:','Noise_Sweep_Num_Points',np.int),
					('Atten_NA_Output:', 'Atten_NA_Output',np.float32), ('Atten_NA_Input:','Atten_NA_Input',np.float32),
					('Atten_RTAmp_Input:','Atten_RTAmp_Input',np.float32),  ('Thermometer_Bias:','Thermometer_Voltage_Bias', eval),
					('Frequency_Range:','Frequency_Range',eval)]
		
		self._read_tokens_in_file(parameter_filename, tokens)

		self.Aux_Voltage = self.measurement_metadata.pop('Aux_Voltage')
		self.Powers = self.measurement_metadata.pop('Powers')
		self.Heater_Voltage = self.measurement_metadata.pop('Heater_Voltage')
		self.Thermometer_Voltage_Bias = self.measurement_metadata.pop('Thermometer_Voltage_Bias')
		self.Frequency_Range  =  self.measurement_metadata.pop('Frequency_Range')

		self.is_aux_sweep   = False if len(self.Aux_Voltage) < 2 else True 
		self.is_noise_sweep = False if self.measurement_metadata['Noise_Sweep_Num_Points'] == 0  else True
		self.is_power_sweep = False if len(self.Powers) < 2 else True 
		self.is_temp_sweep =  False if len(self.Heater_Voltage) < 2 else True 

		self.measurement_start_time = datetime.datetime.now().strftime('%Y%m%d%H%M')
		self.measurement_metadata['Measurement_Start_Time'] = self.measurement_start_time
		self._record_LNA_settings()
		self._define_sweep_data_columns( fsteps = num_points_per_scan, tpoints = len(self.Heater_Voltage), fsteps_syn = self.measurement_metadata['Noise_Sweep_Num_Points'], noise_spectrum_length = 1)
		self.Sweep_Array = np.zeros(1, dtype = self.sweep_data_columns)

		na.set_num_points_per_scan(num_points_per_scan)
		na.set_start_freq(self.Frequency_Range[0])
		na.set_stop_freq(self.Frequency_Range[1])
		na.set_power(self.Powers[0])
		na.set_IFBW(self.measurement_metadata['IFBW'])
		na.query_sweep_time()
		na.setup_single_scan_mode()
		na.trigger_single_scan()
		frequencies, s_parameters = na.read_scan_data()
		
		i = 0
		self._define_sweep_array(i, 
								Fstart = frequencies[0],
								Fstop = frequencies[1],
								Heater_Voltage = self.Heater_Voltage[i],
								Thermometer_Voltage_Bias = self.Thermometer_Voltage_Bias[0],#set to zero unless there is an array of temps in the ScanData
								Pinput_dB = self.Powers[0], #we only want the power coming out of the source, i.e. the NA
								S21 =s_parameters ,
								Frequencies = frequencies,
								#Temperature_Readings = sweep[3].squeeze()[()] if (sweep.size > 3) and (np.shape(sweep[3].squeeze()[()])[0] != 0) else np.array([0]), #set to zero unless there is an array of temps in the ScanData
								Is_Valid = True)
		self.save_hf5(database_filename , overwrite = False)
		#self.current_sweep_data_table = None
		# ('Num_Points_Per_Scan','Num_Points_Per_Scan',np.float)
		# ('LNA', 'LNA:LNA', str), ('HEMT', 'LNA:Vg', str,'Vg'),
		# ('HEMT', 'LNA:Id', str,'Id'),  ('HEMT', 'LNA:Vd', str,'Vd')
	


# self.Sweep_Array = np.zeros(self.metadata.Heater_Voltage.shape[0]*self.metadata.Powers.shape[0]*self.metadata.Freq_Range.shape[0], dtype = self.sweep_data_columns)
# 	def _define_sweep_data_columns(self, fsteps, tpoints):
# 		self.metadata.Fsteps = fsteps
# 		self.metadata.Num_Temperatures  = tpoints

# 		if tpoints < 1: # we dont want a shape = (0,) array. We want at least (1,)
# 			tpoints = 1

# 		self.sweep_data_columns = np.dtype([
# 			("Fstart"         			, np.float64), # in Hz
# 			("Fstop"          			, np.float64), # in Hz
# 			("Heater_Voltage" 			, np.float64), # in Volts
# 			("Pinput_dB"      			, np.float64), # in dB
# 			("Preadout_dB"     			, np.float64), # in dB  - The power at the input of the resonator, not inside the resonator
# 			("Thermometer_Voltage_Bias"	, np.float64), # in Volts
# 			("Temperature_Readings"    	, np.float64,(tpoints,)), # in Kelvin
# 			("Temperature"		    	, np.float64), # in Kelvin
# 			("S21"            			, np.complex128, (fsteps,)), # in complex numbers, experimental values.
# 			("Frequencies"    			, np.float64,(fsteps,)), # in Hz
# 			("Q"						, np.float64),
# 			("Qc"						, np.float64),
# 			("Fr"						, np.float64), # in Hz
# 			("Is_Valid"					, np.bool),
# 			("Chi_Squared"              , np.float64),
# 			("Mask"						, np.bool,(fsteps,)), # array mask selecting data used in phase fit
# 			("R"						, np.float64), #outer loop radius
# 			("r"						, np.float64), # resonance loop radius	
# 			("a"						, np.float64),	
# 			("b"						, np.float64),
# 			#("Normalization"			, np.float64),
# 			("Theta"					, np.float64),
# 			("Phi"						, np.float64),
# 			("cQ"						, np.float64),
# 			("cQc"						, np.float64),
# 			("cFr"						, np.float64), # in Hz
# 			("cIs_Valid"				, np.bool),
# 			("cChi_Squared"             , np.float64),
# 			("cPhi"						, np.float64),
# 			("cTheta"					, np.float64),
# 			("cR"						, np.float64),
# 			("sQ"						, np.float64),
# 			("sQc"						, np.float64),
# 			("sFr"						, np.float64), # in Hz
# 			("sIs_Valid"				, np.bool),
# 			("sChi_Squared"             , np.float64),
# 			("sPhi"						, np.float64),
# 			("sTheta"					, np.float64),
# 			("sR"						, np.float64),

# 			#("S21_Processed"            , np.complex128, (fsteps,)), # Processed S21 used in phase fit 
# 			])
