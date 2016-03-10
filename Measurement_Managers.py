import Instruments
import Fridge_Interfaces
import os, io, datetime, time
import logging
import numpy as np # developed with version 1.10.4
import scipy as sp #developed with version 0.17.0
from scipy import signal 
import tables
import matplotlib as mpl
import matplotlib.pyplot as plt
import contextlib
import fractions
import KAM


data_directory_prefix = 'Z:\\user\\miguel\\KIPS_Data'

class measurement_manager:

	def __init__(self, devices, data_dir):
		'''
		data_dir is the run name
		NOTE: Change Thermometer_Configuration in KAM to Devices,  add measurement_start_time in KAM
		'''
		self.mssg = Instruments.messaging()
		self.mssg.Verbose = True

		self.Show_Plots = True
		self.devices = devices
		inl = Instruments.Instrument_Name_List
		self.na_address = inl.NETWORK_ANALYZER_E5071B
		self.lna_psu_address = inl.POWER_SUPPLY_HPE3631A 
		self.source_meter_address  = inl.SOURCE_METER_2612A
		self.syn_address = inl.SYNTHESIZER_MG3692B
		self.iqd_address = inl.DIGITIZER_NI6120 # iqd = I Q DAQ, i.e the NI6120
		self.atn_address = inl.ATTENUATOR_BOX_8310
		self.na_to_atn_conversion_func = lambda p_na : np.abs(p_na-8) # Converts a NA Pinput_dB to an Atn attenuation setting
		

		self.swp = KAM.sweep()
		self.swp.metadata.Electrical_Delay = 100e-9

		self.start_directory = os.getcwd()
		self.data_directory_path = data_directory_prefix + os.sep + data_dir
		
		if os.path.exists(self.data_directory_path) == False:
			logging.error('data directory path does no exist.')

		self.measurement_metadata = {'Devices': devices, 'Device_Themometer_Channel': None}
		self.measurement_metadata['Synthesizer_Scan_Num_BW'] = 8
		self.measurement_metadata['Synthesizer_Scan_Power'] = 16 
		self.measurement_metadata['Synthesizer_Frequency_Spacing'] = 'Linear' #The LO power of the syn used for all syn sweep points
		#self.total_attenuation = 62 # sum of chan 1 (input to fridge) and chan 2 (output of fridge) attenuator box 
		self.measurement_metadata['IQ_Sample_Rate'] = 8.0e5
		self.bkgd_loop_num_pts = 10
		self.measurement_metadata['Noise_Sample_Rate'] = noise_sampling_rate = 8.0e5
		self.measurement_metadata['Noise_Decimation_Factor'] = decimation_factor =  4.	
		decimated_noise_sampleing_rate = noise_sampling_rate/decimation_factor
		self.measurement_metadata['Noise_Integration_Time'] = 1 # second
		self.measurement_metadata['Noise_Num_Integrations'] = 10
		self.measurement_metadata['Noise_Frequency_Segmentation'] =[100., 5000., decimated_noise_sampleing_rate/2.0]
		self.measurement_metadata['Noise_Frequency_Resolution_Per_Segment'] = [1., 10., 100.]
		self.measurement_metadata['Off_Res_Freq_Offset'] = 3e6 # Hertz
		#NOTE: noise_spectrum_length = noise_sample_rate * noise_integration_time *0.1 / noise_decimation_factor
		#      Must be an integer 

		self.measurement_metadata['System_Calibration'] = None
		self.measurement_metadata['Cable_Calibration'] = None
		self.measurement_metadata['Electrical_Delay'] = None
		self.measurement_metadata['RTAmp'] = 'AML016P3411'

		self._read_about_file()

		self.num_aux_sweep_pts   = 1
		self.num_syn_freq_points =  1
		self.num_power_sweep_pts = 1
		self.num_temp_sweep_pts =  1
		self.measurement_start_time = None  #datetime.datetime.now().strftime('%Y%m%d%H%M')
		self.current_sweep_data_table = None #path within hdf5 file to current data table 
		
		self.min_syn_freq_points = 25


	def __del__(self):
		pass


	def _restore_na_settings(self):
			'''
			Must be executed within _na_ctx()
			'''
			if self.measurement_metadata['NA_Average_Factor']  > 0:
				self.na.turn_off_averaging(channel = 1)
			self.na.setup_continuous_scan_mode( channel = 1)

	@contextlib.contextmanager
	def _na_ctx(self):
		self.na = Instruments.network_analyzer(self.na_address)
		try:
			yield self.na
		except: #Return na to continuous scan mode and raise error
			self._restore_na_settings()
			raise
		finally:
			del(self.na)
		# if getattr(self, 'na', None) is not None:
		# 		del(self.na)

	@contextlib.contextmanager
	def _fi_ctx(self):
		self.fi = Fridge_Interfaces.fridge_interface(self.devices)
		try:
			yield self.fi
		finally:
			del(self.fi)

	@contextlib.contextmanager
	def _syn_ctx(self):
		self.syn = Instruments.synthesizer(self.syn_address)
		try:
			yield self.syn
		finally:
			del(self.syn)

	@contextlib.contextmanager
	def _sm_ctx(self):
		self.sm = Instruments.source_meter(self.source_meter_address)
		try:
			yield self.sm
		finally:
			del(self.sm)

	@contextlib.contextmanager
	def _psu_ctx(self):
		self.psu = Instruments.power_supply(self.lna_psu_address)
		try:
			yield self.psu
		finally:
			del(self.psu)

	@contextlib.contextmanager
	def _iqd_ctx(self):
		self.iqd = Instruments.ni_digitizer(self.iqd_address)
		try:
			yield self.iqd
		finally:
			del(self.iqd)

	@contextlib.contextmanager
	def _atn_ctx(self):
		self.atn = Instruments.attenuator_box(self.atn_address)
		try:
			yield self.atn
		finally:
			del(self.atn)

	@contextlib.contextmanager
	def _digitizer_ctx(self):
		self.switch_to_noise_channel()
		try:
			yield
		except:
			with self._na_ctx():
				self._restore_na_settings()
			raise
		finally:
			self.switch_to_na_channel()


	def _print(self, mssg, nnl = False):
		self.mssg._print(mssg, nnl = nnl)




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

	def _define_sweep_data_columns(self, fsteps = 1, tpoints = 1, fsteps_syn = 1):
		self.measurement_metadata['NA_Scan_Num_Points'] = fsteps
		self.measurement_metadata['Num_Temperatures']  = tpoints
		self.measurement_metadata['Synthesizer_Scan_Num_Points']  = fsteps_syn


		bkgd_loop_num_pts = self.bkgd_loop_num_pts

		if self.measurement_metadata['Measure_On_Res_Noise']:
			noise_spectrum_length = self._compute_noise_spectrum_length()
		else:
			noise_spectrum_length = 1

		# off resonance noise vector will be the same length as the on resoance noise vectors, 
		# unless no off resonance noise it to be measured. In which case, the off resoance noise 
		# are of length 1
		if self.measurement_metadata['Measure_Off_Res_Noise']:
			noise_spectrum_length_off_res = noise_spectrum_length
		else:
			noise_spectrum_length_off_res = 1

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
			("Bkgd_Freq_Syn"    		, np.float64,(bkgd_loop_num_pts,)), # in Hz
			("Bkgd_S21_Syn"            	, np.complex128, (bkgd_loop_num_pts,)), # in complex numbers
			("Q"						, np.float64),
			("Qc"						, np.float64),
			("Fr"						, np.float64), # in Hz
			("Power_Baseline_Noise"    	, np.float64), # dBm -  off resonnce power entering digitizer for noise spectrum measurement		
			("Noise_Freq_On_Res"    	, np.float64), # Hz - freq at which on res noise was taken
			("Noise_II_On_Res"			, np.float64,(noise_spectrum_length,)), # PSD in V^2/Hz
			("Noise_QQ_On_Res"			, np.float64,(noise_spectrum_length,)),
			("Noise_IQ_On_Res"			, np.complex128,(noise_spectrum_length,)),
			("Noise_Freq_Off_Res"    	, np.float64), # Hz - freq at which off res noise was taken			
			("Noise_II_Off_Res"			, np.float64,(noise_spectrum_length_off_res,)),
			("Noise_QQ_Off_Res"			, np.float64,(noise_spectrum_length_off_res,)),
			("Noise_IQ_Off_Res"			, np.complex128,(noise_spectrum_length_off_res,)),
			("Noise_Freq_Vector"    	, np.float64,(noise_spectrum_length,)), # in Hz, the  frequencies of the noise spectrum
			("Noise_Chan_Input_Atn"		, np.uint32), # attenuator box attenuation value on input side of fridge
			("Noise_Chan_Output_Atn"	, np.uint32), # attenuator box attenuation value on output side of fridge
			#auto and cross correlated noise
			#phase cancellation value for carrier cuppresion - np.float64,(fsteps_syn,)
			#ampl cancellation value for carrir suppression -  np.float64,(fsteps_syn,)
			("Scan_Timestamp"			, '|S12'),
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
					self._print( '{0} exists...'.format(filename[0:pos+1]))
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
				self._print( 'Adding data to existing table {0}'.format(table_path))
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
					self._print('Table metadata {0} not defined and is set to None'.format(data))
			sweep_data_table.attrs.keys =  self.measurement_metadata.keys() #NOTE: we record the metadata keys in the metadate
			sweep_data_table.flush()
		
	# def load_hf5(self, database_filename, tablepath):
	# 	''' table path is path to the database to be loaded starting from root. e.g. self.load_hf5('/Run44b/T201312102229')
	# 	database_filename is the name of the hf5 database to be accessed for the  table informaiton'''
	# 	database_filename = self.data_directory_path + os.sep+ database_filename
	# 	if not os.path.isfile(database_filename):
	# 		logging.error('Speficied h5 database does not exist. Aborting...')
	# 		return 
		
	# 	wmode = 'a'
	# 	self.measurement_metadata = {} # Clear metadata

	# 	# use "with" context manage to ensure file is always closed. no need for fileh.close()
	# 	with tables.open_file(database_filename, mode = wmode) as fileh:
	# 		table = fileh.get_node(tablepath)	
	# 		self.Sweep_Array = table.read()
	# 		for key in table.attrs.keys:
	# 			exec('self.measurement_metadata["{0}"] = table.attrs.{0}'.format(key))

	# 	self.sweep_data_columns = self.Sweep_Array.dtype

	def _record_LNA_settings(self):
		self.psu = Instruments.power_supply(self.lna_psu_address)
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

		num_points_per_na_scan = np.int(self.measurement_metadata['Num_Points_Per_NA_Scan']) 
		num_f_steps = self.measurement_metadata['NA_Scan_Num_Points']
		scan_bw = self.scan_bw
		num_sub_scans = np.int(num_f_steps/np.float64(num_points_per_na_scan))
		sub_scan_width = (scan_bw/num_f_steps)*num_points_per_na_scan
		model_freq_array = np.linspace(self.measurement_metadata['Frequency_Range'][0],self.measurement_metadata['Frequency_Range'][1], num = num_f_steps,dtype=np.float64)
		
		with self._na_ctx():
			for i in xrange(num_sub_scans):
				self.na.set_start_freq(model_freq_array[num_points_per_na_scan*i])
				self.na.set_stop_freq(model_freq_array[num_points_per_na_scan*(i+1)-1])			
				self.na.trigger_single_scan()
				frequencies, s_parameters = self.na.read_scan_data()
				self.Sweep_Array[0]['Frequencies'][num_points_per_na_scan*i: num_points_per_na_scan*(i+1)] = frequencies
				self.Sweep_Array[0]['S21'][num_points_per_na_scan*i: num_points_per_na_scan*(i+1)] = s_parameters
			del(model_freq_array)

	


	def _setup_na_for_scans(self):
		'''
		Sets of na for scans:
			-determine number of scan points to use,
			-creates Sweep_Array data type
			-sets of single scan and triggering
		'''
		min_number_of_na_scan_pts = np.float64(100) #use a nice rounds number
		avg_fac = self.measurement_metadata['NA_Average_Factor']
		IFBW = self.measurement_metadata['IFBW']
		min_resolution = self.measurement_metadata['Min_Freq_Resolution']
		scan_bw = self.scan_bw
		min_num_scan_pts_req = np.int(np.ceil(scan_bw/min_resolution))

		
		def round_up(x):
			return np.int(np.ceil(x/min_number_of_na_scan_pts)) * min_number_of_na_scan_pts

		def round_down(x):
			return np.int(np.floor(x/min_number_of_na_scan_pts)) * min_number_of_na_scan_pts

		with self._na_ctx():
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
				self._print('Total num freq pts: {}. Number pts per scan {}. Number of scans {}'.format(num_f_steps,num_points_per_na_scan,num_sub_scans ))

			sub_scan_width = (scan_bw/num_f_steps)*num_points_per_na_scan
			self.measurement_metadata['Num_Points_Per_NA_Scan'] = num_points_per_na_scan
			

			self._define_sweep_data_columns( fsteps = np.int(num_f_steps), 
										tpoints = self.measurement_metadata['Num_Temp_Readings'],
										fsteps_syn = self.num_syn_freq_points
										)
			
			self.na.set_num_points_per_scan(num_points_per_na_scan)
			self.na.set_IFBW(IFBW)
			if avg_fac  > 0:
				self.na.turn_on_averaging(avg_fac, channel = 1)
			self.na.query_sweep_time( update_timout = True)
			self.na.setup_single_scan_mode()

	def execute_sweep(self, sweep_parameter_file, database_filename, use_table = None, End_Heater_Voltage = None):
		'''
		Reads the sweep_parameter_file located in data_directory_path and reads in swee parameters.

		Frequency_Range: [717276850.0, 717351766.0] | [Hz Hz]
		Min_Resolution: 25 | Hz
		IFBW: 30 | Hz
		Average_Factor: 0 | 0 if no averaging
		Heater_Voltage: [0] | [Volts]
		Thermometer_Bias: [0.006] | [Volts]
		Aux_Voltage:  range(0,11)+range(9,-10, -1)+range(-10,1) | [Volts] 
		Delay_Time: 300 | Seconds - Time to wait before starting measurement. 0 if no delay..
		Pause: 1	| Seconds - Time to wait between changing heater voltage
		Powers: [-55] | [dBm]
		Atten_NA_Output: 0 | dB
		Atten_NA_Input: 0 | dB
		Atten_RTAmp_Input: 10 | dB
		Atten_Mixer_Input: 8 | value for switch atten
		RTAmp_In_Use:  True | True or False
		Device_Themometer_Channel: MF3 | Can be 'MF1', 'MF2', or 'MF3'
		Num_Temp_Readings: 10 |  The actual temperature is the median of these readings
		Synthesizer_Scan_Num_Points: 0 | Number of points to use in Syn sweep to deterine f0. 0 for no sweep, or else > 25.
		Measure_On_Res_Noise: True |
		Measure_Off_Res_Noise False |

		'''
		if use_table != None:
			self.current_sweep_data_table = use_table

		sweep_parameter_file = self.data_directory_path + os.sep + sweep_parameter_file
		database_filename = self.data_directory_path + os.sep + database_filename

		update_token = 'Sweep_Data_Table' #line to updated in sweep_parameter_file
	
		
		self.num_noise_spec_freq_points = 1

		tokens = 	[('Powers:','Powers', eval), ('Min_Resolution:','Min_Freq_Resolution', np.float), ('IFBW:','IFBW', np.float),
					('Heater_Voltage:','Heater_Voltage',eval), ('Aux_Voltage:','Aux_Voltage',eval), ('Average_Factor:','NA_Average_Factor', np.int), 
					('Pause:','Wait_Time',np.float), ('RTAmp_In_Use:', 'RTAmp_In_Use', eval),  ('Synthesizer_Scan_Num_Points:','Synthesizer_Scan_Num_Points',np.int),
					('Atten_NA_Output:', 'Atten_NA_Output',np.float32), ('Atten_NA_Input:','Atten_NA_Input',np.float32),
					('Atten_RTAmp_Input:','Atten_RTAmp_Input',np.float32),  ('Thermometer_Bias:','Thermometer_Voltage_Bias', eval),
					('Frequency_Range:','Frequency_Range',eval), ('Delay_Time:','Delay_Time',np.float),
					('Device_Themometer_Channel:', 'Device_Themometer_Channel',str), ('Num_Temp_Readings:','Num_Temp_Readings',np.int),
					('Atten_Mixer_Input:', 'Atten_Mixer_Input', np.int), ('Measure_On_Res_Noise:', 'Measure_On_Res_Noise', eval),
					('Measure_Off_Res_Noise:', 'Measure_Off_Res_Noise', eval )] #dont use bool or np.bool because bool('False') = True!. Use eval 
		
		self._read_tokens_in_file(sweep_parameter_file, tokens)

		
		thermometer_channel_name = self.measurement_metadata['Device_Themometer_Channel']
		num_temp_readings = self.measurement_metadata['Num_Temp_Readings'] 
		self.Delay_Time = self.measurement_metadata.pop('Delay_Time')
		self.Aux_Voltage = self.measurement_metadata.pop('Aux_Voltage')
		self.Powers = self.measurement_metadata.pop('Powers')
		self.Heater_Voltage = self.measurement_metadata.pop('Heater_Voltage')
		self.Thermometer_Voltage_Bias = self.measurement_metadata.pop('Thermometer_Voltage_Bias')
		#self.Frequency_Range  =  self.measurement_metadata.pop('Frequency_Range')
		
		if self.measurement_metadata['Synthesizer_Scan_Num_Points'] == 0:
			self.do_syn_scan = False
		elif self.measurement_metadata['Synthesizer_Scan_Num_Points'] < self.min_syn_freq_points:
			self.do_syn_scan = True
			logging.warn('Synthesizer_Scan_Num_Points < {0}. Setting to {0}'.format(self.min_syn_freq_points))
			self.measurement_metadata['Synthesizer_Scan_Num_Points'] = self.min_syn_freq_points
		else:
			self.do_syn_scan = True
			
		self.num_syn_freq_points = self.measurement_metadata['Synthesizer_Scan_Num_Points']	
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
			plt.rcParams["ytick.labelsize"] = 5
			plt.rcParams["xtick.labelsize"] = 5
			
			self.fig = plt.figure( figsize=(8, 8), dpi=100)

			self.ax = {}
			self.ax[1] = self.fig.add_subplot(2,2,(1,2)) #na scan
			self.ax[2] = self.fig.add_subplot(2,2,3) #syn scan
			self.ax[3] = self.fig.add_subplot(2,2,4) #noise
			plt.subplots_adjust(left=.1, bottom=.1, right=None ,wspace=.35, hspace=.3)
			#position plot window to extreme right
			mngr = plt.get_current_fig_manager()
			#plt.suptitle('Nonlinear Resonator Plots')
			mngr.window.setGeometry(3000,100,800,852)
			# self.background = {}
			# self.background[1] = self.fig.canvas.copy_from_bbox(self.ax[1].bbox)
			# self.background[2] = self.fig.canvas.copy_from_bbox(self.ax[2].bbox)
			# self.background[3] = self.fig.canvas.copy_from_bbox(self.ax[3].bbox)
			plt.ion()
			plt.show()
			#plt.draw()
		
		self._setup_na_for_scans()

		with self._fi_ctx():
			t_index = 0
			for t in self.Heater_Voltage:
				#self.fi.resume()
				self.fi.set_stack_heater_voltage(t)
				self.fi.set_thermometer_bias_voltage(self.Thermometer_Voltage_Bias[t_index])
				self.fi.suspend()

				if (t_index != 0) & (self.measurement_metadata['Wait_Time'] != np.float(0)):
					with self._na_ctx():
						self.na.off()
						time.sleep(self.measurement_metadata['Wait_Time'])
						self._print('Waiting for temperature. Network analyzer stimulus off. Next scan at {}'.format((datetime.datetime.now() + datetime.timedelta(seconds = self.measurement_metadata['Wait_Time'])).strftime( '%m/%d/%Y %H:%M:%S')))
						self.na.on()

				p_index = 0	
				for p in self.Powers:
					with self._na_ctx():
						self.na.set_power(p)

					a_index = 0
					for a in self.Aux_Voltage:
						self.fi.resume()
						self.fi.set_aux_channel_voltage(a)
						self.fi.suspend()


						self._print('Executing scan: Temp {0} of {1}, Pow {2} of {3}, Aux {4} of {5}'.format(t_index+1,self.num_temp_sweep_pts,p_index+1,self.num_power_sweep_pts, a_index+1,self.num_aux_sweep_pts) )					
						
						#sweep_data_columns are defined in _setup_na_for_scans() context manager
						self.Sweep_Array = np.zeros(1, dtype = self.sweep_data_columns)

						######
						#
						# Perform sub scans: and add data to Sweep_Array
						#
						######

						self._perform_subscans()
						
					
						
						self.fi.resume()
						temperature_readings =  np.empty((num_temp_readings,), np.float64)
						for i in xrange( temperature_readings.size):
							temperature_readings[i] = self.fi.read_temp(thermometer_channel_name)
						self.fi.suspend()

						self._define_sweep_array(0,  #The  array  is only 1 record long and this record is index 0
												Fstart = self.Sweep_Array[0]['Frequencies'][0],
												Fstop = self.Sweep_Array[0]['Frequencies'][-1],
												Heater_Voltage = t,
												Temperature_Readings = temperature_readings,
												Temperature = np.median(temperature_readings),
												Thermometer_Voltage_Bias = self.Thermometer_Voltage_Bias[t_index],#set to zero unless there is an array of temps in the ScanData
												Pinput_dB = p, #we only want the power coming out of the source, i.e. the NA
												Is_Valid = True,
												Aux_Voltage = a,
												Scan_Timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M')
												)

						

						if self.Show_Plots: # issue '%matplotlib qt' in spyder?
							#self.ax[1].clear()
							
							line = self.ax[1].plot(self.Sweep_Array[0]['Frequencies'],20*np.log10(np.abs(self.Sweep_Array[0]['S21'])))
							self.ax[1].set_xlabel('Frequency [Hz]')
							self.ax[1].set_ylabel('$20*Log_{10}[|S_{21}|]$ [dB]')

							self.ax[1].set_title('Run: {0}; Sensor: {1}; Ground: {2}; Power {3} dBm'.format(self.measurement_metadata['Run'], 
																									self.measurement_metadata['Sensor'], 
																									self.measurement_metadata['Ground_Plane'], 
																									p))		
							#ax.legend(loc = 'best', fontsize = 9)
							self.ax[1].grid(which = 'both')
							# self.fig.canvas.restore_region(self.background[1])
							# self.ax[1].draw_artists(line[0])
							# self.fig.canvas.blit(self.ax[1].bbox)
							plt.draw()

							time.sleep(0.2) # Dont use plt.pause() .. it's deprecated


						if self.do_syn_scan:
							######
							#
							# Find F0, Q from current NA scan
							#
							######
							###### Update This Code Later. Remove use of  self.swp....
							frequencies_size = self.sweep_data_columns['Frequencies'].subdtype[1][0]
							self.swp._define_sweep_data_columns(frequencies_size, 1)
							self.swp.Sweep_Array = np.zeros(1, dtype = self.swp.sweep_data_columns)
							self.swp._define_sweep_array(0,
														S21 = self.Sweep_Array[0]['S21'],
														Frequencies =  self.Sweep_Array[0]['Frequencies'],
														Is_Valid = True
														)
							
							
							self.swp.fill_sweep_array(Fit_Resonances = True, Compute_Preadout = False, Add_Temperatures = False, Complete_Fit = False, Remove_Gain_Compression = False )
							self.swp.pick_loop(0)
							
							if self.swp.Sweep_Array[0]['Is_Valid']:
								
								self._define_sweep_array(0,
														Fr = self.swp.Sweep_Array[0]['Fr'],
														Q = self.swp.Sweep_Array[0]['Q'],
														Qc = self.swp.Sweep_Array[0]['Qc'],
														Is_Valid = True
														)
								f_center = self.swp.Sweep_Array[0]['Fr']
								Q_center = self.swp.Sweep_Array[0]['Q']
								######
								#
								# Proceed with syn scan/noise only if fit is valid
								#
								######
								self._syn_scan_s21_noise(f_center, Q_center)


							else:
								self.Sweep_Array[0]['Is_Valid'] = False 
								self._print('Unable to fit NA scan.')

						
						self.save_hf5(database_filename , overwrite = False)
						a_index = a_index +1
					p_index = p_index +1		
				t_index = t_index + 1
		self._update_file(sweep_parameter_file, update_token, self.current_sweep_data_table)
		self.current_sweep_data_table = None
		
		with self._na_ctx():
			self._restore_na_settings()

		if  End_Heater_Voltage != None:
			with self._fi_ctx():
				self.fi.resume()
				self.fi.set_stack_heater_voltage(End_Heater_Voltage)
				self.fi.suspend()


	def get_current_na_frequency_range(self):
		with self._na_ctx():
			start = self.na.get_start_freq()
			stop = self.na.get_stop_freq()
			num_pts = np.float64(self.na.get_num_points_per_scan())
		freq_resolution = (stop - start)/num_pts
		self._print('Start: {}    Stop: {}    Freq Resolution: {}'.format(start,stop, freq_resolution))

	def pulse_stack_heater_voltage(self, voltage, time_seconds, return_voltage = None):
		'''
		set stack heater voltage to 'voltage' for a duration of 'time_seconds', 
		then return to 'return_voltage' (if it not None) or to the voltage it was at before.
		'''
		with self._fi_ctx():
			if return_voltage == None:
				return_voltage = self.fi.get_stack_heater_voltage()	
			self.fi.set_stack_heater_voltage(voltage)
			time.sleep(time_seconds)
			self.fi.set_stack_heater_voltage(return_voltage)
			self.fi.suspend()
		
	def read_temp(self):
		'''
		Read the temperature on the currently selected thermometer channel.
		'''
		thermometer_channel_name = self.measurement_metadata['Device_Themometer_Channel']
		if thermometer_channel_name == None:
			self._print('No thermometer channel is selected. Update metadate....')
			return 
		with self._fi_ctx():
			self.fi = Fridge_Interfaces.fridge_interface(self.devices)	
			temp = self.fi.read_temp(thermometer_channel_name)
			bias = self.fi.DAC_Bias
			self.fi.suspend()
		self._print('Temperature is {:.4f} K on channel {}, which is biased at {} V'.format(temp, thermometer_channel_name, bias))
		return temp

	def switch_to_noise_channel(self):
		'''
		first turn off na stimulus
		then apply 28 Volts to switches to switch from na to noise readout channels
		then turn on synthesizer
		'''
		with self._na_ctx():
			self.na.off()


		with self._sm_ctx():
			voltage = 28 # Volts
			self.sm.set_voltage('a', voltage)
			self.sm.set_voltage('b', voltage)

			self.sm.on('a')
			self.sm.on('b')

		with self._syn_ctx():
			self.syn.on()

	def switch_to_na_channel(self):
		'''
		first turn off syn
		then switch off source meter to actuate switches to na channel
		then turn on an stimulus
		'''
		with self._syn_ctx():
			self.syn.off()

		# now switch to na circuit
		with self._sm_ctx():
			voltage = 0 # Volts
			self.sm.set_voltage('a', voltage)
			self.sm.set_voltage('b', voltage)

			self.sm.off('a')
			self.sm.off('b')

		with self._na_ctx():
			self.na.on()

	def _record_LNA_settings(self):
		'''
		query and record in measurement_metadata th  LNA voltage and current settings.
		'''
		with self._psu_ctx():
			self.measurement_metadata['LNA'].update(Vg = self.psu.get_Vg(),
													Vd = self.psu.get_Vd(), 
													Id = self.psu.get_Id()) 

	def _construct_freq_array(self, f_center, Q, Freq_Spacing = 'Linear'):
		'''
		Constructs a freq array with different spacing functions between the points. 
		Choose between:
			linear
			triangular
			exponential
		The latter two put a high density of points around f_center. 
		The Frequency array is constructed to include f_center.

		NOTE: The Anritsu MG3690B has a frequency resolution of 0.01 Hz. 
		The latter two spacing can be narrower than this very close to f_center for vary high Q resonance 

		'''
		num = self.measurement_metadata['Synthesizer_Scan_Num_Points']
		num_BW = self.measurement_metadata['Synthesizer_Scan_Num_BW']
		fad =  {'f_center':f_center,'num_BW':num_BW,'num': num, 'Freq_Spacing':Freq_Spacing} # Freq Array Dict
		BW = fad['num_BW']*fad['f_center']/Q
		
		
		num = np.floor(fad['num']/2.0)
		idx_start = 1 if np.mod(fad['num'],2) == 0 else 0

		if fad['Freq_Spacing'].lower() == 'triangular': #Triangular numbers - Denser around f_0, (f-f_center)^2 distribution
			T = np.linspace(0, num,  num=num+1, endpoint=True, retstep=False, dtype=np.float64)
			T = T*(T+1.0)/2.0
			f_plus = (BW/2)*(T/T[-1]) + fad['f_center']

		if fad['Freq_Spacing'].lower() == 'linear': #linear
			f_plus = np.linspace(fad['f_center'], fad['f_center'] + BW/2,  num=num+1, endpoint=True, retstep=False, dtype=np.float64)

		if fad['Freq_Spacing'].lower() == 'exponential': #exponential
			base = BW/2.0
			exp = np.linspace(0, 1,  num=num+1, endpoint=True, retstep=False, dtype=np.float64)
			f_plus = fad['f_center'] + (np.power(base, exp) - 1)

		f_minus = -f_plus[:0:-1] + 2*fad['f_center']
		f = np.hstack((f_minus[idx_start:], f_plus[0:])) #f_plus[0] is fad['f_center']

		return f

	def _syn_scan_s21_noise(self,f_center, Q_center):
		'''
		obtains current power from self.Sweep_Array[0]['Pinput_dB']. matches that power with attenuator box.
		'''
		LO_power = self.measurement_metadata['Synthesizer_Scan_Power']
		Pinput_dB = self.Sweep_Array[0]['Pinput_dB']
		frequencies_syn = self._construct_freq_array(f_center, Q_center, Freq_Spacing = self.measurement_metadata['Synthesizer_Frequency_Spacing'])
		s21_syn = np.empty_like(frequencies_syn, dtype = np.complex128)

		attenuation_1 = np.int(self.na_to_atn_conversion_func(Pinput_dB))
		attenuation_2 = np.int(self.measurement_metadata['Atten_Mixer_Input'])

		

		with self._digitizer_ctx(), self._syn_ctx(), self._iqd_ctx(), self._atn_ctx():
			#set syn LO power
			self.syn.set_power(LO_power)

			#set attenuator box attenuation values
			self.atn.set_atn_chan(1,attenuation_1)
			self.atn.set_atn_chan(2,attenuation_2)

			#setup NI digitizer
			self.iqd.create_channel()
			self.iqd.configure_sampling(sample_rate = self.measurement_metadata['IQ_Sample_Rate'])
			
			#first measure background loop
			self._print('Measuring background loop')
			bkgd_BW = 10e6
			bkgd_freqs = np.linspace(self.Sweep_Array[0]['Frequencies'][0], self.Sweep_Array[0]['Frequencies'][0] + bkgd_BW , num= self.bkgd_loop_num_pts, endpoint=True, retstep=False, dtype=np.float64)
			s21_bkgd = np.empty_like(bkgd_freqs, dtype = np.complex128)
			for n in xrange(self.bkgd_loop_num_pts):
				self._print('Synthesizer point {} of {}'.format(n+1, self.bkgd_loop_num_pts), nnl= True)
				self.syn.set_freq(bkgd_freqs[n])

				I_bkgd , Q_bkgd  = self.iqd.acquire_readings_2chan()
				s21_bkgd[n] = np.median(I_bkgd)  + 1j*np.median(Q_bkgd)

			self._define_sweep_array(0,
						Bkgd_S21_Syn  = s21_bkgd,
						Bkgd_Freq_Syn  = bkgd_freqs
						)
			self._print('')

			#loop through frequencies and record I Q points
			self._print('Measuring loop') 
			for n in xrange(self.num_syn_freq_points):
				self._print('Synthesizer point {} of {}'.format(n+1, self.num_syn_freq_points), nnl= True)
				self.syn.set_freq(frequencies_syn[n])

				I , Q  = self.iqd.acquire_readings_2chan()
				s21_syn[n] = np.median(I)  + 1j*np.median(Q)
			self._print('')
			self._define_sweep_array(0,
									S21_Syn = s21_syn,
									Frequencies_Syn = frequencies_syn,
									Noise_Chan_Input_Atn = attenuation_1, 
									Noise_Chan_Output_Atn = attenuation_2)
			if self.Show_Plots: # issue '%matplotlib qt' in spyder?
				self.ax[2].clear()

				line = self.ax[2].plot(self.Sweep_Array[0]['S21_Syn'].real,self.Sweep_Array[0]['S21_Syn'].imag)
				self.ax[2].set_xlabel('Frequency [Hz]')
				self.ax[2].set_ylabel('$20*Log_{{10}}[|S_{{21}}|]$ [dB]')
				self.ax[2].set_title( r'Syn Scan, $f_{{ctr}}$' + ' = {freq:0.3f} MHz, Q = {Qt:0.0f}k'.format(freq = f_center/1.e6, Qt = Q_center/1.e3))		
				#ax.legend(loc = 'best', fontsize = 9)
				self.ax[2].grid()
				plt.draw()
				time.sleep(0.2)
				
				#plt.show()
				# self.ax[2].grid()
				# self.fig.canvas.restore_region(self.background[2])
				# self.ax[2].draw_artists(line[0])
				# self.fig.canvas.blit(self.ax[2].bbox)
				
			###### Update This Code Later. Remove use of  self.swp....

			self.swp._define_sweep_data_columns(frequencies_syn.size, 1)
			self.swp.Sweep_Array = np.zeros(1, dtype = self.swp.sweep_data_columns)
			self.swp._define_sweep_array(0,
										S21 = s21_syn,
										Frequencies = frequencies_syn,
										Is_Valid = True 
										)
			#self.swp.pick_loop(0)
			self.swp.fill_sweep_array(Fit_Resonances = True, Compute_Preadout = False, Add_Temperatures = False, Complete_Fit = False, Remove_Gain_Compression = False )
			f_noise = self.swp.Sweep_Array[0]['Fr']

			######

			######
			#
			# Measure on resonance noise at f_noise if fit is valid, then measure off resonance noise
			#
			######			
			if self.swp.Sweep_Array[0]['Is_Valid'] & self.measurement_metadata['Measure_On_Res_Noise']:
				self.syn.set_freq(f_noise)
				time.sleep(1)
				# print('disconnect cables now')
				# time.sleep(40)
				self._print('Measuring on resonance noise')
				PII, PQQ, PIQ, f  = self._measure_noise()
				self._define_sweep_array(   0,
											Noise_Freq_On_Res   = f_noise,
											Noise_II_On_Res     = PII,
											Noise_QQ_On_Res     = PQQ,
											Noise_IQ_On_Res     = PIQ,
											Noise_Freq_Vector   = f    )
				
				######
				#
				# Measure off resonance noise at f_noise, if necessary
				#
				######

				if self.measurement_metadata['Measure_Off_Res_Noise']:
					f_noise = f_noise + self.measurement_metadata['Off_Res_Freq_Offset']
					self.syn.set_freq(f_noise)
					time.sleep(1)
					self._print('Measuring off resonance noise')
					PII, PQQ, PIQ, f  = self._measure_noise()
					self._define_sweep_array(   0,
												Noise_Freq_Off_Res   = f_noise,
												Noise_II_Off_Res     = PII,
												Noise_QQ_Off_Res     = PQQ,
												Noise_IQ_Off_Res     = PIQ)

				if self.Show_Plots: # issue '%matplotlib qt' in spyder?
					self.ax[3].clear()

					line = self.ax[3].loglog(self.Sweep_Array[0]['Noise_Freq_Vector'],self.Sweep_Array[0]['Noise_II_On_Res'], label = 'PII On')
					line = self.ax[3].loglog(self.Sweep_Array[0]['Noise_Freq_Vector'],self.Sweep_Array[0]['Noise_QQ_On_Res'], label = 'PQQ On')
					line = self.ax[3].loglog(self.Sweep_Array[0]['Noise_Freq_Vector'],np.abs(self.Sweep_Array[0]['Noise_IQ_On_Res']), label = 'PIQ On')
					if self.measurement_metadata['Measure_Off_Res_Noise']: 
						line = self.ax[3].loglog(self.Sweep_Array[0]['Noise_Freq_Vector'],self.Sweep_Array[0]['Noise_II_Off_Res'], label = 'PII Off')
						line = self.ax[3].loglog(self.Sweep_Array[0]['Noise_Freq_Vector'],self.Sweep_Array[0]['Noise_QQ_Off_Res'], label = 'PQQ Off')
						line = self.ax[3].loglog(self.Sweep_Array[0]['Noise_Freq_Vector'],np.abs(self.Sweep_Array[0]['Noise_IQ_Off_Res']), label = 'PIQ Off')

					self.ax[3].set_xlabel('Frequency')
					self.ax[3].set_ylabel('$V^2/Hz$')

					self.ax[3].set_title(r'I and Q PSD, $f_{{noise}}$ = {fn:0.3f} MHz'.format(fn = f_noise/1e6))		
					self.ax[3].legend(loc = 'best', fontsize = 6)
					self.ax[3].grid(which='both')
					self.ax[3].set_ylim(bottom = 8e-17)
					plt.draw()
					time.sleep(0.2)
					#plt.show()
			else:
				self._print('Unable to fit synthesizer scan')
											
	def _measure_noise(self):
		'''
		Takes noise at current synthesizer frequency, and attenuator/power settings.  If hardware AA filter is desired, 
		this should be in place before call this function

		Takes num_integrations, each of duration integration_time, for a total integration time of 
		integration_time * num_integrations.
		'''
		decimation_factor = self.measurement_metadata['Noise_Decimation_Factor']
		integration_time = self.measurement_metadata['Noise_Integration_Time'] # second
		num_integrations = self.measurement_metadata['Noise_Num_Integrations']*1.0
		noise_sample_rate = self.measurement_metadata['Noise_Sample_Rate'] 
		num_samples = noise_sample_rate * integration_time

		#Recongifure NI digitizer
		self.iqd.configure_sampling(sample_rate = noise_sample_rate, num_samples = num_samples)

		# PII = np.array([], dtype = np.float64)
		# PQQ = np.array([], dtype = np.float64)
		# PIQ = np.array([], dtype = np.complex128)
		for n in np.arange(num_integrations):
			self._print('Noise integration {} of {}'.format(n+1, num_integrations), nnl= True)
			In , Qn  = self.iqd.acquire_readings_2chan()
			In = In - In.mean()
			Qn = Qn - Qn.mean()
			
			PIIn, PQQn, PIQn, f = self._compute_PSD(In, Qn)
			if n == 0:
				PII = PIIn
				PQQ = PQQn
				PIQ = PIQn
			else:
				PII = PII + PIIn
				PQQ = PQQ + PQQn
				PIQ = PQQ + PIQn

		PII = PII/num_integrations
		PQQ = PQQ/num_integrations
		PIQ = PIQ/num_integrations
		return PII, PQQ, PIQ, f		

	def _compute_PSD(self, I, Q): #, sampling_rate, decimation_factor):
		'''
		I and Q are time series in the format of numpy arrays.

		First, use an order low pass filter (8 Chebyshev type I iir filter) to guard against aliasing
		Then, downsample the I and Q time series by the argument 'decimation_factor'. 
		Then, compute two-sided auto and cross power spectral densities using the Welch method.
		A frequency segmentation is defined, as well as a frequency resolution for each frequency segment.

		This function uses no instruments
		'''
		
		noise_sampling_rate = self.measurement_metadata['Noise_Sample_Rate']
		decimation_factor = self.measurement_metadata['Noise_Decimation_Factor']
		sampling_rate_decimated = noise_sampling_rate/decimation_factor #need to floor/ceil here?

		if I.shape != Q.shape: #make sure I and Q are the same shape
			logging.error('I and Q time series must be the same size.  Aborting')
			return
		# frequency_segmentation = self.[100., 5000., sampling_rate_decimated/2.0]
		# frequency_resolution_per_segment = [1., 10., 100.]
		
		frequency_segmentation = self.measurement_metadata['Noise_Frequency_Segmentation'] 
		frequency_resolution_per_segment = self.measurement_metadata['Noise_Frequency_Resolution_Per_Segment'] 

		I = signal.decimate(I, int(decimation_factor))
		Q = signal.decimate(Q, int(decimation_factor))
		
		num_points = I.size
		
		#df = sampling_rate_decimated*1.0/num_points
		fmin = 0.0
		

		#segpts = [200000,20000,2000,]
		#[2**18, 2**15,2**11 ]
		f =   np.array([], dtype = np.float64)
		PII = np.array([], dtype = np.float64)
		PQQ = np.array([], dtype = np.float64)
		PIQ = np.array([], dtype = np.complex128)		

		for m in xrange(len(frequency_segmentation)):
			fmax = frequency_segmentation[m]
			#num_pts_in_segment = num_points * df/frequency_resolution_per_segment[m]
			num_pts_in_segment = int(sampling_rate_decimated / frequency_resolution_per_segment[m])
			hann_window = signal.get_window('hann',num_pts_in_segment, fftbins=False) #fftbins=True -> periodic window
			#num_pts_overlap = np.floor(num_pts_in_segment * overlap_factor)
		
			# #Compute PSD
			# fIm, PIIm  = signal.welch(I, fs=sampling_rate_decimated, window='hann', nperseg=num_pts_in_segment, noverlap=0, nfft=None, detrend='constant', return_onesided=False, scaling='density', axis=-1)
			# fQm, PQQm  = signal.welch(Q, fs=sampling_rate_decimated, window='hann', nperseg=num_pts_in_segment, noverlap=0, nfft=None, detrend='constant', return_onesided=False, scaling='density', axis=-1)
			# #Compute cross PSD
			# fIQm,PIQm  = signal.csd(I, Q, fs=sampling_rate_decimated, window='hann', nperseg=num_pts_in_segment, noverlap=0, nfft=None, detrend='constant', return_onesided=False, scaling='density', axis=-1)
						#Compute PSD
			fIm, PIIm  = signal.welch(I, fs=sampling_rate_decimated, window=hann_window, nperseg=num_pts_in_segment, noverlap=0, nfft=num_pts_in_segment, detrend='constant', return_onesided=False, scaling='density', axis=-1)
			fQm, PQQm  = signal.welch(Q, fs=sampling_rate_decimated, window=hann_window, nperseg=num_pts_in_segment, noverlap=0, nfft=num_pts_in_segment, detrend='constant', return_onesided=False, scaling='density', axis=-1)
			#Compute cross PSD
			fIQm,PIQm  = signal.csd(I, Q, fs=sampling_rate_decimated, window=hann_window, nperseg=num_pts_in_segment, noverlap=0, nfft=num_pts_in_segment, detrend='constant', return_onesided=False, scaling='density', axis=-1)
			#Note: return_onesided=False -> two-sided
			#Note: scaling='density -> output PSD is V**2/Hz
			#Note: noverlap is the  Number of points to overlap between segments.
			#Note: Pxxm is the same length as I/Q
			condition = (fmin < fIm) & (fIm <= fmax)
			f = np.hstack((f,np.extract(condition, fIm)))
			PII = np.hstack((PII,np.extract(condition, PIIm)))
			PQQ = np.hstack((PQQ,np.extract(condition, PQQm)))
			PIQ = np.hstack((PIQ,np.extract(condition, PIQm)))
			#print('After welch and csd ', PIIm.size, PQQm.size, PIQm.size, 'After partitioning', PII.size, PQQ.size, PIQ.size, f[0],f[99], f[-1]  )

			fmin = fmax

		return PII, PQQ, PIQ, f

	def _compute_noise_spectrum_length(self):
		'''
		Computes length of noise spectrum, whcih is needed for 
		construction of the Sweep_Array data type before measurement of noise.
		'''
		frequency_segmentation = self.measurement_metadata['Noise_Frequency_Segmentation'] 
		frequency_resolution_per_segment = self.measurement_metadata['Noise_Frequency_Resolution_Per_Segment'] 
		noise_spectrum_length = 0
		for i in xrange(len(frequency_segmentation)):
			last_segmentation = frequency_segmentation[i-1] if i > 0 else 0 
			noise_spectrum_length = noise_spectrum_length + (frequency_segmentation[i] - last_segmentation)/frequency_resolution_per_segment[i]
		noise_spectrum_length = noise_spectrum_length - 1
		if divmod(noise_spectrum_length,1)[1] != 0: 
			ValueError('Noise spectrum length is not an integer is not an integer')
		return int(noise_spectrum_length)
