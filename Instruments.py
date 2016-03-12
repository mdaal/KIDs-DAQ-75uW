import visa #developed with version 1.8
import numpy as np #developed with version 1.10.1
import time 
import logging
import string
import struct
import PyDAQmx as pdmx # developed with version 1.3
import sys

class Instrument_Name_List:
	NETWORK_ANALYZER_E5071B = 'TCPIP0::169.229.225.4::inst0::INSTR'
	POWER_SUPPLY_HPE3631A = 'GPIB0::14::INSTR'
	SYNTHESIZER_MG3692B = 'GPIB0::7::INSTR'
	SYNTHESIZER_E4425B = ''
	DIGITIZER_ADC488 = 'GPIB1::14::INSTR'
	VOLTAGE_SOURCE_K213 = 'GPIB1::17::INSTR'
	SOURCE_METER_2612A =  'GPIB0::26::INSTR'
	DIGITIZER_NI6120 = 'NI6120'
	ATTENUATOR_BOX_8310 = 'GPIB0::10::INSTR'

class messaging(object):
	''' For neatly printing messages to the command line
	'''
	def __init__(self):
		self.issue_new_line = False # does new line nees to be issued before the next message
		self.Verbose = True # print messages if true
		self.char_width = 85
	def _print(self, mssg, nnl = False):
		'''
		Print mssg if self.Verbose is True
		nnl - no new line (carriage return on current line)

		'''
		
		if self.Verbose:
			if not nnl:
				if not self.issue_new_line:
					sys.stdout.write(mssg.ljust(self.char_width,'.') + '\n')
					sys.stdout.flush()
				else:
				 	sys.stdout.write('\n' + mssg.ljust(self.char_width,'.')+ '\n')
					sys.stdout.flush()
					self.issue_new_line = False

			
			if nnl:
				sys.stdout.write('\r' + mssg.ljust(self.char_width,'.'))
				sys.stdout.flush()
				self.issue_new_line = True

class Instrument(object):
	rm = visa.ResourceManager()
	inl = Instrument_Name_List


	def __init__(self, resource_name, Name = 'Instrument Superclass'):
		#self.rm = visa.ResourceManager()
		if resource_name is Instrument.inl.DIGITIZER_NI6120:
			self.inst = pdmx.Task()
		else:
			self.inst = Instrument.rm.open_resource(resource_name, open_timeout=0)
			self.inst.timeout = 10000 # milliseconds

		self.resource_name = resource_name
		self.Name = Name
		self.Verbose = False

		self.mssg = messaging() #for priting messgs neatly to screen
		self.mssg.Verbose = self.Verbose
		self._print('Instantiating {}'.format(self.Name))

	def _print(self, mssg, nnl = False):
		self.mssg._print(mssg, nnl = nnl)

	def __del__(self):
		'''
		remember  the del() function calls __del__()
		'''
		if self.resource_name is not Instrument.inl.DIGITIZER_NI6120:
			self.inst.close()
		del(self.inst)
		
		self._print ('Deleting {}'.format(self.Name))

	def __str__(self):
		print('{} at {}'.format(self.Name, self.resource_name))

	def list_resources(self):
		'''
		print a list of all visa device addresses
		'''
		Instrument.rm.list_resources()

	def identify(self):
		identification  = self.inst.query("*IDN?")	
		return identification

	def session(self):
		print(self.inst.session)

	def close(self):
		self.inst.close()
		#self.inst.clear()

	def _wait_stb(self,code = None):
		'''
		Query instrument stb code until a cetain value if obtained. 
		If no code is supplied, then query until stb != 0 is obtained. 
		If code is given, query until that code is obtained. 
		total wait time is limiter to  instrument timeout, self.inst.timeout.
		'''
		start = time.time()
		pause = 0.1 # pause between successvie queries
		#stb = None
		if code == None:
			while  self.inst.read_stb() == 0:
				#self._print ('stb code is: {}'.format(self.inst.read_stb()))
				time.sleep(pause)
				elapsed = time.time()
				if elapsed - start > self.inst.timeout/1000.0:
					logging.error('Instrument timeout reached')
					break
				#stb = self.inst.read_stb()
		else:
			while self.inst.read_stb() != code:
				time.sleep(pause)
				elapsed = time.time()
				if elapsed - start > self.inst.timeout/1000.0:
					logging.error('Instrument timeout reached')
					break
				#stb = self.inst.read_stb()
		#return stb
	
	def _wait_opc(self):
		'''
		Blocks command  line until opc == 1.0 is received from instrument. 
		this  indicats that  the instrument is ready for the next command.
		'''
		self.inst.query_ascii_values('*OPC?;') # will equal 1.0 when instrument has completed operation

class attenuator_box(Instrument):
	def __init__(self, resource_name, Name = 'Attenuator Box'):
		Instrument.__init__(self, resource_name, Name = Name)
		self.atn_func = lambda a: (a//2)*2 # floors to factor of 2

	def set_atn_chan(self,channel,attenuation):
		'''
		channel can be 1 or 2
		attenuation can be 0 to 62, in steps of 2. Thus, a positive number
		'''
		if (attenuation <= 62) & (attenuation >= 0):
			self.inst.write('CHAN {}'.format(channel))
			self.inst.write('ATTN {}'.format(self.atn_func(attenuation)))
		else:
			logging.error('Attenuation value of of bounds.')

class ni_digitizer(Instrument):
	def __init__(self,resource_name, Name = 'NI6120 Digitizer'):
		Instrument.__init__(self, resource_name, Name = Name)
		self.read = pdmx.int32() # for sampsPerChanRead
		
		self.inputrange_max = 0.2 # can be 42, 20, 10, 5, 2, 1, 0.5, 0.2 Volts
		self.inputrange_min = -1.0*self.inputrange_max
		self.sample_rate = 8.0e5 # samples per second
		self.timeout = 20.0 # seconds
		self.samples_per_channel = 2e4 

	def create_channel(self, channel = "Dev1/ai0:1"):
		'''
		Creates two channel by default
		'''
		self.channel = channel # physical channel(s)

		self.inst.CreateAIVoltageChan(self.channel, "", 
									pdmx.DAQmxConstants.DAQmx_Val_Cfg_Default, 
									self.inputrange_min, self.inputrange_max, 
									pdmx.DAQmxConstants.DAQmx_Val_Volts, None)

	def configure_sampling(self, sample_rate = None, num_samples = None):
		if sample_rate != None:
			self.sample_rate = sample_rate

		if num_samples != None:
			self.samples_per_channel = num_samples

		self.inst.CfgSampClkTiming("", np.int(self.sample_rate),
									pdmx.DAQmxConstants.DAQmx_Val_Rising,
									pdmx.DAQmxConstants.DAQmx_Val_FiniteSamps,
									np.int(self.samples_per_channel))

	def acquire_readings_2chan(self):
		'''
		returns the digitized readings for two channels.
		these vectors are each of shape: (self.samples_per_channel,) and dtype =  np.float64
		'''
		data_vector = np.empty((np.int(self.samples_per_channel*2),), dtype = np.float64)
		
		self.inst.StartTask()

		self.inst.ReadAnalogF64(np.int(self.samples_per_channel),self.timeout,
								pdmx.DAQmxConstants.DAQmx_Val_GroupByChannel, #as opposed to DAQmx_Val_GroupByScanNumber
								data_vector,
								np.int(self.samples_per_channel*2),
								pdmx.byref(self.read), # byref() Returns a pointer lookalike to a C instance
								None)

		self.inst.StopTask()

		#self.inst.ClearTask() #deprecated bc garbage collection is supposed to handle this.

		[AI0_readings, AI1_readings] = np.hsplit(data_vector, (np.int(self.samples_per_channel),))

		return  AI0_readings,  AI1_readings

	def enable_AA_filter(self):
		'''
		enable the built-in 5-pole Bessel 100kHz antialias filter of the NI6120 
		'''
		self.inst.SetAILowpassEnable("",1)

	def disable_AA_filter(self):
		'''
		disable the built-in 5-pole Bessel 100kHz antialias filter of the NI6120 
		'''
		self.inst.SetAILowpassEnable("",0)


class source_meter(Instrument):
	def __init__(self,resource_name, Name = 'Source Meter'):
		Instrument.__init__(self, resource_name, Name = Name)
		#self.inst.read_termination = u'\r\n'

		# set display to show both channels a and b
		self.setup_display()

		#set sourcing state to voltag  sourcing
		self.set_source_funct('a', 1) 
		self.set_source_funct('b', 1) 

		#set voltage compliance
		self.set_voltage_current_compliance_limits('a', 'v', 30)
		self.set_voltage_current_compliance_limits('b', 'v', 30)

		#set voltage range to Range to allow 28V for SMA switch
		self.set_voltage_current_range('a', 'v', 30)
		self.set_voltage_current_range('b', 'v', 30)

	
	def set_voltage_current_compliance_limits(self, channel, voltage_or_current,  voltage_or_current_value):
		'''
		set allowable voltage output range.
		channel  can be  'a' or 'b'.
		voltage_or_current can be 'v' or 'i'
		'''
		self.inst.write('smu{}.source.limit{} = {}'.format(channel.lower(), voltage_or_current.lower(), voltage_or_current_value))


	def set_voltage_current_range(self, channel, voltage_or_current,  voltage_or_current_value):
		'''
		set allowable voltage output range.
		channel  can be  'a' or 'b'.
		voltage_or_current can be 'v' or 'i'
		'''
		self.inst.write('smu{}.source.range{} = {}'.format(channel.lower(), voltage_or_current.lower(), voltage_or_current_value))

	def set_voltage(self, channel, voltage):
		'''
		channel  is 'a' or 'b'.
		can do +/- 200V
		'''
		self.inst.write('smu{}.source.levelv = {}'.format(channel.lower(),voltage))

	def query_voltage(self,channel):
		'''
		channel is 'a' or 'b'
		'''
		voltage = self.inst.query_ascii_values('x = smu{}.source.levelv print(x)'.format(channel))[0]
		return voltage

	def set_source_funct(self, channel, voltage_or_current):
		'''
		Sets the sourceing state. 
		channel is 'a' or 'b'
		voltage_or_current can be: 
		0 for DC Amps
		1 for DC Volts
		'''
		self.inst.write('smu{}.source.func = {}'.format(channel, voltage_or_current))

	def clear_errors(self):
		self.inst.write('errorqueue.clear()')

	def on(self,channel):
		'''
		channel  is 'a' or 'b'.
		'''
		self.toggle_on_off(channel, 1)

	def off(self,channel):
		'''
		channel  is 'a' or 'b'.
		'''
		self.toggle_on_off(channel, 0)

	def setup_display(self):
		'''
		sets  display screen to show both channel a and channel b
		'''
		self.inst.write('display.screen = 2') # 2 is both a & b; 1 is b; and 0 is a 
		


	def toggle_on_off(self,channel, state):
		'''
		channel  is 'a' or 'b'.
		state is 0 or  off and 1 for on
		'''
		self.inst.write('smu{}.source.output={}'.format(channel, state))

class network_analyzer(Instrument):
	def __init__(self,resource_name, Name = 'Network Analyzer'):
		Instrument.__init__(self, resource_name, Name = Name)
		self.inst.chunk_size = 2**15 # in bytes
		self.inst.timeout = 155000 #milliseconds, None is infinite timeout  NOTE: This number applies to queries
		self.inst.values_format.is_binary = True
		self.inst.values_format.datatype = 'd' # 'f' floats and 'd' double
		self.inst.values_format.is_big_endian = True
		self.inst.values_format.container = np.array
		self.max_na_scan_points = 1601
		self.min_na_scan_points = 2
		self.min_power_output = -55 #dBm
		self.average_count = 0

	def set_sweep_type(self, sweep_type  = 'LIN', channel = 1):
		'''
		sweep_type can be:
		LIN - linear sweep
		LOG
		SEGM
		POW  - power sweep
		'''
		self.inst.write(':SENS{0}:SWE:TYPE {1} ;'.format(channel, sweep_type))

	def get_sweep_type(self, channel = 1):
		'''
		sweep_type can be:
		LIN - linear sweep
		LOG
		SEGM
		POW  - power sweep
		'''
		sweep_type = self.inst.query_ascii_values(':SENS{0}:SWE:TYPE?;'.format(channel), converter =  lambda x: str(x).strip('\n'))[0]
		return sweep_type

	def set_CW_freq(self, cw_freq_Hz, channel = 1):
		'''
		Set the continuous wave frequency. Used for power sweeps.

		Frequency is on hertz
		'''
		self.inst.write(':SENS{0}:FREQ {1:f} ;'.format(channel, cw_freq_Hz))


	def on(self):
		self.inst.write('OUTP ON ;')

	def off(self):
		self.inst.write('OUTP OFF ;')

	def set_power(self, power_dBm, channel = 1):
		'''
		Change output power on specified channel.
		'''
		self.inst.write(':SOUR{0}:POW {1} ;'.format(channel, power_dBm))

	def get_power(self, channel = 1):
		'''
		Get current output power on specified channel.
		'''
		power = self.inst.query_ascii_values(':SOUR{0}:POW?;'.format(channel))[0]
		return power

	def set_start_power(self, start_power_dBm, channel = 1):
		'''
		Set start power for power sweep output power on specified channel.
		
		Start power must be lower than stop power.
		'''
		self.inst.write(':SOUR{0}:POW:STAR {1} ;'.format(channel, start_power_dBm))

	def get_start_power(self, channel = 1):
		'''
		Get start power for power sweep output power on specified channel.
		
		Start power must be lower than stop power.
		'''
		start_power = self.inst.query_ascii_values(':SOUR{0}:POW:STAR?;'.format(channel))[0]
		return start_power

	def set_stop_power(self, stop_power_dBm, channel = 1):
		'''
		Set start power for power sweep output power on specified channel.
		
		Start power must be lower than stop power.
		'''
		self.inst.write(':SOUR{0}:POW:STOP {1} ;'.format(channel, stop_power_dBm)) 

	def get_stop_power(self, channel = 1):
		'''
		Get stop power for power sweep output power on specified channel.
		
		Start power must be lower than stop power.
		'''
		stop_power = self.inst.query_ascii_values(':SOUR{0}:POW:STOP?;'.format(channel))[0]
		return stop_power

	def set_IFBW(self, IFBW_Hz,  channel = 1):
		'''
		Change IF bandwidth on specified channel.
		channel can be 1 to 16.
		Can range from 10 to 100,000
		'''
		self.inst.write(':SENS{0}:BAND {1} ;'.format(channel, IFBW_Hz))

	def get_IFBW(self,  channel = 1):
		'''
		Get IF bandwidth on specified channel.
		channel can be 1 to 16.
		'''
		IFBW = self.inst.query_ascii_values(':SENS{0}:BAND?;'.format(channel))[0]
		return IFBW

	def get_averaging_state(self,channel = 1):
		'''
		query if averaging is on or off
		'''
		avg_state = self.inst.query_ascii_values(':SENS{0}:AVER?;'.format(channel), converter = lambda x: str(x).strip('\n'))[0]
		return avg_state

	def get_averaging_count(self,channel = 1):
		'''
		query if averaging is on or off
		'''
		avg_count = self.inst.query_ascii_values(':SENS{0}:AVER:COUNT?;'.format(channel))[0]
		return avg_count

	def turn_on_averaging(self, avg_count, channel = 1):
		'''
		Set up ageraging and  start average.
		'''
		self.average_count =  avg_count
		self.inst.write(':SENS{0}:AVER:COUNT {1};'.format(channel, avg_count))
		self.inst.write(':SENS{0}:AVER ON;'.format(channel))
		self.inst.write(':TRIG:AVER ON;')

	def turn_off_averaging(self, channel = 1):
		self.inst.write(':TRIG:AVER OFF;')
		self.inst.write(':SENS{0}:AVER OFF;'.format(channel))
		
	def query_sweep_time(self, channel = 1, update_timout = True):
		'''
		return sweep  time in seconds. 
		if update_timout == True , make timeout = scan_time + 2 seconds/ 
		'''
		sweep_time = self.inst.query_ascii_values(':SENS{0}:SWE:TIME?;'.format(channel))[0] # returns list type
		avg_count = self.average_count if self.average_count > 0 else 1
		if update_timout:
			#This does not seem to work propely -> workaraound is to set high timeout value in __init__()
			self.inst.timeout = (sweep_time*avg_count*1.50 ) * 1000 # milliseconds
		return sweep_time #seconds

	def setup_single_scan_mode(self, channel = 1):
		'''
		Instrument is set to take one scan at a time upon receipt of bus trigger. 
		Any scan currently underway is aborted.
		'''
		self.inst.write(':ABOR;') # abort current  scan  and wait fo trigger
		self.inst.write(':INIT{0}:CONT OFF;'.format(channel)) #switch to single scan (as opposed to continuous scan)
		self.inst.write(':TRIG:SOUR BUS;');  # sets trigger source to bus trigger ** so that *OPC? tells when measurement is over 

	def setup_continuous_scan_mode(self, channel = 1):
		'''
		Set into continuous scan using internal trigger.
		'''
		self.inst.write(':TRIG:SOUR INT;')
		self.inst.write(':INIT{0}:CONT ON;'.format(channel))

	def set_display_format(self, disp_format = 'MLOG', channel = 1):
		'''
		Set  output display format. Choose between many possible format inc,
		MLOG - Log magnitude
		POL - polar, (real/Imag)
		MLIN - linear magnitude
		... and others.
		'''
		self.inst.write(':CALC{0}:FORM {1};'.format(channel, disp_format))

	def  trigger_single_scan(self, channel = 1):
		'''
		Initializes trigger into its startup state, then asserts a trigger.
		Blocks command line until scan is complete.
		'''
		self.inst.write(':INIT{0}:IMM;'.format(channel))
		self.inst.write(':TRIG:SING;') # trigger single scan
		self._wait_opc()

	def set_num_points_per_scan(self, points_per_scan,  channel = 1):
		self.inst.write(':SENS{0}:SWE:POIN {1} ;'.format(channel, points_per_scan))

	def get_num_points_per_scan(self, channel = 1):
		num_pts = self.inst.query_ascii_values(':SENS{0}:SWE:POIN?;'.format(channel))[0]
		return num_pts

	def set_start_freq(self,f_Hz, channel = 1):
		self.inst.write(':SENS{0}:FREQ:STAR {1:f} ;'.format(channel, f_Hz))

	def get_start_freq(self, channel = 1):
		start_freq = self.inst.query_ascii_values(':SENS{0}:FREQ:STAR?;'.format(channel))[0]
		return start_freq

	def set_stop_freq(self,f_Hz, channel = 1):
		self.inst.write(':SENS{0}:FREQ:STOP {1:f} ;'.format(channel, f_Hz))

	def get_stop_freq(self, channel = 1):
		stop_freq = self.inst.query_ascii_values(':SENS{0}:FREQ:STOP?;'.format(channel))[0]
		return stop_freq

	def set_center_freq(self,f_Hz, channel = 1):
		self.inst.write(':SENS{0}:FREQ:CENT {1:f} ;'.format(channel, f_Hz))

	def auto_scale(self, channel = 1, trace = 1):
		self.inst.write(':DISP:WIND{}:TRAC{}:Y:AUTO;'.format(channel, trace))

	def read_scan_data(self,channel = 1):
		'''
		Data transfer is set to be in 64-bit floating  point binary format. Data from  last scan is queried.
		Readings are of the S-parameter "corrected data"  from the network analyzer (using the CALC{0}:DATA:SDAT? query),
		which is in Re/Im format.   

		Returns two numpy arrays, one for the sparameters and one for the frequencies.
		'''
		self.inst.write(':FORM:DATA REAL;') #Set data transfer format to  64-bit floating point binary
		
		#### NOTE: self.inst.values_format.... settings set before are ignored in the query_binary_values call, 
		####       so am forced to explicitly indicate the settings as keyward arguments for each call.
		frequencies = self.inst.query_binary_values(':SENS{0}:FREQ:DATA?'.format(channel), datatype='d', is_big_endian=True, container = np.array)
		s_parameters_1d_2n = self.inst.query_binary_values(':CALC{0}:DATA:SDAT?'.format(channel), datatype='d', is_big_endian=True, container = np.array)
		#s_parameters_1d_2n is a 1 dimensional array with real and imag parts listed as separate elements. this the array is twice as long as it should be.
		s_parameters_2d_2n = s_parameters_1d_2n.reshape((-1,2)) #-1 indicates to use as many row as necessary to fit the data.
		#s_parameters_1d_2n is a 2 dimensional array with real and imag parts listed as separet columns in each row.
		s_parameters = s_parameters_2d_2n[:,0]
		s_parameters = s_parameters + 1j*s_parameters_2d_2n[:,1]

		return frequencies, s_parameters

class power_supply(Instrument):
	def __init__(self,resource_name, Name = 'LNA power supply'):
		Instrument.__init__(self, resource_name, Name = Name)
		#self.inst.values_format.container = np.array
		self.VdLimit = 3
		self.IdLimit = 0.025
		self.VgLimit = 3

	def on(self):
		self.inst.write('OUTP:STAT ON')

	def off(self):
		self.inst.write('OUTP:STAT OFF')		

	def output_state(self):
		output_state = self.inst.query_ascii_values('OUTP:STAT?', converter = lambda y: False if y == 0.0 else True) #convert value to boolean
		return output_state[0] #query_ascii_values returns something like [True], but er only want True

	def get_Vd(self): 
		Val_Vd = self.inst.query_ascii_values('MEAS:VOlT:DC? P6V')[0]
		return Val_Vd

	def get_Id(self): 
		Val_Id = self.inst.query_ascii_values('MEAS:CURR:DC? P6V')[0]
		return Val_Id

	def get_Vg(self): 
		Val_Vg = self.inst.query_ascii_values('MEAS:VOlT:DC? P25V')[0]
		return Val_Vg

	def set_Vd(self, Vd):
		self.inst.write('INST P6V; VOLT {}'.format(Vd))

	def set_Vg(self, Vg):
		self.inst.write('INST P25V; VOLT {}'.format(Vg))

class synthesizer(Instrument):
	def __init__(self,resource_name, Name = 'Synthesizer'):
		Instrument.__init__(self, resource_name, Name = Name)
		self.settle_time = 0.1 # Seconds    Is the time th  syn needs to be within 1KHz of set frequency

		identification = self.identify()
		if 'MG3692B' in identification: # Anritsu
			self.model = 'MG3692B'
		elif 'E4425B' in identification: # Hewlett-Packard
			self.model = 'E4425B'
		else:
			self.model = None
			print('Model not Recognized.')

	def set_freq(self, freq): 
		if self.model == 'MG3692B':
			self.inst.write('F0 {:12.11f} HZ CF0'.format(freq)) # or e.g. 'F0 %12.11f HZ CF0' %  freq
		elif self.model == 'E4425B':
			self.inst.write(':FREQ {:12.11f} HZ'.format(freq))
		time.sleep(self.settle_time)

	def set_power(self,power):
		if self.model == 'MG3692B':
			self.inst.write('XL0 {:f} DM LO'.format(power))		
		elif self.model == 'E4425B':
			self.inst.write(':POW:AMPL {:f} dBm'.format(power))	
		time.sleep(self.settle_time)

	def on(self):
		if self.model == 'MG3692B':
			self.inst.write('RF1')		
		elif self.model == 'E4425B':
			pass		

	def off(self):
		if self.model == 'MG3692B':
			self.inst.write('RF0')		
		elif self.model == 'E4425B':
			pass

class fridge_DAC(Instrument): #for the Keithley 213
	_allchars = ''.join(chr(i) for i in xrange(256))
	_identity = string.maketrans('','')
	_nondigits_additive = _allchars.translate(_identity,string.digits+"+-.")
	

	def __init__(self,resource_name, Name = 'Keithly 213 voltage supply'):
		Instrument.__init__(self, resource_name, Name = Name)
		self.inst.query_delay = 0.0

	def set_voltage(self, DAC_port, voltage):
		self._print ('Setting voltage to {}'.format(voltage))
		self.inst.write('P{0} C0 A1 X V{1:f} X'.format(DAC_port,voltage)) 
		self._wait_stb()


	def _convert_to_float(self,DAC_reply):
		converted = float(str(DAC_reply).translate(fridge_DAC._identity,fridge_DAC._nondigits_additive))
		return converted

	def read_voltage(self, DAC_port):
		reply = self.inst.query_ascii_values('P{0} C0 A1 X V?'.format(DAC_port) , converter= self._convert_to_float)[0]
		self._wait_stb()
		return reply

	def select_ADC_channel(self,channel):
		self.inst.write('C0 X D{} X'.format(channel))
		self._wait_stb()

	def configure_square_wave(self,DAC_port, DAC_bias,): 
		if DAC_port == 1:
			buffer_location = 0
		elif DAC_port == 2:
			buffer_location = 1024
		else: 
			logging.error('Invalid DAC_Port used in configuring square wave.' )	

		self.inst.write('C3 P{0} A0 N0 T0 X T{0} F{1},2 I20 L{1} B3,{2:08.6f} X B3,-{2:08.6f} X L{1} X @'.format(DAC_port,buffer_location, DAC_bias*100))
		self._wait_stb()

class fridge_ADC(Instrument): #for the IOtech ADC488
	'''
	This is for the IOtech ADC488 digitizer. This is a 16-bit digitizer. 
	Digitized values returned from it are 16-bit signed integers in big_endian format (matlab 'int16', python '>h'). 

	'''
	def __init__(self,resource_name, Name = 'IO Tech Digitizer'):
		Instrument.__init__(self, resource_name, Name = Name)
		self.inst.chunk_size = 2**15 # in bytes
		self.inst.values_format.is_binary = True
		self.inst.values_format.datatype = 'h' #'h' for short,  'f' floats and 'd' double
		self.inst.values_format.is_big_endian = True
		self.inst.values_format.container = np.array

	def read_adc(self,channel, voltage_range,scan_interval, num_readings):
		'''
		Read from  digitizer and converts digitizer units to volts
		channel: Acqusition challel, example, 5 - LHe Level, 6 - MC1, 1 - MF1/MF2 (I), 2 - MF1/MF2 (V), 4 - MF3 (V), 3 - MF3 (I)
		num_readings: number of points in A 
		voltage_range: 0 is +/- 1 Volt range, 1 is +/- 2 Volt range, 2 is +/- 5 Volt range, 3 is +/- 10 Volt range,
		scan_interval: 0 is 10 usec (100 kHz),  5 is 500 usec (2 kHz),

		returns numpy array of readings
		'''
		self.inst.write('B-OX')
		self._wait_stb()
		self.inst.clear()
		self.inst.write('A0C{0}R{1}I{2}N0,{3}L0T1G9X'.format(channel, voltage_range, scan_interval, num_readings))
		self._wait_stb(code = 48)
		self.inst.assert_trigger()
		self._wait_stb()
		self.inst.write('B-OX')
		self._wait_stb()
		data_string = self.inst.read_raw() 
		self._wait_stb()
		self.inst.clear()
		self._wait_stb()

		data_array = np.array(self._convertBinData16(data_string))
		if voltage_range == 0:
			data_array = 0.000033 * data_array
		elif voltage_range == 1:
			data_array = 0.000066 * data_array
		elif voltage_range == 2:
			data_array = 0.000166 * data_array
		elif voltage_range == 3:
			data_array = 0.000333 * data_array
		else:
			logging.warning('Invalid voltage_range provided.')

		return data_array

	def _convertBinData16(self,binDataString):
		""" Converts binary data string to list of integers.
		Assumes binary dats is 16bit signed ints ('>h') . 
		from: Z:\backup_cedar\data\mkids\readout\python\MKIDs_Data_Automation\mkid.py as of Feb 2016
		"""
		data = []
		for i in range(len(binDataString)/4):
			data.append(struct.unpack('>h', binDataString[i*4:i*4+2])[0])
			data.append(struct.unpack('>h', binDataString[i*4+2:i*4+4])[0])

		return data

