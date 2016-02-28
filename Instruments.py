import visa
import numpy as np
import time 
import logging
import string
import struct

class Instrument_Name_List:
	NETWORK_ANALYZER_E5071B = 'TCPIP0::169.229.225.4::inst0::INSTR'
	POWER_SUPPLY_HPE3631A = 'GPIB0::14::INSTR'
	SYNTHESIZER_MG3692B = 'GPIB0::7::INSTR'
	SYNTHESIZER_E4425B = ''
	DIGITIZER_ADC488 = 'GPIB1::14::INSTR'
	VOLTAGE_SOURCE_K213 = 'GPIB1::17::INSTR'

	


class Instrument(object):
	rm = visa.ResourceManager()

	def __init__(self, resource_name):
		#self.rm = visa.ResourceManager()
		self.inst = Instrument.rm.open_resource(resource_name, open_timeout=0)
		self.inst.timeout = 10000 # milliseconds
		self.resource_name = resource_name

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
				print(self.inst.read_stb())
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
	#before_close()

class network_analyzer(Instrument):
	def __init__(self,resource_name):
		Instrument.__init__(self, resource_name)
		self.inst.chunk_size = 2**15 # in bytes
		self.inst.timeout = 25000 #milliseconds, None is infinite timeout  NOTE: This number applies to queries
		self.inst.values_format.is_binary = True
		self.inst.values_format.datatype = 'd' # 'f' floats and 'd' double
		self.inst.values_format.is_big_endian = True
		self.inst.values_format.container = np.array

	def on(self):
		self.inst.write('OUTP ON ;')

	def off(self):
		self.inst.write('OUTP OFF ;')

	def set_power(self, power_dBm, channel = 1):
		'''
		Change output power on specified channel.
		'''
		self.inst.write(':SOUR{0}:POW {1} ;'.format(channel, power_dBm))

	def set_IFBW(self, IFBW_Hz,  channel = 1):
		'''
		Change IF bandwidth on specified channel.
		channel can be 1 to 16.
		'''
		self.inst.write(':SENS{0}:BAND {1} ;'.format(channel, IFBW_Hz))

	def turn_on_averaging(self, avg_count, channel = 1):
		'''
		Set up ageraging and  start average.
		'''
		self.inst.write(':SENS{0}:AVER:COUNT {1};'.format(channel, avg_count))
		self.inst.write(':SENS{0}:AVER ON;'.format(channel))
		self.inst.write(':TRIG:AVER ON;')

	def turn_off_averageing(self, channel = 1):
		self.inst.write(':TRIG:AVER OFF;')
		self.inst.write(':SENS{0}:AVER OFF;'.format(channel))
		
	def query_sweep_time(self, channel = 1, update_timout = True):
		'''
		return sweep  time in seconds. 
		if update_timout == True , make timeout = scan_time + 2 seconds/ 
		'''
		sweep_time = self.inst.query_ascii_values(':SENS{0}:SWE:TIME?;'.format(channel)) # returns list type
		if update_timout:
			self.inst.timeout = (sweep_time[0] + 2.0) * 1000 # milliseconds
		return sweep_time[0] #seconds

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

	def set_start_freq(self,f_Hz, channel = 1):
		self.inst.write(':SENS{0}:FREQ:STAR {1} ;'.format(channel, f_Hz))

	def set_stop_freq(self,f_Hz, channel = 1):
		self.inst.write(':SENS{0}:FREQ:STOP {1} ;'.format(channel, f_Hz))

	def set_center_freq(self,f_Hz, channel = 1):
		self.inst.write(':SENS{0}:FREQ:CENT {1} ;'.format(channel, f_Hz))

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
	def __init__(self,resource_name):
		Instrument.__init__(self, resource_name)
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
		Val_Vd = self.inst.query_ascii_values('MEAS:VOlT:DC? P6V')
		return Val_Vd

	def get_Id(self): 
		Val_Id = self.inst.query_ascii_values('MEAS:CURR:DC? P6V')
		return Val_Id

	def get_Vg(self): 
		Val_Vg = self.inst.query_ascii_values('MEAS:VOlT:DC? P25V')
		return Val_Vg

	def set_Vd(self, Vd):
		self.inst.write('INST P6V; VOLT {}'.format(Vd))

	def set_Vg(self, Vg):
		self.inst.write('INST P25V; VOLT {}'.format(Vg))

class synthesizer(Instrument):
	def __init__(self,resource_name):
		Instrument.__init__(self, resource_name)
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

	def set_power(self,power):
		if self.model == 'MG3692B':
			self.inst.write('XL0 {:f} DM LO'.format(power))		
		elif self.model == 'E4425B':
			self.inst.write(':POW:AMPL {:f} dBm'.format(power))	

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

	def __init__(self,resource_name):
		Instrument.__init__(self, resource_name)
		self.inst.query_delay = 0.0

	def set_voltage(self, DAC_port, voltage):
		print('Setting voltage to {}'.format(voltage))
		self.inst.write('P{0} C0 A1 X V{1:f} X'.format(DAC_port,voltage)) 
		self._wait_stb()


	def _convert_to_float(self,DAC_reply):
		converted = float(str(DAC_reply).translate(fridge_DAC._identity,fridge_DAC._nondigits_additive))
		return converted

	def read_voltage(self, DAC_port):
		reply = self.inst.query_ascii_values('P{0} C0 A1 X V?'.format(DAC_port) , converter= self._convert_to_float)
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
	def __init__(self,resource_name):
		Instrument.__init__(self, resource_name)
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
		"""
		data = []
		for i in range(len(binDataString)/4):
			data.append(struct.unpack('>h', binDataString[i*4:i*4+2])[0])
			data.append(struct.unpack('>h', binDataString[i*4+2:i*4+4])[0])

		return data

	# def parse_binary(bytes_data, is_big_endian=False, is_single=False):
	#     """Parse ascii data and return an iterable of numbers.
	#     To be deprecated in 1.7
	#     :param bytes_data: data to be parsed.
	#     :param is_big_endian: boolean indicating the endianness.
	#     :param is_single: boolean indicating the type (if not is double)
	#     :return:
	#     """
	#     data = bytes_data

	#     hash_sign_position = bytes_data.find(b"#")
	#     if hash_sign_position == -1:
	#         raise ValueError('Could not find valid hash position')

	#     if hash_sign_position > 0:
	#         data = data[hash_sign_position:]

	#     data_1 = data[1:2].decode('ascii')

	#     if data_1.isdigit() and int(data_1) > 0:
	#         number_of_digits = int(data_1)
	#         # I store data and data_length in two separate variables in case
	#         # that data is too short.  FixMe: Maybe I should raise an error if
	#         # it's too long and the trailing part is not just CR/LF.
	#         data_length = int(data[2:2 + number_of_digits])
	#         data = data[2 + number_of_digits:2 + number_of_digits + data_length]
	#     else:
	#         data = data[2:]
	#         if data[-1:].decode('ascii') == "\n":
	#             data = data[:-1]
	#         data_length = len(data)

	#     if is_big_endian:
	#         endianess = ">"
	#     else:
	#         endianess = "<"

	#     try:
	#         if is_single:
	#             fmt = endianess + str(data_length // 4) + 'f'
	#         # if is_short:
	#         #     fmt = endianess + str(data_length // 2) + 'h'
	#         else:
	#             fmt = endianess + str(data_length // 8) + 'd'

	#         result = list(struct.unpack(fmt, data))
	#     except struct.error:
	#         raise ValueError("Binary data itself was malformed")

	#     return result