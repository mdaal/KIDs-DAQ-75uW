import Instruments
import numpy as np
import logging
import matplotlib as mpl
import matplotlib.pyplot as plt

configuration_file =  "fridge_interface_configuration.py"

class fridge_interface:
	#def construct_configuration(self):
	with open(configuration_file) as f:
		code = compile(f.read(),configuration_file, 'exec')
		exec(code)
	del(code, f)

	def __init__(self, devices):
		'''
		thermometers is a list of tuples: (device/thermoeter name, channel name) as strings
		'''
		#self.construct_configuration()
		self.Max_Stack_Heater_Voltage = 3.1 # Volts
		self.Max_Aux_Channel_Voltage = 10.0  # Volts
		self.DAC_Bias = .002 #Volts
		self.Rb = 100000. #Ohms
		self.df = 0.
		self.f = 1.
		self.Sample_Heater_Voltage = 0.0 #Volts
		self.Stack_Heater_Voltage = 0.0 #Volts
		self.Thermometers = {}
		self.Diagnostic_Plots = True
		inl = Instruments.Instrument_Name_List
		self.ADC= Instruments.fridge_ADC(inl.DIGITIZER_ADC488)
		self.DAC = Instruments.fridge_DAC(inl.VOLTAGE_SOURCE_K213)

		
		self.thermometer_list = []
		self.device_list = []
		thermometer_list_index = 0 
		device_list_index = 0
		self.stack_heater_index = None
		self.aux_channel_index = None
		for device in devices:
			if isinstance(fridge_interface.channel_calibration.device_dict[device[0]], fridge_interface.channel_calibration.thermometer_calibration):
				fridge_interface.channel_calibration.device_dict[device[0]].__dict__.update(fridge_interface.channel_config.config_dict[device[1]].__dict__)
				self.thermometer_list.append(fridge_interface.channel_calibration.device_dict[device[0]])
				thermometer_list_index += 1
			elif  isinstance(fridge_interface.channel_calibration.device_dict[device[0]], fridge_interface.channel_calibration.device_calibration):
				fridge_interface.channel_calibration.device_dict[device[0]].__dict__.update(fridge_interface.channel_config.config_dict[device[1]].__dict__)
				self.device_list.append(fridge_interface.channel_calibration.device_dict[device[0]])
				if fridge_interface.channel_calibration.device_dict[device[0]].Channel_Name == 'Stack_Heater':
					self.stack_heater_index = device_list_index
				elif fridge_interface.channel_calibration.device_dict[device[0]].Channel_Name == 'Aux':
					self.aux_channel_index = device_list_index
				device_list_index += 1
			else:
				logging.error('Unrecognized device')


		for device in self.device_list:
			voltage = self.DAC.read_voltage(device.DAC_Port)
			print('device {} has voltage set to {} V'.format(device.Name, voltage[0]))


	def set_thermometer_bias_voltage(self, bias_voltage):
		if np.abs(bias_voltage) > 0.2:
			logging.warning('Attempt to set thermomter bias to {} V. Setting to 0.02 V instead.'.format(bias_voltage))
		else: 
			self.DAC_Bias =  bias_voltage
			print('Themometer bias set to {} V.'.format(bias_voltage))

	def set_stack_heater_voltage(self,voltage):
		if self.stack_heater_index ==None:
			logging.error('Attempt to set stack heater voltage when no stack heater is configured')			
		elif  np.abs(voltage) > self.Max_Stack_Heater_Voltage:	
			logging.warning('Attempt to set stack heater to {} V. Setting to {} V instead.'.format(voltage, self.Max_Stack_Heater_Voltage))
			#self.DAC.set_voltage(fridge_interface.channel_config.Stack_Heater.DAC_Port, self.Max_Stack_Heater_Voltage)
			self.DAC.set_voltage(self.device_list[self.stack_heater_index].DAC_Port, self.Max_Stack_Heater_Voltage)
		else:
			#self.DAC.set_voltage(fridge_interface.channel_config.Stack_Heater.DAC_Port, heater_voltage)
			self.DAC.set_voltage(self.device_list[self.stack_heater_index].DAC_Port, voltage)

	def set_aux_channel_voltage(self,voltage):
		if self.aux_channel_index ==None:
			logging.error('Attempt to set aux channel when no aux channel is configured')	
		elif  np.abs(voltage) > self.Max_Aux_Channel_Voltage:	
			logging.warning('Attempt to set aux channel to {} V. Setting to {} V instead.'.format(voltage, self.Max_Aux_Channel_Voltage))
			#self.DAC.set_voltage(fridge_interface.channel_config.Sample_Heater.DAC_Port, self.Max_Sample_Heater_Voltage)
			self.DAC.set_voltage(self.device_list[self.aux_channel_index_index].DAC_Port, self.Max_Stack_Heater_Voltage)
		else:
			#self.DAC.set_voltage(fridge_interface.channel_config.Sample_Heater.DAC_Port, voltage)
			self.DAC.set_voltage(self.device_list[self.aux_channel_index].DAC_Port, voltage)

	def read_LHe_level(self):
		num_readings = 1000
		values = self.ADC.read_adc(fridge_interface.channel_config.LHe_Monitor.ADC_V_Channel, 0,0, num_readings) 

		processed_values = self._process_data(values, 'two_sigma')

		level = fridge_interface.channel_calibration.LHe_Monitor.Level_Conversion(processed_values.mean()) # need to confirm constants : 0.532 and 191.2

		if self.Diagnostic_Plots: # issue '%matplotlib qt' in spyder
			fig = plt.figure(figsize = (8,6), dpi = 100)
			ax = fig.add_subplot(111)
			line = ax.plot(values,'g-', label = 'Raw Readings')
			line = ax.plot(processed_values,'b-', label = 'Within 2$\sigma$')

			ax.set_ylabel('Voltage [V]')
			ax.set_title('LHe Level Reading: {:.1f}%'.format(level))
			
			ax.legend(loc = 'best', fontsize = 9)

			ax.grid()

			plt.show()
		
		return level

	def _process_data(self,data_in,process_method):
		'''
		returns the elements of Data_In which lie within num_sigma
		standard deviations from the mean of data_dn.

		data_in is a numpy array
		'''
		m = data_in.mean()
		sigma = data_in.std()
		num_sigma = 2

		if process_method == 'two_sigma':
			data_out = data_in[(data_in < (m + num_sigma*sigma)) & (data_in > (m - num_sigma*sigma))]
		else:
			logging.error('invalid process method.')

		return data_out

	def read_MC1(self):
		num_readings = 200
		values =  self.ADC.read_adc(fridge_interface.channel_config.MC1.ADC_V_Channel, 3,0, num_readings)
		Temp = (values*10-0.03).mean() * 1000 #ohms
		#print('Resistance is {}, corresponding to a temp of {}')
		return Temp

	def read_temp(self, thermometer_list_index):
		'''
		Note Num_Readings * (ADC Scan_Interval) = (i) * (DAC Voltage Interval * 2 for quare wave)
		e.g. Scan_Interval = 'I5' = 500 usec/point, DAC Voltage Intergal = 'I20' = 20 msec, so wave period is 40 msec
		Then 400 point * 500 usec/point = 200000 usec reading  -> wavelengths
		this consideration is not so critical anymore since we
		correct the mean with medians of positive and negative point!
		'''
		num_readings = 400
		n = thermometer_list_index

		#switch mux to proper channel
		self.DAC.select_ADC_channel(self.thermometer_list[n].MUX_Channel)
		
		#Set up square wave form
		self.DAC.configure_square_wave(self.thermometer_list[n].DAC_Port, self.DAC_Bias)

		#Read I Waveform
		I =  self.ADC.read_adc(self.thermometer_list[n].ADC_I_Channel, 3,5,num_readings)

		#Read V Waveform
		V =  self.ADC.read_adc(self.thermometer_list[n].ADC_V_Channel, 3,5,num_readings)

		# Clean up noisey date,  and find aplitude of crest and trough
		ii = I - I.mean()
		ipp = ii[ii > 0]
		inn = ii[ii < 0]
		im = (np.median(ipp) + np.median(inn))/2.0 

		vv = V - V.mean()
		vpp = vv[vv > 0]
		vnn = vv[vv < 0]
		vm = (np.median(vpp) + np.median(vnn))/2.0 

		i = np.abs(ii-im)
		v = np.abs(vv-vm)

		medi = np.median(i)
		medv = np.median(v)	

		#Resistance measured
		R = medv / (medi/self.Rb)

		if self.Diagnostic_Plots: # issue '%matplotlib qt' in spyder
			fig = plt.figure(figsize = (8,6), dpi = 100)
			ax = fig.add_subplot(111)
			I_line = ax.plot(I,'k-', label = 'I')
			V_line = ax.plot(V,'r-', label = 'V')
			

			i_line = ax.plot(i,'k:', label = 'I amplitude')
			v_line = ax.plot(v,'r:', label = 'V amplitude')

			medv_line = ax.plot([0, num_readings],[medi, medi],'k-.', linewidth = 3, label = 'I median amp')
			medv_line = ax.plot([0, num_readings],[medv, medv],'r-.', linewidth = 3, label = 'V median amp')

			ax.set_ylabel('Voltage [V]')
			ax.set_title('Thermometer {} on channel {}, R = {:.3f} $\Omega$'.format(self.thermometer_list[n].Name, self.thermometer_list[n].Channel_Name, R))
			
			ax.legend(loc = 'best', fontsize = 9)

			ax.grid()

			plt.show()

	
		#Determine Temperature from R...

		# tc is the Resistance to Temp conversion function
		tc = self.thermometer_list[n].Temp_Conversion

		if R > tc[0]:
			Temp = -1 #Resistance is too high and out of calibration range
		elif R < tc[-1]:
			Temp = -2 #Resistance is too low and out of calibration range
		else:
			for j in range(len(tc))[1:-1:2]: #slicing notation a[start:stop:step]
				if (R < tc[j-1]) & (R > tc[j+1]):
					Temp = tc[j](R) # tc[j]() is the Temp Converison function
					break
				else:
					Temp = -100 # someting has gone wrong

		return Temp 



