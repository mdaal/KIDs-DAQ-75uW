#import numpy as np

# def construct_configuration():
	# Channel_Name = dict(
	# 	MF1 = {'DAC_Port' : 1, 'ADC_I_Channel' : 1, 'ADC_V_Channel' : 2, 'MUX_Channel' : 32},
	# 	MF2 = {'DAC_Port' : 1, 'ADC_I_Channel' : 1, 'ADC_V_Channel' : 2, 'MUX_Channel' : 48},
	# 	MF3 = {'DAC_Port' : 2, 'ADC_I_Channel' : 4, 'ADC_V_Channel' : 3, 'MUX_Channel' : 36},
	# 	MU2 = {'DAC_Port' : 2, 'ADC_I_Channel' : 4, 'ADC_V_Channel' : 3, 'MUX_Channel' : 32},
	# 	CU  = {'DAC_Port' : 2, 'ADC_I_Channel' : 4, 'ADC_V_Channel' : 3, 'MUX_Channel' : 33},
	# 	MC1 = {'DAC_Port' :'', 'ADC_I_Channel' :'', 'ADC_V_Channel' : 6, 'MUX_Channel' :  0},
	# 	Stack_Heater  = {'DAC_Port' : 3},
	# 	Sample_Heater = {'DAC_Port' : 4},
	# 	LHe_Monitor   = {'ADC_V_Channel' : 5},
	# 	)

# class channel_element:
# 	def __init__(self, DAC_Port, ADC_I_Channel, ADC_V_Channel, MUX_Channel, Channel_Name):
# 		self.DAC_Port = DAC_Port
# 		self.ADC_I_Channel = ADC_I_Channel
# 		self.ADC_V_Channel = ADC_V_Channel
# 		self.MUX_Channel = MUX_Channel
# 		self.Channel_Name = Channel_Name

class Channel_Config:
	config_list = []
	class channel_element:
		def __init__(self, DAC_Port, ADC_I_Channel, ADC_V_Channel, MUX_Channel, Channel_Name):
			self.DAC_Port = DAC_Port
			self.ADC_I_Channel = ADC_I_Channel
			self.ADC_V_Channel = ADC_V_Channel
			self.MUX_Channel = MUX_Channel
			self.Channel_Name = Channel_Name
	MF1 = channel_element(1,1,2,32, 'MF1'); config_list.append(MF1)
	MF2 = channel_element(1,1,2,48, 'MF2'); config_list.append(MF2)
	MF3 = channel_element(2,3,4,37, 'MF3'); config_list.append(MF3)
	MU2 = channel_element(2,4,3,32, 'MU2'); config_list.append(MU2)
	CU 	= channel_element(2,3,4,33, 'CU'); config_list.append(CU)
	MC1 = channel_element(None,None,6,0,'MC1'); config_list.append(MC1)
	Stack_Heater = channel_element(3,None,None,None, 'Stack_Heater'); config_list.append(Stack_Heater)
	Aux = channel_element(4,None,None,None, 'Aux'); config_list.append(Aux) # Usually the Sample_Heater, or Helmholtz_Coil
	LHe_Monitor = channel_element(None,None,5,None, 'LHe_Monitor'); config_list.append(LHe_Monitor)
	config_dict = {}

	for config in config_list:
		config_dict[config.Channel_Name] = config 






class Channel_Calibration:
	# def __init__(self):
	# 	pass

	def _chebyshev_temp( ZL, ZU, Coefficients):
		'''
		Returns a function for evaluating the temperature corresponfing to a resistance
		'''
		Z = np.log10 # Is a function

		X = lambda r: ((Z(r)-ZL) - (ZU-Z(r)))/(ZU-ZL)

		An = np.array(Coefficients)
		n = np.arange(An.size)
		Temp = lambda r: (An*np.cos(n * np.arccos(X(r)))).sum()

		return Temp

	def _NTD_temp(D, alpha, R0 ):
		Temp = lambda r:  D *np.power(np.log(r/R0), -1.0/alpha)
		return  Temp

	class device_calibration:
		def __init__(self, name, R = None):
			self.Name = name
			self.R = R
		#def merge_channel_element(channel_element):

	class thermometer_calibration(object):
		def __init__(self, name, serial, temp_conversion):
			self.Name = name
			self.Serial = serial
			self.Temp_Conversion =  temp_conversion
			#thermometer_calibration.devices.update[serial] = self
			
	device_dict = {}
	device_list = []
	#GE1
	R1  = _chebyshev_temp( 1.66224576693, 4.53551045257, [0.325048,-0.447054,0.258502,-0.134858,0.066405,-0.030897,0.013660,-0.005960,0.002546,-0.001058,0.000555,0.000371] )
	R2  = _chebyshev_temp( 1.35888610193, 1.77332277922, [2.523400,-2.186563,0.839245,-0.336767,0.158959,-0.077854,0.041830,-0.025260,0.012299,-0.010317,0.002461,0.004326] )
	#Serial_Number.update( {'U24569' : {'Name' : 'Ge1', 'Temp_Conversion' : [34317, R1, 55, R2, 23 ]}})
	#Serial_Number.update( U24569 = {'Name' : 'Ge1', 'Temp_Conversion' : [34317, R1, 55, R2, 23 ]})
	# Formating  for 'Temp_Conversion' is [Low Temp Resistance Limit, R1, corossover resistance for R1 and R2, R2,High Temp Resistance Limit]
	U24569 = thermometer_calibration('Ge1','U24569', [34317, R1, 55, R2, 23 ])
	#devices['U24569'] = U24569
	device_list.append(U24569)
	#devices['U24569'] =  thermometer_calibration('Ge1','U24569', [34317, R1, 55, R2, 23 ])

	#RuOx2
	R1  = _chebyshev_temp( 3.35043604264766, 3.97381109866825, [1.37112002043110E-01,-1.85309901503749E-01,9.33634620145133E-02,-3.79685465244926E-02,1.38327895621806E-02,-5.11299739948956E-03,1.92082204900090E-03,-7.00339148458662E-04] )
	R2  = _chebyshev_temp( 3.17250619373603, 3.40079693473316, [1.77883468946374E+00,-2.17372002307393E+00,9.87118526749293E-01,-3.86026531789366E-01,1.33506015911897E-01,-3.89806321228718E-02,7.92880598631056E-03,-1.14233167749821E-03] )
	R3  = _chebyshev_temp( 3.06598038475416, 3.18838096295415, [1.89362536909991E+01,-1.93969096504693E+01,5.31242495426572E+00,-1.06166721403071E+00,2.54584718972344E-01,-4.52253502270920E-02,5.64879074832310E-03,-8.73354487268332E-03,3.71836422648148E-03] ) 
	#Serial_Number.update( U03746 = {'Name' : 'RuOx2', 'Temp_Conversion' : [8348.7985, R1, 2398.72, R2, 1518.3499, R3, 1179.9478 ]}) 
	U03746 =  thermometer_calibration('RuOx2','U03746',  [8348.7985, R1, 2398.72, R2, 1518.3499, R3, 1179.9478 ])
	#devices['U03746'] = U03746
	device_list.append(U03746)
	#devices['U03746'] = thermometer_calibration('RuOx2','U03746',  [8348.7985, R1, 2398.72, R2, 1518.3499, R3, 1179.9478 ])
	
	#GE3
	R1  = _chebyshev_temp( 1.67464991024243, 5.43941164928578, [3.07370635054904E-01,-4.17581598982993E-01,2.36202025691414E-01,-1.22979724374782E-01,6.03350390826610E-02,-2.90249667045680E-02,1.37935697090126E-02,-6.23184609631060E-03,3.11914623113467E-03,-1.57005231284113E-03,5.30697611627237E-04,-6.21623977893971E-05] )
	R2  = _chebyshev_temp( 1.27820783945640, 1.72841590038206, [2.62392317538138E+00,-2.15262148674796E+00,8.15269134162476E-01,-3.30926642817096E-01,1.49789168099246E-01,-7.38230220952596E-02,3.80727849706133E-02,-2.15369441522447E-02,1.10002418207796E-02,-7.65929116851400E-03,2.60216254876841E-03,-3.25735860650095E-03] )
	#Serial_Number.update( U30593 = {'Name' :'Ge3', 'Temp_Conversion' : [140400, R1, 48.9786, R2, 19.9707]})
	U30593 =  thermometer_calibration('Ge3','U30593', [140400, R1, 48.9786, R2, 19.9707])
	#devices['U30593'] = U30593
	device_list.append(U30593)
	#devices['U30593'] = thermometer_calibration('Ge3','U30593', [140400, R1, 48.9786, R2, 19.9707])
	
	#GE4
	R1  = _chebyshev_temp(1.53742238419421, 5.18055594070364, [2.98682446649821E-01,-4.11876193785050E-01,2.38450889551739E-01,-1.26954942301777E-01,6.35912168199612E-02,-3.11348926123644E-02,1.48807594246117E-02,-7.19457771191232E-03,3.55645228136590E-03,-1.42359987432806E-03,9.64611840647526E-04,-6.12664551763595E-05,3.75288951720578E-04])
	R2  = _chebyshev_temp(1.17340826002934, 1.58485482308996, [2.61568062082427E+00,-2.14243644095415E+00,8.15257861223272E-01,-3.31227533343859E-01,1.54103942850631E-01,-7.61667656314401E-02,4.03443704840625E-02,-2.36715428346808E-02,1.13496557181916E-02,-1.04445104539317E-02,1.45298723522177E-03,-4.12296876196607E-03])
	#Serial_Number.update( U30817 = {'Name' :'Ge4', 'Temp_Conversion' : [60609.163, R1, 35.30, R2, 15.60]})
	U30817 = thermometer_calibration('Ge4', 'U30817',  [60609.163, R1, 35.30, R2, 15.60])
	#devices['U30817'] = U30817
	device_list.append(U30817)

	#CU
	R1 = _NTD_temp(6.12,0.25, 360.6)
	#Serial_Number.update( CU =  {'Name' :'CU', 'Temp_Conversion' : [12146, R1, 526]}) # 12146 Ohms --> 0.040 Kelvin and 526 --> 300 Kelvin
	CU = thermometer_calibration('CU', 'CU', [12146, R1, 526])
	#devices['CU'] = CU
	device_list.append(CU)

	#MC1
	R1 = _NTD_temp(6.12, 0.25, 360.6) # The following callibration is copied from CU
	#Serial_Number.update( MC1 =  {'Name' :'MC1', 'Temp_Conversion' : [12146, R1, 526]}) # 12146 Ohms --> 0.040 Kelvin and 526 --> 300 Kelvin
	MC1 = thermometer_calibration('MC1', 'MC1',[12146, R1, 526])
	#devices['MC1'] = MC1
	device_list.append(MC1)

	#H1 - Original Straight Sample Heater
	#Serial_Number.update( H1 =  {'Name' : 'H1', 'R': 20000}) #R = 20kOhm
	H1 = device_calibration('H1', R = 20000) #R = 20kOhm
	#devices['H1'] = H1 
	device_list.append(H1)

	#H2 - Right Angle Sample Heater
	#Serial_Number.update( H2 =  {'Name' : 'H2', 'R': 20000}) #R = 20kOhm
	H2 =  device_calibration('H2', R = 20000) #R = 20kOhm
	#devices['H2'] = H2
	device_list.append(H2)

	#Stack_Heater_1
	#Serial_Number.update( Stack_Heater_1 = {'Name' : 'Stack_Heater_1', 'R': 1500}) #R = 1.5kOhm
	Stack_Heater_1 =  device_calibration('Stack_Heater_1', R = 1500) #R = 1.5kOhm
	#devices['Stack_Heater_1']  = Stack_Heater_1
	device_list.append(Stack_Heater_1)

	#Helmholtz Coil
	Helmholtz_Coil =  device_calibration('Helmholtz_Coil', R = 20)
	#devices['Helmholtz_Coil']  = Helmholtz_Coil
	device_list.append(Helmholtz_Coil)

	#LHe_Monitor
	#Serial_Number.update( LHe_Monitor = {'Name' : 'LHe_Monitor', 'Level': lambda data_mean: (0.532 - 10 * data_mean)*191.2}) 
	LHe_Monitor =  device_calibration('LHe_Monitor')
	LHe_Monitor.Level_Conversion = lambda data_mean: (0.532 - 10 * data_mean)*191.2
	#devices['LHe_Monitor'] = LHe_Monitor
	device_list.append(LHe_Monitor)

	for device in device_list:
		device_dict[device.Name] = device



channel_config = Channel_Config()
channel_calibration = Channel_Calibration()
#	return channel_calibration, channel_config


#channel_calibration, channel_config = construct_configuration()


