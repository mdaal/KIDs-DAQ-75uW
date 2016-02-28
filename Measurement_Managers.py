import Instruments
import Fridge_Interfaces

class measurement_manager:
	
	def __init__(self, devices):
		self.fi = Fridge_Interfaces.fridge_interface(devices)
		self.inl = Instruments.Instrument_Name_List
		self.na = Instruments.network_analyzer(inl.NETWORK_ANALYZER_E5071B)

	def single_scan(self):
		pass
