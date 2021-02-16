import numpy as np

def f_LoadI16File(pstr_FileName,s_FirstIndex = 1,s_LastIndex = -1):
	"""
	Loads integer signal from a file where each data value is stored in 2 bytes.

	Inputs:
		pstr_FileName: name of file
		ps_FirstIndex: number of the first sample to read from
		ps_LastIndex: number of last sample (default end of file)

		Outputs:
			v_Data: loaded data
	"""

	s_Size = (s_LastIndex - s_FirstIndex) + 1
	if s_FirstIndex < 1 or (s_LastIndex > 0 and s_Size < 1):
		print('ERROR: Problem in f_LoadI16File')
		return

	try:
		s_File = open(pstr_FileName,'rb')
	except:
		raise Exception('ERROR: problem opening data file')

	if s_FirstIndex > 1:
		s_File.seek(2 * (s_FirstIndex - 1),1)

	if s_LastIndex < 0:
		v_Data = np.fromfile(s_File,dtype = 'int16',count = s_LastIndex)
	else:
		v_Data = np.fromfile(s_File,dtype = 'int16',count = s_Size)

	s_File.close()

	return v_Data