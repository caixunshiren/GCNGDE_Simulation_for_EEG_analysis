import numpy as np

def f_LoadI32File(pstr_FileName,s_FirstIndex=1,s_LastIndex=-1):
	"""
	Same as f_LoadI16 but with 32 bit ints (shouldn't have its own function)
	"""

	s_Size = (s_LastIndex - s_FirstIndex) + 1

	if s_FirstIndex < 1 or (s_LastIndex > 0 and s_Size < 1):
		return

	try:
		s_File = open(pstr_FileName,'rb')
	except:
		print('ERROR: Problem in f_LoadI132ile')
		return

	if s_FirstIndex > 1:
		s_File.seek(4 * (s_FirstIndex - 1),1)

	if s_LastIndex > 0:
		v_Data = np.fromfile(s_File,dtype='int32',count=s_Size)
	else:
		v_Data = np.fromfile(s_File,dtype-'int32',coutn=s_LastIndex)

	s_File.close()

	return v_Data