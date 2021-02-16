# -*- coding: utf-8 -*-
import pdb,traceback,sys
import bin_file
reload(bin_file)
import f_LoadI16File
import f_LoadI32File
reload(f_LoadI16File)
reload(f_LoadI32File)

if __name__ == '__main__':
    try:
        bf = bin_file.bin_file('data.data','head.head') # correct
        #=> get_header info correct
        data,time = bf.def_data_access(5,1,['RFA1','RFA2','RFA3'],2) # correct
        #=> get_bin_signals correct
        #=> get_next window correct
            #=> get_previous window correct
        #=> redefine_data_access correct
        bf.set_elect_ave(['RFA1','RFA2','RFA3'])
    except:
        type,value,tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)