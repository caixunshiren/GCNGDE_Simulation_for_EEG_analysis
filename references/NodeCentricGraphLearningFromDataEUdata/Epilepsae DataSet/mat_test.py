
import matlab.engine


matlab_engin = matlab.engine.start_matlab()
X, Y, conv_sizes, sel_win_nums, clip_sizes = \
                matlab_engin.python_online_wrapper(253, 100, nargout=5)