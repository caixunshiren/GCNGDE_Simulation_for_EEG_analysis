"""
crossbary.py
Louis Primeau
University of Toronto Department of Electrical and Computer Engineering
louis.primeau@mail.utoronto.ca
July 29th 2020
Last updated: March 18th 2021

https://github.com/louisprimeau/node-crossbar/blob/main/networks/crossbar/crossbar.py

"""

"""
Circuit Solver taken from:
A Comprehensive Crossbar Array Model With Solutions for Line Resistance and Nonlinear Device Characteristics
An Chen
IEEE TRANSACTIONS ON ELECTRON DEVICES, VOL. 60, NO. 4, APRIL 2013
"""

"""
crossbar.py
Louis Primeau
University of Toronto Department of Electrical and Computer Engineering
louis.primeau@mail.utoronto.ca
July 29th 2020
Last updated: May 22nd 2021
"""

"""
Circuit Solver taken from:
A Comprehensive Crossbar Array Model With Solutions for Line Resistance and Nonlinear Device Characteristics
An Chen
IEEE TRANSACTIONS ON ELECTRON DEVICES, VOL. 60, NO. 4, APRIL 2013
"""

import torch
import numpy as np
import itertools
import time

class crossbar:
    def __init__(self, device_params, deterministic=False):

        # Useful for debugging
        self.deterministic = deterministic

        # Power Supply Voltage
        self.V = device_params["Vdd"]

        # DAC resolution
        self.input_resolution = device_params["dac_resolution"]
        self.output_resolution = device_params["adc_resolution"]

        # Wordline Resistance 
        self.r_wl = torch.Tensor((device_params["r_wl"],))

        # Bitline Resistance
        self.r_bl = torch.Tensor((device_params["r_bl"],))

        # Number of rows, columns
        self.size = device_params["m"], device_params["n"]

        # Crossbar conductance model
        self.method = device_params["method"]

        # Device Programming Error approximation
        # 'linear' programming assumes that there is are 2^resolution states between the on and off resistances of the device.
        # Any number programmed onto the crossbar is rounded to one of those states.
        if (self.method == "linear"):

            if self.deterministic:
                self.g_on = torch.ones(self.size) / device_params["r_on"]
                self.g_off =  torch.ones(self.size) / device_params["r_off"]
            else:
                self.g_on = 1 / torch.normal(device_params["r_on"], device_params["r_on_stddev"], size=self.size)
                self.g_off = 1 / torch.normal(device_params["r_off"], device_params["r_off_stddev"], size=self.size)

            # Resolution
            self.resolution = device_params["device_resolution"]
            self.conductance_states = torch.cat([torch.cat([torch.linspace(self.g_off[i,j], self.g_on[i,j],2**self.resolution - 1).unsqueeze(0)
                                                        for j in range(self.size[1])],dim=0).unsqueeze(0)
                                             for i in range(self.size[0])],dim=0)

        # 'viability' assumes ideal programming to any conductance but the end result is perturbed by gaussian noise with spread
        # equal to some percentage (the "viability") of the conductance. 
        elif self.method == "viability":

            if self.deterministic:
                self.g_on = torch.ones(self.size) / device_params["r_on"]
                self.g_off =  torch.ones(self.size) / device_params["r_off"]
                self.viability = 0.0
            else:
                self.g_on = 1 / torch.normal(device_params["r_on"], device_params["r_on_stddev"], size=self.size)
                self.g_off = 1 / torch.normal(device_params["r_off"], device_params["r_off_stddev"], size=self.size) 
                self.viability = device_params["viability"]
            
            
        else:
            raise ValueError("device_params['method'] must be \"linear\" or \"viability\"")

        # conductance of the word and bit lines. 
        self.g_wl = torch.Tensor((1 / device_params["r_wl"],))
        self.g_bl = torch.Tensor((1 / device_params["r_bl"],))
                
        # Bias Scheme
        self.bias_voltage = self.V * device_params["bias_scheme"]
        
        # Tile size (1x1 = 1T1R, nxm = passive, etc.)
        self.tile_rows = device_params["tile_rows"]
        self.tile_cols = device_params["tile_cols"]
        assert self.size[0] % self.tile_rows == 0, "tile size does not divide crossbar size in row direction"
        assert self.size[1] % self.tile_cols == 0, "tile size does not divide crossbar size in col direction"
        
        # Resistance of CMOS lines (NOT IMPLEMENTED)
        self.r_cmos_line = device_params["r_cmos_line"]

        # WL & BL resistances
        self.g_s_wl_in = torch.ones(self.tile_rows) * 1
        self.g_s_wl_out = torch.ones(self.tile_rows) * 1e-9
        self.g_s_bl_in = torch.ones(self.tile_rows) * 1e-9
        self.g_s_bl_out = torch.ones(self.tile_rows) * 1

        # WL & BL voltages that are not the signal, assume bl_in, wl_out are tied low and bl_out is tied to 1 V. 
        self.v_bl_in = torch.zeros(self.size[1])
        self.v_bl_out = torch.ones(self.size[1])
        self.v_wl_out = torch.zeros(self.size[0])
        
        # Conductance Matrix; initialize each memristor at the on resstance
        self.W = torch.ones(self.size) * self.g_on

        # Stuck-on & stuck-on device nonideality 
        self.p_stuck_on = device_params["p_stuck_on"]
        self.p_stuck_off = device_params["p_stuck_off"]
        state_dist = torch.distributions.categorical.Categorical(probs=torch.Tensor([self.p_stuck_on, self.p_stuck_off, 1 - self.p_stuck_on - self.p_stuck_off]))
        self.state_mask = state_dist.sample(self.size)

        # Storage for all mapped tensors and their positions. Used to get data off the crossbar after simulation. 
        self.mapped = []
        self.tensors = [] #original data of all mapped weights
        self.saved_tiles = {}
        self.current_history = []

        # NOT TESTED: GPU CAPABILITY
        # self.device = device_params["device"]
        
    # Iterates through the tiles and solves each and then adds their outputs together. 
    def solve(self, voltage):

        output = torch.zeros((voltage.size(1), self.size[1]))
        for i, j in self.programmed_tiles():
            coords = (slice(i*self.tile_rows, (i+1)*self.tile_rows), slice(j*self.tile_cols, (j+1)*self.tile_rows))
            if str(coords) not in self.saved_tiles.keys():
                self.make_M(coords) # Lazy hash
        
        # This part would be super easy to parallelize.
        Es_all = [None] * (self.size[0] // self.tile_rows)
        for i in set(j for j, k in self.programmed_tiles()):
            Es_all[i] = self.make_Es(voltage[i*self.tile_rows:(i+1)*self.tile_rows,:])
        
        for i, j in self.programmed_tiles():
            coords = (slice(i*self.tile_rows, (i+1)*self.tile_rows), slice(j*self.tile_cols, (j+1)*self.tile_rows))
            M = self.saved_tiles[str(coords)]
            V = torch.transpose(-torch.sub(*torch.chunk(torch.matmul(M, Es_all[i]), 2, dim=0)), 0, 1).view(-1, self.tile_rows, self.tile_cols)
            output[:, j*self.tile_cols:(j+1)*self.tile_cols] += torch.sum(V * self.W[coords], axis=1)

        self.current_history.append(output)
        return output

    # extremely lazy implementation of finding which tiles have been programmed. But crossbars aren't very large so whatever.
    def programmed_tiles(self):
        tile_coords = []
        for coords in self.mapped:
            for i in range(self.size[0] // self.tile_rows):
                for j in range(self.size[1] // self.tile_cols):
                    if i * self.tile_rows <= coords[0] + coords[2] and j * self.tile_cols <= coords[1] + 2*coords[3] and (i,j) not in tile_coords:
                        tile_coords.append((i,j))
        return tile_coords

    # Constructs the E matrix in MV = E.
    # used like: Es = torch.cat(tuple(self.make_E(vectors[:, i]).view(-1,1) for i in range(vectors.size(1))), axis=1)
    # saved here for posterity
    def make_E(self, v_wl_in):
        m, n = self.tile_rows, self.tile_cols
        E = torch.cat([torch.cat(((v_wl_in[i]*self.g_s_wl_in[i]).view(1), torch.zeros(n-2), (self.v_wl_out[i]*self.g_s_wl_out[i]).view(1))) for i in range(m)] +
                      [torch.cat(((-self.v_bl_in[i]*self.g_s_bl_in[i]).view(1), torch.zeros(m-2),(-self.v_bl_in[i]*self.g_s_bl_out[i]).view(1))) for i in range(n)]).view(-1, 1)
        return E

    # Vectorized version of make_E
    def make_Es(self,  v_wl_ins):
        width = v_wl_ins.size(1)
        m, n = self.tile_rows, self.tile_cols
        Es = torch.cat([torch.cat(((v_wl_ins[i, :] * self.g_s_wl_in[i]).view(1, width), torch.zeros(n-2, width), (self.v_wl_out[i].view(-1, 1).repeat(1, width) * self.g_s_wl_out[i]).view(1, width))) for i in range(m)] + [torch.cat(((-self.v_bl_in[i].view(-1, 1).repeat(1, width) * self.g_s_bl_in[i]).view(1, width), torch.zeros(m-2, width), (-self.v_bl_in[i].view(-1, 1).repeat(1, width) * self.g_s_bl_out[i]).view(1,  width))) for i in range(n)]).view(-1, width)
        return Es
    
    # Constructs the M matrix in MV = E. 
    def make_M(self, coords):
        
        g = self.W[coords]
        m, n = self.tile_rows, self.tile_cols

        def makec(j):
            c = torch.zeros(m, m*n)
            for i in range(m):
                c[i,n*(i) + j] = g[i,j]
            return c
  
        def maked(j):
            d = torch.zeros(m, m*n)
            
            i = 0
            d[i, j] = -self.g_s_bl_in[j] - self.g_bl - g[i, j]
            d[i, n*(i+1) + j] = self.g_bl
            
            for i in range(1, m):
                d[i, n*(i-1) + j] = self.g_bl
                d[i, n*i + j] = -self.g_bl - g[i,j] - self.g_bl
                d[i,j] = self.g_bl
                   
            i = m - 1
            d[i, n*(i-1) + j] = self.g_bl
            d[i, n*i + j] = -self.g_s_bl_out[j] - g[i,j] - self.g_bl
                
            return d
        
        A = torch.block_diag(*tuple(torch.diag(g[i,:])
                          + torch.diag(torch.cat((self.g_wl, self.g_wl * 2 * torch.ones(n-2), self.g_wl)))
                          + torch.diag(self.g_wl * -1 * torch.ones(n-1), diagonal = 1)
                          + torch.diag(self.g_wl * -1 * torch.ones(n-1), diagonal = -1)
                          + torch.diag(torch.cat((self.g_s_wl_in[i].view(1), torch.zeros(n - 2), self.g_s_wl_out[i].view(1))))
                                   for i in range(m)))
        B = torch.block_diag(*tuple(-torch.diag(g[i,:]) for i in range(m)))
        C = torch.cat([makec(j) for j in range(n)],dim=0)
        D = torch.cat([maked(j) for j in range(0,n)], dim=0)
        M = torch.inverse(torch.cat((torch.cat((A,B),dim=1), torch.cat((C,D),dim=1)), dim=0))

        self.saved_tiles[str(coords)] = M

        return M

    # Handles programming for the crossbar instance. 
    def register_linear(self, matrix, bias=None):

        self.tensors.append(matrix)
        row, col = self.find_space(matrix.size(0), matrix.size(1))
        # Need to add checks for bias size and col size
        
        # Scale matrix                            
        if (self.method == "linear"):                
            mat_scale_factor = torch.max(torch.abs(matrix)) / torch.max(self.g_on) * 2
            scaled_matrix = matrix / mat_scale_factor
            midpoint = self.conductance_states.size(2) // 2
            for i in range(row, row + scaled_matrix.size(0)):
                for j in range(col, col + scaled_matrix.size(1)):
                    shifted = self.conductance_states[i,j] - self.conductance_states[i,j,midpoint]
                    idx = torch.min(torch.abs(shifted - scaled_matrix[i-row,j-col]), dim=0)[1]
                    self.W[i,2*j+1] = self.conductance_states[i,j,idx]
                    self.W[i,2*j] = self.conductance_states[i,j,midpoint-(idx-midpoint)]
                    
        elif (self.method == "viability"):
            mat_scale_factor = torch.max(torch.abs(matrix)) / (torch.max(self.g_on) - torch.min(self.g_off)) * 2
            scaled_matrix = matrix / mat_scale_factor
            for i in range(row, row + scaled_matrix.size(0)):
               for j in range(col, col + scaled_matrix.size(1)):
                   midpoint = (self.g_on[i,j] - self.g_off[i,j]) / 2 + self.g_off[i,j]
                   right_state = midpoint + scaled_matrix[i-row,j-col] / 2
                   left_state = midpoint - scaled_matrix[i-row,j-col] / 2
                   self.W[i,2*j+1] = self.clip(right_state + torch.normal(mean=0,std=right_state*self.viability), i, 2*j+1)
                   self.W[i,2*j] = self.clip(left_state + torch.normal(mean=0,std=left_state*self.viability), i, 2*j)

        if not self.deterministic: self.apply_stuck()
        
        return ticket(row, col, matrix.size(0), matrix.size(1), matrix, mat_scale_factor, self)
    
    def clip(self, tensor, i, j):
        if self.g_off[i,j] < tensor < self.g_on[i,j]:
            return tensor
        elif tensor > self.g_on[i,j]:
            return self.g_on[i,j]
        else:
            return self.g_off[i,j]
    
    def apply_stuck(self):
        self.W[self.state_mask == 0] = self.g_off[self.state_mask==0]
        self.W[self.state_mask == 1] = self.g_on[self.state_mask==1]

    def which_tiles(self, row, col, m_row, m_col):
        return itertools.product(range(row // self.tile_rows, (row + m_row) // self.tile_rows + 1),
                                 range(col // self.tile_cols,(col + m_col) // self.tile_cols + 1),
        )

    def find_space(self, m_row, m_col):

        if m_row > self.size[0] or m_col*2 > self.size[1]:
                raise ValueError("Matrix with size ({}, {}) is too large for crossbar of size ({}, {})".format(m_row, m_col, self.size[0], self.size[1]))
            
        # Format is (*indexes of top left corner, *indexes of bottom right corner + 1 (it's zero indexed))
        if not self.mapped:
            self.mapped.append((0,0,m_row,m_col))
        else:
            if self.mapped[-1][3] + m_col < self.size[1]:
                self.mapped.append((self.mapped[-1][0], self.mapped[-1][3], m_row, m_col))
            else:
                if m_col > (self.size[0] - self.mapped[-1][2]):
                    raise ValueError("Matrix with {} rows does not fit on crossbar with {} free rows".format(m_col, self.size[0] - self.mapped[-1][2]))    
                self.mapped.append((self.mapped[-1][2], 0, m_row, m_col))
                
        #self.mapped.append((self.mapped[-1][0] + self.mapped[-1][2], self.mapped[-1][1] + self.mapped[-1][3], m_row, m_col))
        return self.mapped[-1][0], self.mapped[-1][1] 
    
    def clear(self):
        self.mapped = []
        self.tensors = []
        self.saved_tiles = {}
        self.W = torch.ones(self.size) * self.g_on


class ticket:
    def __init__(self, row, col, m_rows, m_cols, matrix, mat_scale_factor, crossbar):
        self.row, self.col = row, col
        self.m_rows, self.m_cols = m_rows, m_cols
        self.crossbar = crossbar
        self.mat_scale_factor = mat_scale_factor
        self.matrix = matrix
        
        
    def prep_vector(self, vector, v_bits):

        # Scale vector to [0, 2^v_bits]
        vect_min = torch.min(vector)
        vector = vector - vect_min        
        vect_scale_factor = torch.max(vector) / (2**v_bits - 1)
        vector = vector / vect_scale_factor if vect_scale_factor != 0.0 else vector

        # decompose vector by bit
        bit_vector = torch.zeros(vector.size(0),v_bits)
        bin2s = lambda x : "".join(reversed( [str((int(x) >> i) & 1) for i in range(v_bits)] ) )
        for j in range(vector.size(0)):
            bit_vector[j,:] = torch.Tensor([float(i) for i in list(bin2s(vector[j]))])
        bit_vector *= self.crossbar.V

        # Pad bit vector with unselected voltages
        pad_vector = torch.zeros(self.crossbar.size[0], v_bits)        
        pad_vector[self.row:self.row + self.m_rows,:] = bit_vector

        return pad_vector, vect_scale_factor, vect_min
    
    def vmm(self, vector, v_bits=4):
        assert vector.size(1) == 1, "vector wrong shape"

        crossbar = self.crossbar
        
        # Rescale vector and convert to bits.
        pad_vector, vect_scale_factor, vect_min = self.prep_vector(vector, v_bits)
        
        # Solve crossbar circuit
        output = crossbar.solve(pad_vector)
        
        # Get relevant output columns and add binary outputs        
        output = output.view(v_bits, -1, 2)[:,:,0] - output.view(v_bits, -1, 2)[:,:,1]
    
        for i in range(output.size(0)):
            output[i] *= 2**(v_bits - i - 1)
        output = torch.sum(output, axis=0)[self.col:self.col + self.m_cols] 
        
        # Rescale output
        magic_number = 1 # can use to compensate for resistive losses in the lines. Recommend multiplying a bunch of 8x8 integer matrices to find this.
        
        output = (output / crossbar.V * vect_scale_factor * self.mat_scale_factor) / magic_number + torch.sum(vect_min * self.matrix, axis=0)
        
        return output.view(-1, 1)

'''
device_params = {"Vdd": 1.8,
                 "r_wl": 20,
                 "r_bl": 20,
                 "m": 16,
                 "n": 16,
                 "r_on": 1e4,
                 "r_off": 1e5,
                 "dac_resolution": 4,
                 "adc_resolution": 14,
                 "bias_scheme": 1 / 3,
                 "tile_rows": 4,
                 "tile_cols": 4,
                 "r_cmos_line": 600,
                 "r_cmos_transistor": 20,
                 "r_on_stddev": 1e3,
                 "r_off_stddev": 1e4,
                 "p_stuck_on": 0.01,
                 "p_stuck_off": 0.01,
                 "method": "viability",
                 "viability": 0.05,
                 }

device_params = {"Vdd": 1.8,
                 "r_wl": 20,
                 "r_bl": 20,
                 "m": 16,
                 "n": 16,
                 "r_on": 1e4,
                 "r_off": 1e5,
                 "dac_resolution": 4,
                 "adc_resolution": 14,
                 "bias_scheme": 1 / 3,
                 "tile_rows": 4,
                 "tile_cols": 4,
                 "r_cmos_line": 600,
                 "r_cmos_transistor": 20,
                 "p_stuck_on": 0.01,
                 "p_stuck_off": 0.01,
                 "r_on_stddev": 1e3,
                 "r_off_stddev": 1e4,
                 "device_resolution": 4,
                 "method": "linear",
                 }

'''
# Utility function, should be moved
def print_mapping(tensors, mapping, crossbar_size):
    cb = torch.zeros(*crossbar_size)
    for t, m in zip(tensors, mapping):
        cb[m[0]:m[0] + m[2], m[1]:m[1] + m[3]] = t
    rows = torch.nonzero(cb, as_tuple=True)[0].tolist()
    cols = torch.nonzero(cb, as_tuple=True)[1].tolist()
    values = cb[torch.nonzero(cb, as_tuple=True)].tolist()
    for val in zip(rows, cols, values):
        print(val[0], val[1], val[2], sep=", ")