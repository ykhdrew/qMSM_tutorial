# Copyright 2021 Huang Group, Department of Chemistry, University of Wisconsin-Madison

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

    # http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import os
import matplotlib.pyplot as plt
import scipy.linalg

class QuasiMSM(object):
    """
    Wrapper for the functions used for the development of a Quasi-Markov State Model.

    Parameters
    ----------
    input_len: int, default = 1
        The number of flatten TPM in your input. It will be used to reshape the input TPM.
        input_len = (largest lag_time - smallest lag_time) / delta_time

    dimension: int, default = 1
        The dimensions of the TPM. This number is equal to the number of macrostate in your model.
        It will be used to reshape the input TPM.

    delta_time: float, optional, default = 1.0
        The time interval between each flatten TPM. All the calculation share the
        same time unit as delta_time. When using the default vaule 1,
        one should pay attention to the unit.

    There are other initial properties passed to the Class once they are created.
    TPM property can be retrieved through reshaping the original input data.
    lag_time property can be calculated through the input delta_time.
    dTPM_dt property is the derivative of TPM and dTPM_dt_0 is the first element of dTPM_dt.
    tau_k property is the time of memory kernel used to build qMSM for long time dynamics prediction.
    K_matrix property is set to store the memory kernel matrix K in qMSM.
    MIK property is a matrix set to store the Mean Integral of the memory kernel.
    Other properties are private properties which are used to determine the process inside the program.

    """
    def __init__(self, input_len=1, dimension=1, delta_time=1.0):
        self.delta_time = delta_time
        self.input_len = input_len
        self.dimension = dimension
        self.TPM = np.zeros((input_len, dimension, dimension))
        self.lag_time = np.zeros(input_len)
        self.dTPM_dt = np.zeros((input_len-1, dimension, dimension))
        self.dTPM_dt_0 = 0
        self.tau_k = 0
        self.K_matrix = []
        self.MIK = []
        self.__row_norm = False
        self.__col_norm = False
        self.__get_data = False
        self.__pre_set_data = False
        self.__get_dTPM_dt = False
        self.__calculate_K_matrix = False


    def GetData(self, input_data):
        """
        Fetch the initial TPM data and determine the data format.
        :param input_data: input_data is the TPMs(Transition Probability Matrix) at different lag_time;
        The input_data can be either row normalized or column normalized, and in any shape.
        The input_data will be reshaped accordingly for our calculation.
        """
        self.raw_data= input_data
        if not isinstance(self.raw_data, np.ndarray):
            raise TypeError("Loading input data is not np.ndarray type")
        elif not self.raw_data.ndim == 2:
            raise IOError("Dimension of input data is not 2")
        else:
            self.__get_data = True

    def Pre_SetData(self):
        """
        This method preprocesses the input raw data, measures its characteristics and ensures the stability
        of the subsequent calculations. The input data needs to be consistent with the length and dimensionality
        of the input, and if the test passes, the initial data is reconstructed to produce a TPM tensor.
        It will also determine whether the TPM is row normalized or column normalized, and the different normalization
        schemes will affect the subsequent matrix multiplication operations.
        """
        if self.input_len != len(self.raw_data):
            raise IOError("Input length is inconsistent with real data length")
        elif not self.dimension == np.sqrt(len(self.raw_data[0])):
            raise IOError("Input dimension is inconsistent with real data dimension")
        else:
            self.TPM = np.reshape(self.raw_data, (self.input_len, self.dimension, self.dimension))
            for i in range(self.input_len):
                self.lag_time[i] = self.delta_time * (i+1)

        if abs((np.sum(self.TPM[3, 0])) - 1) < 1e-3:
            self.__row_norm = True
            print("The Transtion Probability Matrix is row normalized and row normalization algorithm is used !")
        elif abs(np.sum(self.TPM[3, :, 0]) - 1) < 1e-3:
            self.__col_norm = True
            print("The Transtion Probability Matrix is column normalized and column normalization algorithm is used !")
            for i in range(len(self.TPM)):
                self.TPM[i] = self.TPM[i].T
        else:
            raise IOError("Transition Probablity Matrix is not normalized, cannot do qMSM")

        self.__pre_set_data = True

    def Get_dTPM_dt(self):
        """
        This method is designed to calculate the derivative of TPM;
        Notably, the derivative at time zero point should be computed individually.
        (When computing the zero point, ignore the influence of memory kernel term)

        Returns
        -------
        dTPM_dt_0 is the derivative at zero point, dTPM_dt is a derivative for different lag_time
        """
        for k in range(0, int(self.input_len) - 1):
            self.dTPM_dt[k] = (self.TPM[k + 1] - self.TPM[k]) / self.delta_time
            self.dTPM_dt_0 = np.dot(np.linalg.inv(self.TPM[0]), self.dTPM_dt[0])
        self.__get_dTPM_dt = True
        return self.dTPM_dt_0, self.dTPM_dt

    def Calculate_K_matrix(self, cal_step=10, outasfile=False,outdir="./"):
        """
        This method is designed to calculate the Memory Kernel K matrices.
        This step uses the greedy algorithm to iteratively compute  memory kernel items, and
        requires great attention to the handling of the initial items.

        Parameters
        ----------
        cal_step: Equal to tau_k, corresponding to the point where memory kernel decays to zero.
        (This is a very import parameter for qMSM. You may want to try different values to optimize your result)
        outasfile: Decide whether to output the results of memory kernel calculations to a file.
        outdir: The output directory for your result
        
        Returns
        -------
        K_matrix: Memory kernel calculation result tensor with cal_step entries.
        """
        self.K_matrix = np.zeros((int(cal_step), self.dimension, self.dimension))
        self.tau_k = cal_step
        if not self.__get_data:
            raise ValueError('Please use get_data method to get appropriate TPM data')
        if not self.__pre_set_data:
            raise NotImplementedError('Please use pre_set_data method to reset TPM')
        if not self.__get_dTPM_dt:
            raise NotImplementedError('Please use get_dTPM_dt method to calculate derivative')

        n = 0
        while n < self.tau_k:
            memory_term = np.zeros((self.dimension, self.dimension))
            if n > 0:
                for m in range(0, n):
                    memory_term += np.dot(self.TPM[n - m], self.K_matrix[m])
                self.K_matrix[n] = np.dot(np.linalg.inv(self.TPM[0]),
                                     (((self.dTPM_dt[n] - np.dot(self.TPM[n], self.dTPM_dt_0)) / self.delta_time) - memory_term))
            else:
                self.K_matrix[n] = np.dot(np.linalg.inv(self.TPM[0]), ((self.dTPM_dt[n] - np.dot(self.TPM[n], self.dTPM_dt_0)) / self.delta_time))
            n += 1

        if outasfile:
            kernel = np.zeros((int(cal_step), self.dimension**2))
            for i in range(int(cal_step)):
                kernel[i] = np.reshape(self.K_matrix[i], (1, -1))
            with open("{}calculate_K_matrix_output.txt".format(outdir), 'ab') as file:
                if not os.path.getsize("{}calculate_K_matrix_output.txt".format(outdir)):
                    np.savetxt(file, kernel, delimiter=' ')
                else:
                    raise IOError('Output File already exists, please create another!!')
            del kernel
        self.__calculate_K_matrix = True
        return self.K_matrix

    def KernelPlot(self, K_matrix):
        """
        This method is designed to plot a figure for memory kernel at different lag time.
        The trend of the memory kernel can determine the adequate values of tau_k and cal_step.

        Parameters
        ----------
        K_matrix
        The returned tensor from Calculate_K_matrix method.
        """
        if not isinstance(K_matrix, np.ndarray) or not K_matrix.ndim == 3:
            raise ValueError('K_matrix should be a return value of Calculate_K_matrix method')
        else:
            length = len(K_matrix)
            lag_time = np.zeros(length)
            for i in range(length):
                lag_time[i] = (i+1) * self.delta_time
            plt.figure(1)
            for i in range(self.dimension):
                for j in range(self.dimension):
                    plt.subplot(self.dimension, self.dimension, i*self.dimension+j+1)
                    plt.plot(lag_time, K_matrix[:, i, j], color='black', label='K'+str(i+1)+str(j+1))
                    plt.legend(loc='best', frameon=True)
            plt.rcParams['xtick.direction'] = 'in'
            plt.rcParams['ytick.direction'] = 'in'
            plt.show()
            del length
            del K_matrix

    def MeanIntegralKernel(self, MIK_time=0, figure=False,outdir="./"):
        """
        This method is designed to calculate the integral of the memory kernel over a period of time (MIK_time).
        The trend of the integral value over time can also be plotted using figure parameters.
        It can be used to determine parameters such as tau_k and cal_step.

        Parameters
        ----------
        MIK_time: Time used to compute the integral of the memory kernel.
        Shouldn't be Out of the total range of the K_matrix.
        figure: Decide weather to plot the figure or not.
        outdir: The output directories for your result
        """
        if not self.__calculate_K_matrix:
            raise NotImplementedError('Please use calculate_K_matrix to calculate kernel matrix in advance')
        if self.tau_k < MIK_time:
            raise ValueError('MIK_time is longer than kernel matrix length, not enough data for calculation')
        integral_kernel = np.zeros((int(MIK_time), self.dimension, self.dimension))
        integral_kernel[0] = self.K_matrix[0]
        for i in range(1, MIK_time):
            integral_kernel[i] = integral_kernel[i-1] + self.K_matrix[i]
        integral_kernel = integral_kernel * self.delta_time
        self.__dict__['MIK'] = np.zeros(int(MIK_time))
        for i in range(MIK_time):
            self.MIK[i] = np.sqrt(np.sum(np.power(integral_kernel[i], 2))) / self.dimension

        del integral_kernel

        if figure:
            plt.figure(figsize=(8, 5))
            plt.plot(self.lag_time[:int(MIK_time)], self.MIK, color='black')
            plt.title("Mean Integral of the Memory Kernel(MIK)")
            plt.rcParams['xtick.direction'] = 'in'
            plt.rcParams['ytick.direction'] = 'in'
            plt.ylim(bottom=0)
            plt.tick_params(labelsize='large')
            plt.xlabel("Lag time(ns)",size=16)
            plt.ylabel("MIK(1/Lag time)",size=16)
            plt.savefig("{}MIK.png".format(outdir))
            plt.show()

    def QuasiMSMPrediction(self, kernel_matrix, tau_k=10, end_point=100, outasfile=False, out_RMSE=False, outdir="./"):
        """
        This method is designed to use the qMSM algorithm in combination with the memory kernel
        for accurate prediction of long time scale dynamics.

        Parameters
        ----------
        kernel_matrix: K_matrix calculated from the previous method; containing all information about memory;
        tau_k:  The Point where memory term decays to zeros and prediction starts;
        end_point: The Point where prediction stops;
        outasfile: Decide whether to output the results of prediction to a file;
        out_RMSE: Decide whether to output the results of  predicted TPM for later RMSE calculations;
        outdir: The output directory for your result

        Returns
        -------
        TPM_propagate: The result of prediction using qMSM and memory kernel;
        TPM_gen_RMSE: The result of predicted TPMs and lag_time used for later RMSE calculations.

        """
        if not isinstance(kernel_matrix, np.ndarray):
            raise TypeError("Loading input matrix is not numpy.ndarray type")
        elif not kernel_matrix.ndim == 3:
            raise IOError("Dimension of input data is not correct, should be 3-D tensor")
        elif not len(kernel_matrix) > tau_k+1:
            raise IOError('Input kernel_matrix is inconsistent with tau_k set')
        elif tau_k >= end_point:
            raise ValueError('tau_k is longer than end_point, cannot propogate')
        elif not self.__get_data:
            raise ValueError('Please use get_data method to get appropriate TPM data')
        elif not self.__pre_set_data:
            raise NotImplementedError('Please use pre_set_data method to set TPM')
        else:
            TPM_propagate = np.zeros((int(end_point+1), self.dimension, self.dimension))
            TPM_propagate[:tau_k,:, :] = self.TPM[:tau_k, :, :]
            TPM_grad = np.zeros((int(end_point), self.dimension, self.dimension))
            kernel = kernel_matrix[:tau_k-1, :, :]
            for i in range(tau_k-1):
                TPM_grad[i] = (TPM_propagate[i + 1] - TPM_propagate[i]) / self.delta_time
            TPM_grad0 = np.dot(np.linalg.inv(TPM_propagate[0]), TPM_grad[0])

            n = tau_k-2
            while n < end_point:
                memory_kernel = 0
                for m in range(0, min(tau_k-1, n+1), 1):
                    memory_kernel += np.dot(TPM_propagate[n - m], kernel[m])
                TPM_grad[n] = np.dot(TPM_propagate[n], TPM_grad0) + self.delta_time * memory_kernel
                TPM_propagate[n + 1] = (self.delta_time * TPM_grad[n]) + TPM_propagate[n]
                n += 1

            TPM_gen = np.zeros((int(end_point+1), self.dimension ** 2))
            TPM_gen_RMSE = np.zeros((int(end_point+1), self.dimension**2 + 1))
            for i in range(int(end_point+1)):
                TPM_gen[i] = np.reshape(TPM_propagate[i], (1, -1))
                TPM_gen_RMSE[i] = np.insert(TPM_gen[i], 0, values=(self.delta_time * i), axis=0)
            if outasfile:
                with open("{}qMSM_Propagate_TPM.txt".format(outdir), 'ab') as file1:
                    if not os.path.getsize("{}qMSM_Propagate_TPM.txt".format(outdir)):
                        np.savetxt(file1, TPM_gen, delimiter=' ')
                    else:
                        raise IOError('Output File already exists, please create another!!')
            if out_RMSE:
                with open('qMSM_Propagate_TPM_RMSE.txt', 'ab') as file2:
                    if not os.path.getsize('qMSM_Propagate_TPM_RMSE.txt'):
                        np.savetxt(file2, TPM_gen_RMSE, delimiter=' ')
                    else:
                        raise IOError('Output File already exists, please create another!!')

            del kernel
            del TPM_grad
            del TPM_grad0
            del TPM_gen
        return TPM_propagate, TPM_gen_RMSE  

    def MSMPrediction(self, tau=10, end_point=100, add_iden_mat=False, outasfile=False, out_RMSE=False, outdir="./"):
        """
        This method is designed to propagate long time scale dynamics TPMs using Markov State Model.
        T[n tau] = (T[tau])^{n}
        Parameters
        ----------
        tau: Lag-time for Markov Chain propagations;
        end_point: Point where prediction stops;
        add_iden_mat: Decide weather add an identity matrix at the beginning of the TPM;
        outasfile: Decide whether to output the results of propagation to a file;
        out_RMSE: Decide whether to output the results of  propagated TPM for later RMSE calculations;
        outdir: The output directory for your result

        Returns
        -------
        TPM_propagate: The result of prediction using MSM;
        TPM_gen_RMSE: The result of propagated TPMs and lag_time used for later RMSE calculations.

        """
        if tau >= end_point:
            raise ValueError('tau is longer than end_point, cannot propagate')
        elif not self.__get_data:
            raise ValueError('Please use get_data method to get appropriate TPM data')
        elif not self.__pre_set_data:
            raise NotImplementedError('Please use pre_set_data method to set TPM')
        else:
            TPM_propagate = np.zeros(((int(add_iden_mat) + (end_point // tau)), self.dimension, self.dimension))
            time = np.zeros(int(add_iden_mat) + (end_point // tau))
            if add_iden_mat:
                TPM_propagate[0] = np.identity(self.dimension)
                TPM_propagate[1] = self.TPM[tau]
                time[0] = 0
                time[1] = tau * self.delta_time
            else:
                TPM_propagate[0] = self.TPM[tau]
                time[0] = tau * self.delta_time

            for i in range((int(add_iden_mat) + 1), len(TPM_propagate), 1):
                TPM_propagate[i] = np.dot(TPM_propagate[i - 1], self.TPM[tau])
                time[i] = time[i - 1] + (tau * self.delta_time)
            TPM_gen = np.zeros((len(TPM_propagate), self.dimension**2))
            TPM_gen_RMSE = np.zeros((len(TPM_propagate), self.dimension**2 + 1))
            for i in range(len(TPM_propagate)):
                TPM_gen[i] = np.reshape(TPM_propagate[i], (1, -1))
                TPM_gen_RMSE[i] = np.insert(TPM_gen[i], 0, values=time[i], axis=0)

            if outasfile:
                with open("{}MSM_Propagate_TPM.txt".format(outdir), 'ab') as file1:
                    if not os.path.getsize("{}MSM_Propagate_TPM.txt".format(outdir)):
                        np.savetxt(file1, TPM_gen, delimiter=' ')
                    else:
                        raise IOError('Output File already exists, please create another!!')
            if out_RMSE:
                with open("{}MSM_Propagate_TPM_RMSE.txt".format(outdir), 'ab') as file2:
                    if not os.path.getsize("{}MSM_Propagate_TPM_RMSE.txt".format(outdir)):
                        np.savetxt(file2, TPM_gen_RMSE, delimiter=' ')
                    else:
                        raise IOError('Output File already exists, please create another!!')

            del time
            del TPM_gen
            return TPM_propagate, TPM_gen_RMSE

    def CK_figure(self, qMSM_TPM, MSM_TPM, grid=np.zeros(2), slice_dot=10, add_iden_mat=False, diag=True, outdir="./"):
        """
        This method is designed to plot a figure for Chapman–Kolmogorov (CK) tests of results generated by
        MD(reference), qMSM and MSM. The CK test can be used to visualize the differences and similarities between
        qMSM, MSM prediction results for long time kinetics and real MD results.
        Parameters
        ----------
        qMSM_TPM: TPM containing lag_time information predicted by qMSM;
        MSM_TPM： TPM containing lag_time information predicted by MSM;
        grid: Distribution of image positions for different TPM elements;
        slice_dot: Lag_time to draw the scatter plots;
        add_iden_mat: Decide weather add an identity matrix or not at the beginning of the TPM sequence;
        diag: Decide whether to draw the diagonal elements of the TPM matrix;
        outdir: The output directory for your result

        """
        if not self.__get_data or not self.__pre_set_data:
            raise NotImplementedError('Please use get_data method and pre_set_data method to set TPM')
        if not len(MSM_TPM) == 0:
            if not isinstance(MSM_TPM, np.ndarray) or not MSM_TPM.ndim == 2:
                raise ValueError('MSM_TPM should be a return value of MSMPrediction method')
            if not len(MSM_TPM[0]) == self.dimension**2+1:
                raise ValueError('Time information should be included in the input TPM')
        if not isinstance(qMSM_TPM, np.ndarray) or not qMSM_TPM.ndim == 2:
            raise ValueError('qMSM_TPM should be a return value of QuasiMSMPrediction method')
        if not len(qMSM_TPM[0]) == self.dimension**2+1:
            raise ValueError('Time information should be included in the input TPM')
        if not len(grid) == 2 or not (grid[0]*grid[1] == self.dimension or grid[0]*grid[1] == self.dimension**2):
            raise ValueError('Please set appropriate 2-D grid structure')

        if not len(MSM_TPM) == 0:
            MSM_time = MSM_TPM[:, 0]
            MSM_TPM_plt = np.reshape(MSM_TPM[:, 1:], (len(MSM_TPM), self.dimension, self.dimension))
        qMSM_time = qMSM_TPM[:, 0]
        qMSM_TPM_plt = np.reshape(qMSM_TPM[:, 1:], (len(qMSM_TPM), self.dimension, self.dimension))
        if add_iden_mat:
            qMSM_time = np.insert(qMSM_time, 0, values=0, axis=0)
            qMSM_TPM_plt = np.insert(qMSM_TPM_plt, 0, values=np.identity(self.dimension), axis=0)
        if len(qMSM_TPM) > (self.input_len) :
            print("Length of referred TPM is shorter, use the referred TPM length as cut-off; ")
            num_dot = len(self.TPM) // slice_dot
        if len(qMSM_TPM) <= (self.input_len):
            print("Length of qMSM TPM is shorter, use the qMSM TPM length as cut-off;")
            num_dot = len(qMSM_TPM) // slice_dot

        del qMSM_TPM
        qMSM_time_dot = np.zeros(num_dot)
        qMSM_TPM_dot = np.zeros((num_dot, self.dimension, self.dimension))
        MD_time_dot = np.zeros(num_dot)
        MD_TPM_dot = np.zeros((num_dot, self.dimension, self.dimension))
        for i in range(0, num_dot):
            qMSM_time_dot[i] = qMSM_time[i*slice_dot]
            qMSM_TPM_dot[i] = qMSM_TPM_plt[i*slice_dot]
            MD_time_dot[i] = self.lag_time[i*slice_dot]
            MD_TPM_dot[i] = self.TPM[i*slice_dot]
        if diag:
            plt.figure(figsize=(20, 5))
            for i in range(self.dimension):
                plt.subplot(1, grid[0], i+1)
                if not len(MSM_TPM) == 0:
                    plt.scatter(MSM_time, MSM_TPM_plt[:, i, i], marker='o', color='green', s=10)
                    plt.plot(MSM_time, MSM_TPM_plt[:, i, i],color='green', linewidth=2.5, linestyle='--', label='MSM')
                plt.scatter(qMSM_time_dot, qMSM_TPM_dot[:, i, i], marker='o', color='red', s=10)
                plt.plot(qMSM_time, qMSM_TPM_plt[:, i, i], color='red', linewidth=2.5, label='qMSM')
                plt.scatter(MD_time_dot, MD_TPM_dot[:, i, i], marker='o', color='white', edgecolors='gray', s= 20, label='MD')
                
                plt.title(r"$P_{" + str(i)*2 + "}$", fontsize=15)
                plt.legend(loc='best', frameon=True)
                plt.xlim(left=0)
                plt.ylim(top=1)
                plt.tick_params(labelsize='large')
                plt.xlabel("Lag time(ns)",size=16)       
                plt.ylabel("Residence Probability",size=16)
                plt.tight_layout()
            plt.savefig("{}CK_plot.png".format(outdir))
            plt.show()
        else:
            plt.figure(figsize=(20, 5))
            for i in range(self.dimension):
                for j in range(self.dimension):
                    plt.subplot(grid[0], grid[1], i*self.dimension+j+1)
                    if not len(MSM_TPM) == 0:
                        plt.scatter(MSM_time, MSM_TPM_plt[:, i, j], marker='o', color='green', s=10)
                        plt.plot(MSM_time, MSM_TPM_plt[:, i, j], color='green', linewidth=2.5, linestyle='--', label='MSM')
                    plt.scatter(qMSM_time_dot, qMSM_TPM_dot[:, i, j], marker='o', color='red', s=10)
                    plt.plot(qMSM_time, qMSM_TPM_plt[:, i, j], color='red', linewidth=2.5, label='qMSM')
                    plt.scatter(MD_time_dot, MD_TPM_dot[:, i, j], marker='o', color='white', edgecolors='gray', s=20, label='MD')
                    plt.title(r"$P_{" + str(i)*2 + "}$", fontsize=15)
                    plt.legend(loc='best', frameon=True)
                    plt.xlim(0, num_dot)
                    plt.ylim(top=1)
                    plt.tick_params(labelsize='large')
                    plt.xlabel("Lag time(ns)",size=16)       
                    plt.ylabel("Residence Probability",size=16)
                    plt.tight_layout()
            plt.savefig("{}CK_plot.png".format(outdir))
            plt.show()
        if not len(MSM_TPM_plt) == 0:
            del MSM_time
            del MSM_TPM_plt
            del MSM_TPM
        del qMSM_TPM_plt
        del qMSM_time
        del qMSM_TPM_dot
        del qMSM_time_dot
        del MD_time_dot
        del MD_TPM_dot

    def RMSE(self, kernel, end_point=100, figure=False, outasfile=False,outdir="./"):
        """
        This method is used to compute time-averaged root mean squared error(RMSE) of qMSM and MSM.
        RMSE can evaluate the performance of qMSM and MSM. 
        We can also optimize tau_k(the lag_time of qMSM) through RMSE calculation.
        Parameters
        ----------
        kernel: Memory kernel used to do qMSM, generated from the Calculate_K_matrix method;
        end_point: The end point for calculation for RMSE;
        figure: Decide whether to plot a figure for RMSE or not;
        outasfile: Decide whether to output the data of RMSE or not;
        outdir: The output directory for your result

        Returns
        -------
        the detailed data for RMSE of both qMSM and MSM of different tau_k;
        """
        if not self.__get_data or not self.__pre_set_data:
            raise NotImplementedError('Please use get_data method and pre_set_data method to set TPM')
        if not isinstance(kernel, np.ndarray):
            raise TypeError("Loading input matrix is not numpy.ndarray type")
        if not kernel.ndim == 3:
            raise IOError("Dimension of input data is not correct, should be 3-D tensor")
        if len(kernel)-1 < end_point:
            raise ValueError("The length of memory kernel matrices is shorter than end point")

        qMSM_RMSE = []
        MSM_RMSE = []
        RMSE_time = []
        n = 2
        eign_val, eign_vec = scipy.linalg.eig(self.TPM[10], right=False, left=True)
        eign_vec = eign_vec.real
        tolerance = 1e-8
        mask = abs(eign_val - 1) < tolerance
        # temp = eign_vec[:, mask]
        temp = eign_vec[:, mask].T
        p_k = temp / np.sum(temp)
        p_k = np.reshape(p_k, (4))
        p_k = np.diag(p_k)
        # for i in range(len(p_k)):
        #     eigenval, p_k[i] = scipy.linalg.eig(self.TPM[i], right=False, left=True)
        while n < end_point:
            qMSM_TPM, qMSM_TPM_RMSE = self.QuasiMSMPrediction(kernel_matrix=kernel, tau_k=n, end_point=end_point)
            MSM_TPM, MSM_TPM_RMSE = self. MSMPrediction(tau=n, end_point=end_point)
            MSM_TPM_time = MSM_TPM_RMSE[:, 0]
            qMSM_delt_mat = np.zeros((len(qMSM_TPM), self.dimension, self.dimension))
            for i in range(end_point):
                # qMSM_delt_mat[i] = (qMSM_TPM[i] - self.TPM[i])
                # qMSM_delt_mat[i] = np.dot((qMSM_TPM[i] - self.TPM[i]), p_k)
                qMSM_delt_mat[i] = np.dot(p_k, (qMSM_TPM[i] - self.TPM[i]))
                qMSM_delt_mat[i] = np.power(qMSM_delt_mat[i], 2)
            MSM_delt_mat = np.zeros((len(MSM_TPM_time), self.dimension, self.dimension))
            for i in range(len(MSM_TPM_time)):
                # MSM_delt_mat[i] = (MSM_TPM[i] - self.TPM[int(MSM_TPM_time[i]/self.delta_time - 1)])
                # MSM_delt_mat[i] = np.dot((MSM_TPM[i] - self.TPM[int(MSM_TPM_time[i] / self.delta_time - 1)]), p_k)
                MSM_delt_mat[i] = np.dot(p_k, (MSM_TPM[i] - self.TPM[int(MSM_TPM_time[i] / self.delta_time - 1)]))
                MSM_delt_mat[i] = np.power(MSM_delt_mat[i], 2)

            qMSM_RMSE = np.append(qMSM_RMSE, 100*(np.sqrt((np.sum(qMSM_delt_mat) / self.dimension**2 / (len(qMSM_delt_mat))))))
            MSM_RMSE = np.append(MSM_RMSE, 100*(np.sqrt((np.sum(MSM_delt_mat) / self.dimension**2 / (len(MSM_delt_mat))))))
            RMSE_time.append(n * self.delta_time)
            n += 1
        del qMSM_TPM
        del qMSM_TPM_RMSE
        del MSM_TPM
        del MSM_TPM_RMSE
        if figure:
            plt.figure(figsize=(5, 5))
            plt.plot(RMSE_time, qMSM_RMSE, color='red', label='qMSM', linewidth=2.5)
            plt.plot(RMSE_time, MSM_RMSE, color='black', label='MSM', linewidth=2.5)
            plt.legend(loc='best', frameon=True)
            plt.ylabel('RMSE(%)',size=16)
            plt.xlabel('Lag Time(ns)',size=16)
            plt.xlim(left=0,right=RMSE_time[int((len(RMSE_time) - 1)/2)])
            plt.ylim(bottom=0)
            plt.tick_params(labelsize='large')
            plt.tight_layout()
            plt.savefig("{}RMSE.png".format(outdir))
            plt.show()
        if outasfile:
            qMSM_RMSE_out = np.zeros((len(qMSM_RMSE), 2))
            qMSM_RMSE_out[:, 0] = RMSE_time
            qMSM_RMSE_out[:, 1] = qMSM_RMSE
            MSM_RMSE_out = np.zeros((len(MSM_RMSE), 2))
            MSM_RMSE_out[:, 0] = RMSE_time
            MSM_RMSE_out[:, 1] = MSM_RMSE
            with open("{}qMSM_RMSE.txt".format(outdir), 'ab') as file1:
                if not os.path.getsize("{}qMSM_RMSE.txt".format(outdir)):
                    np.savetxt(file1, qMSM_RMSE_out, delimiter=' ')
                else:
                    raise IOError('Output File already exists, please create another!!')
            with open("{}MSM_RMSE.txt".format(outdir), 'ab') as file2:
                if not os.path.getsize("{}MSM_RMSE.txt".format(outdir)):
                    np.savetxt(file2, MSM_RMSE_out, delimiter=' ')
                else:
                    raise IOError('Output File already exists, please create another!!')
            del qMSM_RMSE_out
            del MSM_RMSE_out

        return qMSM_RMSE, MSM_RMSE

#####################################################

#For calculation done in 
#Cao S. et al. On the advantages of exploiting memory in Markov state models for biomolecular dynamics.
#J. Chem. Phys. 153. 014105. (2020), https://doi.org/10.1063/5.0010787

## qMSM for Alaine Dipeptide
# input_data = np.loadtxt("ala2-pccap-4states-0.1ps-50ps.txt", dtype=float)
# qmsm = QuasiMSM(input_len=500, delta_time=0.1, dimension=4)
# qmsm.GetData(input_data)
# qmsm.Pre_SetData()
# qmsm.Get_dTPM_dt()
# km = qmsm.Calculate_K_matrix(cal_step=300)
# qmsm.MeanIntegralKernel(MIK_time=50, figure=True)
# qmsm_tpm, qmsm_tpm_time = qmsm.QuasiMSMPrediction(kernel_matrix=km, tau_k=15, end_point=200)
# msm_tpm, msm_tpm_time = qmsm.MSMPrediction(tau=15, end_point=200, add_iden_mat=False)
# qmsm.CK_figure(qMSM_TPM=qmsm_tpm_time, MSM_TPM=msm_tpm_time, add_iden_mat=True, diag=False, grid=[4,4], slice_dot=10)
# qmsm.RMSE(kernel=km, end_point=200, figure=True, outasfile=False)

## qMSM for FIP35 WW_Domain
# input_data = np.loadtxt("FIP35_TPM_4states_1ns_2us.txt", dtype=float)
# qmsm = QuasiMSM(input_len=2000, delta_time=1, dimension=4)
# qmsm.GetData(input_data)
# qmsm.Pre_SetData()
# qmsm.Get_dTPM_dt()
# km = qmsm.Calculate_K_matrix(cal_step=400)
# qmsm.MeanIntegralKernel(MIK_time=250, figure=True)
# qmsm_tpm, qmsm_tpm_time = qmsm.QuasiMSMPrediction(kernel_matrix=km, tau_k=25, end_point=400)
# msm_tpm, msm_tpm_time = qmsm.MSMPrediction(tau=25, end_point=400, add_iden_mat=True)
# qmsm.CK_figure(qMSM_TPM=qmsm_tpm_time, MSM_TPM=msm_tpm_time, add_iden_mat=True, diag=True, grid=[4,4], slice_dot=40)
# qmsm.RMSE(kernel=km, end_point=399, figure=True, outasfile=False)

#####################################################

#For calculations done in 
#Zhu, L. et al. Critical role of backbone coordination in the mRNA recognition by RNA induced silencing complex. 
#Commun. Biol. 4. 1345. (2021). https://doi.org/10.1038/s42003-021-02822-7


## qMSM for hAgo2 System
# input_data = np.loadtxt("Lizhe_TPM.sm.macro-transpose.5-800ns.txt", dtype=float)
# qmsm = QuasiMSM(input_len=160, delta_time=1, dimension=4)
# qmsm.GetData(input_data)
# qmsm.Pre_SetData()
# qmsm.Get_dTPM_dt()
# km = qmsm.Calculate_K_matrix(cal_step=100)
# qmsm.MeanIntegralKernel(MIK_time=80, figure=True)
# qmsm.KernelPlot(km)
# qmsm_tpm, qmsm_tpm_time = qmsm.QuasiMSMPrediction(kernel_matrix=km, tau_k=50, end_point=200)
# msm_tpm, msm_tpm_time = qmsm.MSMPrediction(tau=5, end_point=150, add_iden_mat=True)
# qmsm.CK_figure(qMSM_TPM=qmsm_tpm_time, MSM_TPM=msm_tpm_time, add_iden_mat=True, diag=True, grid=[4,4], slice_dot=10)
# qmsm.RMSE(kernel=km, end_point=80, figure=True, outasfile=False)