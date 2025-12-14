import os
import numpy as np
import math
import json
import shutil
import threading
import logging
 
from utils import get_ac_score, get_dc_score, find_op_files, ParseParams
            

class AMPEnv:
    def __init__(self, output_dir, simulation_path, config_file, name='AMP_NMCF'):
        self.simulation_flag = False    # indicate if the simulation is done without Error
        self.sample_idx = 0
        self.wrong_idx = 0
        self.output_dir = output_dir
        # self.wrong_output_dir = output_dir + '_wrong'
        self.simulation_path = simulation_path
        if name[-6:-1] == '_copy':
            self.name = name[:-6]
        elif "_copy" in name:
            self.name = name[:-5]
        else:
            self.name = name
        self.lock = threading.Lock()
        os.makedirs(output_dir, exist_ok=True)
        # os.makedirs(self.wrong_output_dir, exist_ok=True)

        self.files_to_save = [
            self.name+"_ACDC_AC",
            self.name+"_ACDC_DC",
            self.name+"_ACDC_GBW_PM",
            self.name+"_ACDC.log",
            self.name+"_op",
            self.name+"_Tran",
            self.name+"_Tran.log",
            self.name+"_vars.spice",
            "tran.dat"
        ]

        self.parseparams = ParseParams(config_file)

        """Some target specifications for the final design"""
        self.PSRP_target = -90
        self.PSRN_target = -90 
        
        self.TC_target = 1e-6
        self.Power_target = 2e2
        self.vos_target = 4e-5
        
        self.cmrrdc_target = -80 
        self.dcgain_target = 130
        self.GBW_target = 1e6
        self.phase_margin_target = 60 

        self.sr_target = 4e5
        self.settlingTime_target = 5e-6

        self.best_reward = -1e10

    def step(self, parameters):
        # First run the simulations with new parameters
        self.do_simulation(parameters) 
        
        # Then get the performance metrics and reward
        # If the simulation failed, return -inf
        if not self.simulation_flag:
            # with self.lock:
            #     cur_wrong_idx = self.wrong_idx
            #     self.wrong_idx += 1
        
            # sample_folder = os.path.join(self.wrong_output_dir, f'sample_{cur_wrong_idx}')
            # os.makedirs(sample_folder, exist_ok=True)

            # op_files = find_op_files(self.simulation_path)
            # if len(op_files) > 1:
            #     self.files_to_save += [self.name+"_op_contd"]

            # # copy the current output to the sample_folder
            # for file in self.files_to_save:
            #     source_file = os.path.join(self.simulation_path, file)
            #     if os.path.exists(source_file):
            #         dst_file = os.path.join(sample_folder, file)
            #         shutil.copy(source_file, dst_file)
            #         if file != self.name+"_vars.spice":
            #             os.remove(source_file)

            # # Log the fitness score
            # logging.info(f"Wrong sample {cur_wrong_idx}: Reward = {-1e10}")

            return -1e10

        # If the simulation was successful, get the performance metrics and save the results
        with self.lock:
            cur_idx = self.sample_idx
            self.sample_idx += 1

        sample_folder = os.path.join(self.output_dir, f'sample_{cur_idx}')
        os.makedirs(sample_folder, exist_ok=True)
        # logging.info(f"Sample {cur_idx} is correct")

        '''get performance matrics'''
        info, reward = self.get_performace()
        # Save performance metrics in both directories
        with open(os.path.join(sample_folder, "performance.json"), 'w') as file:
            json.dump(info, file, indent=4)

        op_files = find_op_files(self.simulation_path)
        if len(op_files) > 1:
            self.files_to_save += [self.name+"_op_contd"]

        # copy the current output to the sample_folder
        for file in self.files_to_save:
            source_file = os.path.join(self.simulation_path, file)
            dst_file = os.path.join(sample_folder, file)
            shutil.copy(source_file, dst_file)
            # if file != self.name+"_vars.spice":
            #     os.remove(source_file)

        # parse the operating points and save them to .json
        cur_op = self.parseparams.parse_dcop(op_files)
        with open(os.path.join(sample_folder, "op.json"), 'w') as file:
            json.dump(cur_op, file, indent=4)

        # Log the fitness score
        logging.info(f"Correct sample {cur_idx}: Reward = {reward}")
        if reward > self.best_reward:
            self.best_reward = reward
            with open(os.path.join(self.output_dir, "best_sample.json"), 'w') as file:
                json.dump({'sample_idx': cur_idx, 'reward': reward}, file, indent=4)
                
        return reward
    
    def do_simulation(self, parameters):
        # update netlist
        try:
            # open the netlist of the testbench
            netlist_path = os.path.join(self.simulation_path, self.name+'_vars.spice')
            with open(netlist_path, 'r') as f:
                lines = f.readlines()

            parameters = [str(p) for p in parameters]
            param_iter = iter(parameters)

            new_lines = []
            for line in lines:
                # Split the line by spaces
                tokens = line.split()
                # The first token is likely '.param'
                # The rest are parameter assignments of the form NAME=VALUE
                for i in range(1, len(tokens)):
                    # Split by '=' to separate name and value
                    if '=' in tokens[i]:
                        name, old_value = tokens[i].split('=', 1)

                        # get the corresponding value form the paramters
                        try:
                            new_value = next(param_iter)
                        except StopIteration:
                            raise ValueError("Ran out of replacement values in 'parameters' list.")
                        
                        # If this parameter's name starts with "CAPACITOR" (case-insensitive),
                        # append "e-12" to the value
                        if name.upper().startswith("CAPACITOR"):
                            new_value += "e-12"

                        # If this parameter's name starts with "RESISTOR" (case-insensitive),
                        # append "e3" to the value
                        if name.upper().startswith("RESISTOR"):
                            new_value += "e3"     

                        tokens[i] = f"{name}={new_value}"

                # Re-join the modified line
                new_line = ' '.join(tokens) + '\n'
                new_lines.append(new_line)
            
            with open(netlist_path, 'w') as f:
                f.writelines(new_lines)

            os.system(f'cd "{self.simulation_path}" && ngspice -b -o {self.name}_ACDC.log {self.name}_ACDC.cir')
            os.system(f'cd "{self.simulation_path}" && ngspice -b -o {self.name}_Tran.log {self.name}_Tran.cir')
            # logging.info('*** Simulations Done! ***')
            # Check if the simulation was successful
            self.simulation_flag = True
            with open(os.path.join(self.simulation_path, self.name + "_ACDC.log"), 'r') as log_file:
                for line_number, line in enumerate(log_file, start=1):
                    if "error" in line.lower():  # Case insensitive check
                        self.simulation_flag = False
                        # print(f"Error found on line {line_number}: {line.strip()}")
            with open(os.path.join(self.simulation_path, self.name + "_Tran.log"), 'r') as log_file:
                for line_number, line in enumerate(log_file, start=1):
                    if "error" in line.lower():  # Case insensitive check
                        self.simulation_flag = False
                        # print(f"Error found on line {line_number}: {line.strip()}")
        except:
            # logging.info('ERROR')
            self.simulation_flag = False

    def get_performace(self):
        # dc results
        dc_results = self.parse_dc(os.path.join(self.simulation_path, self.name+'_ACDC_DC'))
        tc, Power, vos = dc_results[1][1], dc_results[2][1], abs(dc_results[3][1])
        TC_score = get_dc_score(tc, self.TC_target)
        Power_score = get_dc_score(Power, self.Power_target)
        vos_score = get_dc_score(vos, self.vos_target)

        # ac results
        ac_results = self.parse_ac(os.path.join(self.simulation_path, self.name+'_ACDC_AC'))
        cmrrdc, psrp, psrn = ac_results[1][1], ac_results[2][1], ac_results[3][1]
        cmrrdc_score = get_ac_score(cmrrdc, self.cmrrdc_target)
        PSRP_score = get_ac_score(psrp, self.PSRP_target)
        PSRN_score = get_ac_score(psrn, self.PSRN_target)

        dcgain = ac_results[4][1]
        if dcgain > 0:
            try:
                GBW_PM_results = self.parse_GBW_PM(os.path.join(self.simulation_path, self.name+'_ACDC_GBW_PM'))
                gbw, phase_margin = GBW_PM_results[1][1], GBW_PM_results[2][1]
                dcgain_score = np.min([(dcgain - self.dcgain_target) / (dcgain + self.dcgain_target), 0])
                GBW_score = np.min([(gbw- self.GBW_target) / (gbw + self.GBW_target), 0])
                phase_margin_score = np.min([(phase_margin - self.phase_margin_target) / (phase_margin + self.phase_margin_target), 0])
            except: 
                if phase_margin > 180 or phase_margin < 0:
                    phase_margin = 0
                phase_margin_score = -np.inf
                GBW_score = -np.inf
                dcgain_score = -np.inf

        else:
            gbw = 0
            phase_margin = 0
            dcgain_score = -1
            GBW_score = -1
            phase_margin_score = -1

        # Tran results
        tran_results = self.parse_tran(os.path.join(self.simulation_path, self.name+'_Tran'))
        sr = (tran_results[1][1] + tran_results[2][1]) / 2 
        sr_score = np.min([(sr - self.sr_target) / (sr + self.sr_target), 0])

        # setting_time
        meas, d0 = {}, 0.01
        time_data, vin_data, vout_data = self.extract_tran_data(os.path.join(self.simulation_path, 'tran.dat'))
        if time_data is None:
            return None
        d0_settle, d1_settle, d2_settle, stable, SR_p, settlingTime_p, SR_n, settlingTime_n = self.analyze_amplifier_performance(vin_data, vout_data, time_data, d0)
    
        if math.isnan(d0_settle):
            d0_settle = 10
    
        if math.isnan(d1_settle) or math.isnan(d2_settle) :
            if math.isnan(d1_settle):
                d0_settle += 10
            if math.isnan(d2_settle):
                d0_settle += 10
            d_settle = d0_settle
        else:
            d_settle = max(d0_settle, d1_settle, d2_settle)
    
        if math.isnan(SR_p) or math.isnan(SR_n) :
            Sr = -d_settle
        else:
            Sr = min(SR_p, SR_n)
    
        if math.isnan(settlingTime_p) or math.isnan(settlingTime_n) :
            settlingTime = d_settle
        else:
            settlingTime = max(settlingTime_p, settlingTime_n)
        
        meas['d_settle'] = d_settle
        meas['SR'] = Sr
        meas['settlingTime'] = settlingTime
        settlingTime_score = get_dc_score(settlingTime, self.settlingTime_target)

        # Performance Metrics
        performance = {
            'TC': tc, 
            'Power': Power, 
            'vos': vos, 
            'cmrrdc': cmrrdc, 
            'dcgain': dcgain, 
            'GBW': gbw, 
            'phase_margin (deg)': phase_margin, 
            'PSRP': psrp, 
            'PSRN': psrn, 
            'sr': sr, 
            'setting_time': settlingTime
        }

        # Total reward 
        reward = TC_score + Power_score + vos_score + cmrrdc_score + \
            dcgain_score + GBW_score + phase_margin_score + PSRP_score + \
            PSRN_score + sr_score + settlingTime_score

        return performance, reward

    def parse_dc(self, file_name):
        try:
            AMP_dc = open(file_name, 'r')
            lines_dc = AMP_dc.readlines()
            Temp_dc = []                     
            TC_dc = []
            Power_dc = []
            vos_dc = []
            for line in lines_dc:
                Vdc = line.split(' ')
                Vdc = [i for i in Vdc if i != '']
                Temp_dc.append(float(Vdc[0]))
                TC_dc.append(float(Vdc[1]))
                Power_dc.append(float(Vdc[3])) 
                vos_dc.append(float(Vdc[5]))
            
            return Temp_dc, TC_dc, Power_dc, vos_dc
        except:
            logging.info("Simulation errors, no .OP simulation results.")

    def parse_ac(self, file_name):
        try:
            AMP_ac = open(file_name, 'r')  
            lines_ac = AMP_ac.readlines()     
            freq = []                       
            cmrrdc_ac = []
            PSRP_ac = []
            PSRN_ac = []
            dcgain_ac = []
            for line in lines_ac:
                Vac = line.split(' ')                 
                Vac = [i for i in Vac if i != '']    
                freq.append(float(Vac[0]))           
                cmrrdc_ac.append(float(Vac[1]))
                PSRP_ac.append(float(Vac[3]))
                PSRN_ac.append(float(Vac[5]))
                dcgain_ac.append(float(Vac[7]))
                
            return freq, cmrrdc_ac, PSRP_ac, PSRN_ac, dcgain_ac
        except:
            logging.info("Simulation errors, no .AC simulation results.")

    def parse_GBW_PM(self, file_name):
        try:
            AMP_GBW_PM = open(file_name, 'r') 
            lines_GBW_PM = AMP_GBW_PM.readlines()     
            freq = []                       
            GBW_ac = []
            phase_margin_ac = []
            for line in lines_GBW_PM:
                Vac = line.split(' ')                 
                Vac = [i for i in Vac if i != '']     
                freq.append(float(Vac[0]))            
                GBW_ac.append(float(Vac[1]))
                phase_margin_ac.append(float(Vac[3]))
                
            return freq, GBW_ac, phase_margin_ac
        except:
            logging.info("Simulation errors, no .GBW_PM simulation results.")

    def parse_tran(self, file_name):
        try:
            AMP_tran = open(file_name, 'r')
            lines_tran = AMP_tran.readlines()
            time = []                         
            sr_rise = []
            sr_fall = []
            for line in lines_tran:
                line = line.split(' ')
                line = [i for i in line if i != '']
                time.append(float(line[0]))
                sr_rise.append(float(line[1]))
                sr_fall.append(float(line[3]))

            return time, sr_rise, sr_fall
        except:
            logging.info("Simulation errors, no .TRAN simulation results.")

    def extract_tran_data(self, file_name):
        time_points = []
        raw_data = []
        vin_data = []
        vout_data = []
        time_data = []
        data_section = False
        with open(file_name, 'r')as f:
            lines = f.readlines()
            for line in lines:
                if line.strip():
                    if line.startswith('Values:'):
                        data_section = True
                        continue
                    if data_section:
                        parts = line.strip().split()
                        if len(parts) == 2:
                            time_points.append(int(parts[0]))
                            raw_data.append(float(parts[1]))
                        else:
                            raw_data.append(float(parts[0]))  
    
        if len(time_points) != len(raw_data)/3:
            logging.info('Error in extracting transient data')
            return None, None
        for i in time_points:
            time_data.append(raw_data[3*i])
            vin_data.append(raw_data[3*i+2])
            vout_data.append(raw_data[3*i+1])
    
        return time_data, vin_data, vout_data
    
    def analyze_amplifier_performance(self, vinp, vout, time, d0):
        vinp = np.array(vinp)  
        vout = np.array(vout)
        time = np.array(time)
        def get_step_parameters(vinp, time):
            dv = np.diff(vinp)
            t0 = time[np.where(dv > 0)[0][0]]
            t1 = time[np.where(dv < 0)[0][0]]
            v0 = np.median(vinp[time < t0])
            v1 = np.median(vinp[(time > t0) & (time < t1)])
            return v0, v1, t0, t1
        v0, v1, t0, t1 = get_step_parameters(vinp, time)
    
        pre_step_data = vout[time < t0]
        delta0 = (pre_step_data - v0) / v0
        d0_settle = np.mean(np.abs(delta0))
        stable = not np.any(np.abs(delta0) > d0)
    
        def find_settling_time_index(delta, d0):
            for i in range(len(delta)):
                if np.all(np.abs(delta[i:]) < d0):
                    return i
            return None
    
        def get_slope_and_settling_time(vout, time, v0, v1, start_t, end_t, d0, mode):
            idx = (time >= start_t) & (time <= end_t)
            vout_segment = vout[idx]
            time_segment = time[idx]
    
            target_value = v0 + (v1 - v0) / 2
            idx_target = np.where(vout_segment >= target_value)[0][0] if np.any(vout_segment >= target_value) else None
            if idx_target is None:
                SR = np.nan
            else:
                SR = np.gradient(vout_segment, time_segment)[idx_target]
    
            if mode == 'positive':
                delta = (vout_segment - v1) / v1
            else:
                delta = (vout_segment - v0) / v0

            idx_settle = find_settling_time_index(delta, d0)
            if idx_settle is None:
                settling_time = np.nan
                d_settle = np.mean(np.abs(delta))
            else:
                settling_time = time_segment[idx_settle] - start_t
                d_settle = np.mean(np.abs(delta[idx_settle:]))
            return SR, settling_time, d_settle

        SR_p, settling_time_p, d1_settle = get_slope_and_settling_time(vout, time, v0, v1, t0, t1, d0, 'positive')

        SR_n, settling_time_n, d2_settle = get_slope_and_settling_time(vout, time, v0, v1, t1, np.max(time), d0, 'negative')

        return abs(d0_settle), abs(d1_settle), abs(d2_settle), stable, abs(SR_p), abs(settling_time_p), abs(SR_n), abs(settling_time_n)    
                                        
    def close(self):
        return None

