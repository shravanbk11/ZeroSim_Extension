import os
import numpy as np
import math
import json
import shutil
import threading
import logging
 
from utils import get_ac_score, get_dc_score, ParseParams


class LDOtestbenchEnv:
    def __init__(self, output_dir, simulation_path, CL_low, config_file):
        self.CL_low = CL_low
        self.sample_idx = 0
        self.output_dir = output_dir
        self.simulation_path =simulation_path
        self.lock = threading.Lock()
        os.makedirs(output_dir, exist_ok=True)

        self.files_to_save = [
            "LDO_TB_ACDC_LNR_maxload",
            'LDO_TB_ACDC_LNR_minload',
            'LDO_TB_ACDC_LR_Power_vos',
            'LDO_TB_ACDC_PSRR_dcgain_maxload',
            'LDO_TB_ACDC_GBW_PM_maxload',
            'LDO_TB_ACDC_PSRR_dcgain_minload',
            'LDO_TB_ACDC_GBW_PM_minload',
            'LDO_TB_Tran_meas',
            'LDO_TB_op',
            'LDO_TB_ACDC.log',
            'LDO_TB_Tran.log',
            'LDO_TB_vars.spice',
            'LDO_TB_Vdrop_maxload',
            'LDO_TB_Vdrop_minload'
        ]

        self.parseparams = ParseParams(config_file)

        """Some target specifications for the final design"""
        self.LDR_target = 0.1
        self.LNR_target = 0.01
        self.Power_maxload_target =  9e-5
        self.Power_minload_target =  9e-6
        self.vos_target = 2e-3

        self.PSRR_target = -40
        self.GBW_target = 2e6
        self.phase_margin_target = 60 

        self.v_undershoot_target = 0.1
        self.v_overshoot_target = 0.1

        self.best_reward = -np.inf

    def step(self, parameters):
        with self.lock:
            cur_idx = self.sample_idx
            self.sample_idx += 1
        
        sample_folder = os.path.join(self.output_dir, f'sample_{cur_idx}')
        os.makedirs(sample_folder, exist_ok=True)

        ''' run simulations '''
        self.do_simulation(parameters) 

        # parse the operating points and save them to .json
        cur_op = self.parseparams.parse_dcop(os.path.join(self.simulation_path, "LDO_TB_op"))
        with open(os.path.join(sample_folder, "op.json"), 'w') as file:
            json.dump(cur_op, file, indent=4)

        '''get performance matrics'''
        info, reward = self.get_performace(cur_op)
        # Save performance metrics in both directories
        with open(os.path.join(sample_folder, "performance.json"), 'w') as file:
            json.dump(info, file, indent=4)

        # copy the current output to the sample_folder
        for file in self.files_to_save:
            source_file = os.path.join(self.simulation_path, file)
            dst_file = os.path.join(sample_folder, file)
            shutil.copy(source_file, dst_file)

        # Log the fitness score
        logging.info(f"Sample {cur_idx}: Reward = {reward}")
        if reward > self.best_reward:
            self.best_reward = reward
            with open(os.path.join(self.output_dir, "best_reward.json"), 'w') as file:
                json.dump({"best_reward": self.best_reward}, file, indent=4)

        return reward

    def do_simulation(self, parameters):
        # update netlist
        try:
            # open the netlist of the testbench
            netlist_path = os.path.join(self.simulation_path, 'LDO_TB_vars.spice')
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
                        
                        tokens[i] = f"{name}={new_value}"

                # Re-join the modified line
                new_line = ' '.join(tokens) + '\n'
                new_lines.append(new_line)  

            with open(netlist_path, 'w') as f:
                f.writelines(new_lines)

            os.system(f'cd "{self.simulation_path}" && ngspice -b -o LDO_TB_ACDC.log LDO_TB_ACDC.cir')
            os.system(f'cd "{self.simulation_path}" && ngspice -b -o LDO_TB_Tran.log LDO_TB_Tran.cir')
            print('*** Simulations Done! ***')
        except:
            print('ERROR')
        
    def get_performace(self, op_results):
        ''' LNR at maxload '''
        dc_LNR_maxload = self.parse_dc(os.path.join(self.simulation_path, 'LDO_TB_ACDC_LNR_maxload'))
        LNR_maxload = dc_LNR_maxload[1][1]
        if LNR_maxload < 0:
            LNR_maxload_score = -1
        else :      
            LNR_maxload_score = get_dc_score(LNR_maxload, self.LNR_target)

        ''' LNR at minload '''
        dc_LNR_minload = self.parse_dc(os.path.join(self.simulation_path, 'LDO_TB_ACDC_LNR_minload'))
        LNR_minload = dc_LNR_minload[1][1]     
        if LNR_minload < 0:
            LNR_minload_score = -1
        else :           
            LNR_minload_score = get_dc_score(LNR_minload, self.LNR_target)

        ''' LDR '''
        dc_LR_Power_vos = self.parse_LR_Power_vos(os.path.join(self.simulation_path, 'LDO_TB_ACDC_LR_Power_vos'))
        Ldr, Power_maxload, Power_minload, vos_maxload, vos_minload = dc_LR_Power_vos[1][1], dc_LR_Power_vos[2][1], dc_LR_Power_vos[3][1], abs(dc_LR_Power_vos[4][1]), abs(dc_LR_Power_vos[5][1])

        if Ldr < 0:
            LDR_score = -1
        else :
            LDR_score = get_dc_score(Ldr, self.LDR_target)

        Power_maxload_score = get_dc_score(Power_maxload, self.Power_maxload_target)
        Power_minload_score = get_dc_score(Power_minload, self.Power_minload_target)
        vos_maxload_score = get_dc_score(vos_maxload, self.vos_target)
        vos_minload_score = get_dc_score(vos_minload, self.vos_target)
    
        ''' AC & Loop at maxload'''
        PSRR_dcgain_maxload = self.parse_ac(os.path.join(self.simulation_path, 'LDO_TB_ACDC_PSRR_dcgain_maxload'))
        PSRR_maxload = PSRR_dcgain_maxload[1][1]
        PSRR_maxload_score = get_ac_score(PSRR_maxload, self.PSRR_target)

        dcgain_maxload = PSRR_dcgain_maxload[2][1]
        if dcgain_maxload > 0 :
            try:
                GBW_PM_maxload = self.parse_ac(os.path.join(self.simulation_path, 'LDO_TB_ACDC_GBW_PM_maxload'))
                GBW_maxload = GBW_PM_maxload[1][1]
                GBW_maxload_score = np.min([(GBW_maxload - self.GBW_target) / (GBW_maxload + self.GBW_target), 0])
                phase_margin_maxload = GBW_PM_maxload[2][1]
                phase_margin_maxload_score = np.min([(phase_margin_maxload - self.phase_margin_target) / (phase_margin_maxload + self.phase_margin_target), 0])
            except: 
                if phase_margin_maxload > 180 or phase_margin_maxload < 0:
                    phase_margin_maxload = 0
                else:
                    phase_margin_maxload = phase_margin_maxload
        else :
            GBW_maxload = 0
            GBW_maxload_score = -1
            phase_margin_maxload = 0
            phase_margin_maxload_score = -1          

        ''' AC & Loop at minload'''
        PSRR_dcgain_minload = self.parse_ac(os.path.join(self.simulation_path, 'LDO_TB_ACDC_PSRR_dcgain_minload'))
        PSRR_minload = PSRR_dcgain_minload[1][1]
        PSRR_minload_score = get_ac_score(PSRR_minload, self.PSRR_target)

        dcgain_minload = PSRR_dcgain_minload[2][1]
        if dcgain_minload > 0 :
            try:
                GBW_PM_minload = self.parse_ac(os.path.join(self.simulation_path, 'LDO_TB_ACDC_GBW_PM_minload'))
                GBW_minload = GBW_PM_minload[1][1]
                GBW_minload_score = np.min([(GBW_minload - self.GBW_target) / (GBW_minload + self.GBW_target), 0])
                phase_margin_minload = GBW_PM_minload[2][1]
                phase_margin_minload_score = np.min([(phase_margin_minload - self.phase_margin_target) / (phase_margin_minload + self.phase_margin_target), 0])
            except: 
                if phase_margin_minload > 180 or phase_margin_minload < 0:
                    phase_margin_minload = 0
                else:
                    phase_margin_minload = phase_margin_minload
        else :
            GBW_minload = 0
            GBW_minload_score = -1
            phase_margin_minload = 0
            phase_margin_minload_score = -1   

        ''' Tran test '''       
        tran_result = self.parse_tran(os.path.join(self.simulation_path, 'LDO_TB_Tran_meas'))
        v_undershoot, v_overshoot = tran_result[1][1], abs(tran_result[2][1])      
        v_undershoot_score = get_ac_score(v_undershoot, self.v_undershoot_target)
        v_overshoot_score = get_ac_score(v_overshoot, self.v_overshoot_target)

        """ Decap score """
        CL_area_score = (self.CL_low - op_results['CL']['c']) / (self.CL_low + op_results['CL']['c'])
    
        performance = {
            'LDR': Ldr,
            'LNR_maxload': LNR_maxload,
            'LNR_minload': LNR_minload,
            'Power_maxload': Power_maxload,
            'Power_minload': Power_minload,
            'vos_maxload': vos_maxload,
            'vos_minload': vos_minload,

            'PSRR_maxload': PSRR_maxload,
            'PSRR_minload': PSRR_minload,
            'GBW_maxload': GBW_maxload,
            'GBW_minload': GBW_minload,
            'phase_margin_maxload': phase_margin_maxload,
            'phase_margin_minload': phase_margin_minload,
            'v_undershoot': v_undershoot,
            'v_overshoot': v_overshoot
        }

        """ Total reward """
        reward = LDR_score + LNR_maxload_score + LNR_minload_score + \
            Power_maxload_score + Power_minload_score + vos_maxload_score + vos_minload_score + \
            PSRR_maxload_score + PSRR_minload_score + GBW_maxload_score + GBW_minload_score + \
            phase_margin_maxload_score + phase_margin_minload_score + \
            v_overshoot_score + v_undershoot_score  
        
        if reward >= 0:
            reward = reward + CL_area_score + 10

        return performance, reward

    def parse_dc(self, file_name):
        try:
            LDO_testbench_dc = open(file_name, 'r')
            lines_dc = LDO_testbench_dc.readlines()
            Vin_dc = []                    
            Vout_dc = []
            for line in lines_dc:
                Vdc = line.split(' ')
                Vdc = [i for i in Vdc if i != '']
                Vin_dc.append(float(Vdc[0]))
                Vout_dc.append(float(Vdc[1])) 
    
            dx = Vin_dc[1] - Vin_dc[0]
            dydx = np.gradient(Vout_dc, dx)      
            
            return Vin_dc, Vout_dc
        except:
            print("Simulation errors, no .OP simulation results.")

    def parse_LR_Power_vos(self, file_name):
        try:
            LDO_testbench_LR_Power_vos = open(file_name, 'r')
            lines_dc = LDO_testbench_LR_Power_vos.readlines()
            IL = []                    
            LDR = []
            Power_maxload = []
            Power_minload = []
            vos_maxload = []
            vos_minload = []
            for line in lines_dc:
                Vdc = line.split(' ')
                Vdc = [i for i in Vdc if i != '']
                IL.append(float(Vdc[0]))
                LDR.append(float(Vdc[1])) 
                Power_maxload.append(float(Vdc[3])) 
                Power_minload.append(float(Vdc[5])) 
                vos_maxload.append(float(Vdc[7]))
                vos_minload.append(float(Vdc[9]))    
    
            return IL, LDR, Power_maxload, Power_minload, vos_maxload, vos_minload
        except:
            print("Simulation errors, no .OP simulation results.")

    def parse_ac(self, file_name):
        try:
            LDO_testbench_ac = open(file_name, 'r')  
            lines_ac = LDO_testbench_ac.readlines()   
            freq = []  
            PSRR = []                     
            dcgain = []
            for line in lines_ac:
                Vac = line.split(' ')                
                Vac = [i for i in Vac if i != '']     
                freq.append(float(Vac[0]))            
                PSRR.append(float(Vac[1]))
                dcgain.append(float(Vac[3]))
            return freq, PSRR, dcgain
        except:
            print("Simulation errors, no .AC simulation results.") 

    def parse_tran(self, file_name):
        try:
            LDO_testbench_tran = open(file_name, 'r')
            lines_tran = LDO_testbench_tran.readlines()
            time = []                     
            v_undershoot = []
            v_overshoot = []
            for line in lines_tran:
                line = line.split(' ')
                line = [i for i in line if i != '']
                time.append(float(line[0]))
                v_undershoot.append(float(line[1])) 
                v_overshoot.append(float(line[3])) 
            
            return time, v_undershoot, v_overshoot
        except:
                print("Simulation errors, no .TRAN simulation results.")

    def close(self):
        return None