#   coalestr version 2.71
#   1 Nov 2022
#   With minor changes from version of 11 October 2021

'''
Population class methods
    __init__()
    get_coalescent()
    without_migration()
    with_migration()
    with_migration_alternative()
    get_diversity()
    get_segments()
    good_to_go()
    show_settings()
    restore_settings()
    plot_history()
    plot_observations()

Other functions
    species()
    region()
        
'''

import numpy as np
import matplotlib.pyplot as plt

class Population(object):

    # Set default values of mutation and recombination parameters
        
    mu = 1.1e-8                # single nucleotide substitution rate
    r = 7.4e-4                 # locus scaled recombination rate per kb
    v = 9e-5                   # locus scaled mutation rate per kb
    locus_kb = 27              # length of haplotype locus in kb
    phi_seed = 0.2             # initialise phi for haplotype locus
    phi_bias = 1
    
    # Default values for analysis of shared haplotype segments
    
    unit_length = 13.5         # unit length in kb 
    chromosome_length = 100    # chromosome length in units 
    minimum_length = 2         # minimum segment length in units
    
    # default QC metrics
    
    eco = False
    
    def __init__(self, 
                 history,
                 metapopulation = False ):
        
        # start by creating a forward time series array

        if type(history) is list:

            self.t_his = sum([x[0] for x in history])
            parameters_ft = np.zeros((self.t_his,4)) 
        
            count = 0
            for i,_ in enumerate(history):
                for j in range(history[i][0]):
                    parameters_ft[count + j, 0:4] = history[i][1:5]
                count = count + history[i][0]

        elif type(history) is np.ndarray:
            
            parameters_ft = history
            self.t_his = len(parameters_ft)
            
        # check for value errors
        
        for i in range(self.t_his):

            if parameters_ft[i, 3] > parameters_ft[i, 0]:
                raise ValueError('Nm cannot exceed Nh')                
                
        # convert to a backwards time series array

        self.parameters = np.flip(parameters_ft, axis=0)    
        
        self.parameters_index = {
            'about':'backwards time series array',
            'axis 0': 'bt',
            'axis 1': 'parameters',
             0: 'N - effective number of hosts',
             1: 'Q - quantum of transmission',
             2: 'X - crossing rate of transmission chains',
             3: 'Nm - number of migrant hosts from the metapopulation',
             4: 'Nn - number of migrant hosts from a neighbouring population' }
    
        if metapopulation != False:
            if not isinstance(metapopulation, Population):
                print('WARNING: please define the history of this metapopulation.')
            elif self.t_his > metapopulation.t_his:
                print('WARNING: subpopulation history cannot be longer than metapopulation history.')

        self.metapopulation = metapopulation
        
        '''Apply default settings'''
        self.mu = Population.mu
        self.r = Population.r
        self.v = Population.v
        self.locus_kb = Population.locus_kb
        self.phi_seed = Population.phi_seed
        self.phi_bias = Population.phi_bias
        self.unit_length = Population.unit_length
        self.chromosome_length = Population.chromosome_length
        self.minimum_length = Population.minimum_length
        self.eco = Population.eco
        
        self.settings = {
            'mu': self.mu,
            'r': self.r,
            'v': self.v,
            'locus_kb': self.locus_kb,
            'phi_seed': self.phi_seed,
            'phi_bias': self.phi_bias,
            'unit_length': self.unit_length,
            'chromosome_length': self.chromosome_length,
            'minimum_length': self.minimum_length,
            'eco': self.eco }
 
        '''Set variable used by goodToGo function'''
        self.check_uptodate = ('foo') # modified by .heterozygosity() and used by .good_to_go()
        
##########################################################

    def get_coalescent(self, observe = [0], show = True):
        
        if self.metapopulation == False:
            markov_chain_simulation = self.without_migration
            self.t_sim = self.t_his
        
        elif self.t_his > self.metapopulation.t_his:
            print('WARNING: subpopulation history should not be longer than metapopulation history')
        
        elif not hasattr(self.metapopulation, 'coalescent'):
            print('Please get_coalescent() for the metapopulation and try again.')
        
        else:
            markov_chain_simulation = self.with_migration        
            meta = self.metapopulation
            self.t_sim = meta.t_sim
        
        if observe == "auto":
            observe = range(0, int(self.t_his * 0.7), int(self.t_his * 0.25))

        self.observation_times = [int(x) for x in observe]
        if 0 not in self.observation_times:
            self.observation_times.append(0)
        self.observation_times.sort()

        self.check_uptodate = (self.parameters) # used by good_to_go function
    
        self.coalescent = np.zeros((len(self.observation_times), self.t_sim, 2))
        
        self.coalescent_index = {
            'axis 0':'observation_times[i]',
            'axis 1':'bt',
            'axis 2':'probability distribution of time to coalescence',
             0:'beho: probability of coalescence occurring at time bt',
             1:'wiho: probability of coalescence occurring at time bt' }
        
        self.report = np.zeros((len(self.observation_times), 7))

        self.report_index = {
            'axis 0':'observation_times[i]',
            'axis 1':'summary of coalescence events',
             0:'observation time',            
             1:'beho: percentage of coalescent events captured',
             2:'wiho: percentage of coalescent events captured',
             3:'beho: expectation of coalescence time',
             4:'wiho: expectation of coalescence_time',
             5:'eco setpoint: proportion of coalescences to capture', 
             6:'break time' }
        
        if show != False:
            
            print("Observation time.    Events captured.   Mean coalescence time")
            print("                      beho      wiho        beho     wiho")
   
        for i in range(len(self.observation_times)):
    
            t_obs = self.observation_times[i]
            
            summary, b_lineage, w_lineage = markov_chain_simulation(t_obs)
                
            self.report[i, :] = summary
            
            self.coalescent[i, :, 0] = b_lineage[:, 5]
            self.coalescent[i, :, 1] = w_lineage[:, 5]
                
            if t_obs == 0:
                self.beho_lineage = b_lineage
                self.wiho_lineage = w_lineage
                self.t_brk = int(summary[6])
            
            if show != False:
            
                print("{0:9d}{1:18.1f}{2:10.1f}{3:12.1f}{4:9.1f}".format(
                    t_obs,
                    summary[1],     # beho_events_captured
                    summary[2],     # wiho_events_captured
                    summary[3],     # beho_coalescence_time
                    summary[4] ))   # wiho_coalescence_time   

##########################################################

    def without_migration(self, t_obs):

        '''Run a Markov chain simulation without migration'''
        
        b_lineage = np.zeros((self.t_sim, 7))
        w_lineage = np.zeros((self.t_sim, 7))
    
        self.lineage_index = {
            'axis 0':'bt',
            'axis 1':'state of two lineages',
             0:'probability that lineages are in subpopulation and separated',
             1:'probability that lineages are in subpopulation and cotransmitted',
             2:'probability that lineages are in subpopulation and coalesced',
             3:'blank',
             4:'blank',
             5:'probability of coalescent event at bt',
             6:'summation series for expectation of coalescence time' }
    
        # sample two alleles from two different hosts (beho)
        b_lineage[t_obs, 0:3] = [1, 0, 0]

        # sample two alleles from the same host (wiho)
        w_lineage[t_obs, 0:3] = [0, 1, 0]
            
        b_count = 0
        w_count = 0
        t_brk = self.t_his - 1        

        for bt in range(t_obs + 1, self.t_his):
    
            N = self.parameters[bt,0]
            Q = self.parameters[bt,1]
            X = self.parameters[bt,2]
        
            # create a matrix of transition probabilities
        
            transition_matrix = np.ones((3,3)) 

            transition_matrix[0,:] = [
                1 - (1 / N),
                (Q - 1) / (N * Q),
                1 / (N * Q) ]

            transition_matrix[1,:] = [
                Q * X / (2 * Q - 1),
                (Q - 1) * (2 * Q - Q * X - 1) / (Q * (2 * Q - 1)),
                (2 * Q - Q * X - 1) / (Q * (2 * Q - 1)) ]

            transition_matrix[2,:] = [
                0,
                0,
                1 ]
        
            # perform matrix multiplication
       
            b_lineage[bt, 0:3] = np.matmul(b_lineage[bt - 1, 0:3], transition_matrix[:, :])
            b_lineage[bt, 5] = b_lineage[bt, 2] - b_lineage[bt - 1, 2]
            b_lineage[bt, 6] = b_lineage[bt, 5] * (bt - t_obs)
        
            w_lineage[bt, 0:3] = np.matmul(w_lineage[bt - 1, 0:3], transition_matrix[:, :])
            w_lineage[bt, 5] = w_lineage[bt, 2] - w_lineage[bt - 1, 2]
            w_lineage[bt, 6] = w_lineage[bt, 5] * (bt - t_obs)
                
            if self.eco != False and b_lineage[bt, 2] > self.eco and b_count == 0:
                b_count = 1
                    
            if self.eco != False and w_lineage[bt, 2] > self.eco and w_count == 0:
                w_count = 1
                    
            if b_count == 1 and w_count == 1:
                t_brk = bt
                break
                
        # Percentage of coalescent events captured
        b_events_captured = sum(b_lineage[t_obs:t_brk, 5]) * 100
        w_events_captured = sum(w_lineage[t_obs:t_brk, 5]) * 100   
        
        # Expectation of coalescence time
        b_coalescence_time = sum(b_lineage[t_obs:t_brk, 6])
        w_coalescence_time = sum(w_lineage[t_obs:t_brk, 6])
            
        # Write to output array
            
        summary = np.zeros((7))
            
        summary[0] = t_obs
        summary[1] = b_events_captured         
        summary[2] = w_events_captured
        summary[3] = b_coalescence_time         
        summary[4] = w_coalescence_time
        summary[5] = self.eco
        summary[6] = t_brk
            
        return summary, b_lineage, w_lineage
    
##########################################################

    def with_migration(self, t_obs):

        '''Run a Markov chain simulation with migration from a metapopulation'''
        # Here we assume metapopulation has constant population size ...
        # .. and that all lineages entering the metapopulation do so at bt = 0 
            
        b_lineage = np.zeros((self.t_sim, 10))
        w_lineage = np.zeros((self.t_sim, 10))
            
        self.lineage_index = {
            'axis 0':'bt',
            'axis 2':'state of two lineages',
             0: 'probability that lineages are in subpopulation and separated',
             1: 'probability that lineages are in subpopulation and cotransmitted',
             2: 'probability that lineages are in subpopulation and coalesced',
             3: 'probability that lineages are in metapopulation',
             4: 'blank',
             5: 'probability of coalescent event at bt',
             6: 'summation series for expectation of coalescence time',            
             7: 'probability of coalescent event in subpopulation at bt',
             8: 'probability of lineages entering metapopulation at bt',
             9: 'probability of coalescent event in metapopulation at bt' }
            
        # sample two alleles from two different hosts (beho)
        b_lineage[t_obs, 0:4] = [1, 0, 0, 0]
     
        # sample two alleles from the same host (wiho)
        w_lineage[t_obs, 0:4] = [0, 1, 0, 0]
    
        meta = self.metapopulation
        b_count = 0
        w_count = 0
        t_brk = self.t_his - 1
            
        '''Run a Markov chain simulation'''
            
        for bt in range(t_obs + 1, self.t_his):
 
            N = self.parameters[bt, 0]
            Q = self.parameters[bt, 1]
            X = self.parameters[bt, 2]
            Nm = self.parameters[bt, 3]
                
            M = Nm / N
        
            # create a matrix of transition probabilities
      
            transition_matrix = np.ones((4,4)) 
           
            transition_matrix[0,:] = [
                ((N - 1) * (1 - M) ** 2) / N,
                (Q - 1) * (1 - M) / (N * Q),
                1 / (N * Q),
                (M * (Q - 1) + Q * (N - 1) * (2 * M - M ** 2)) / (N * Q) ]
               
            transition_matrix[1,:] = [
                (Q * X * (1 - M) ** 2) / (2 * Q - 1),
                (Q - 1) * (1 - M) * (2 * Q - Q * X - 1) / (Q * (2 * Q - 1)),
                (2 * Q - Q * X - 1) / (Q * (2 * Q - 1)),
                (Q * M * (2 * Q + Q * X + X - Q * X * M - 3) + M) / (Q * (2 * Q - 1)) ]

            transition_matrix[2,:] = [
                0,
                0,
                1,
                0 ]

            transition_matrix[3,:] = [
                0,
                0,
                0,
                1 ]
        
            # perform matrix multiplication
        
            b_lineage[bt, 0:4] = np.matmul(b_lineage[bt - 1, 0:4], transition_matrix[:, :])
            b_lineage[bt, 7] = b_lineage[bt, 2] - b_lineage[bt - 1, 2] # prob of coalescent event in subpop at bt
            b_lineage[bt, 8] = b_lineage[bt, 3] - b_lineage[bt - 1, 3] # prob of entering metapop at bt
                
            w_lineage[bt, 0:4] = np.matmul(w_lineage[bt - 1, 0:4], transition_matrix[:,:])
            w_lineage[bt, 7] = w_lineage[bt, 2] - w_lineage[bt - 1, 2] # prob of coalescent event in subpop at bt
            w_lineage[bt, 8] = w_lineage[bt, 3] - w_lineage[bt - 1, 3] # prob of entering metapop at bt
            
            if self.eco != False and (w_lineage[bt, 2] + w_lineage[bt, 3]) > self.eco and b_count == 0:
                b_count = 1
                    
            if self.eco != False and (b_lineage[bt, 2] + b_lineage[bt, 3]) > self.eco and w_count == 0:
                w_count = 1
                    
            if b_count == 1 and w_count == 1:
                t_brk = bt
                break
                
        b_meta = b_lineage[t_brk, 3]
        w_meta = w_lineage[t_brk, 3]
            
        # prob of coalescent event in meta at bt (approximate by assuming all migrations at bt = 0) 
        
        b_lineage[t_obs:self.t_sim, 9] = b_meta * meta.coalescent[0, 0: self.t_sim - t_obs, 0]
        w_lineage[t_obs:self.t_sim, 9] = w_meta * meta.coalescent[0, 0: self.t_sim - t_obs, 0]        
        
        # overall probability of coalescent event at bt
        
        b_lineage[t_obs:self.t_sim, 5] = b_lineage[t_obs:self.t_sim, 7] + b_lineage[t_obs:self.t_sim, 9]
        w_lineage[t_obs:self.t_sim, 5] = w_lineage[t_obs:self.t_sim, 7] + w_lineage[t_obs:self.t_sim, 9]
                
        for bt in range(t_obs + 1, self.t_sim):
            b_lineage[bt, 6] = b_lineage[bt, 5] * (bt - t_obs)
            w_lineage[bt, 6] = w_lineage[bt, 5] * (bt - t_obs)
                
        # Percentage of coalescent events captured
           
        b_events_captured = sum(b_lineage[t_obs:self.t_sim, 5]) * 100
        w_events_captured = sum(w_lineage[t_obs:self.t_sim, 5]) * 100   
        
        # Expectation of coalescence time
            
        b_coalescence_time = sum(b_lineage[t_obs:self.t_sim, 6])
        w_coalescence_time = sum(w_lineage[t_obs:self.t_sim, 6])
    
        # Write to output array
        
        summary = np.zeros((7))        
            
        summary[0] = t_obs
        summary[1] = b_events_captured         
        summary[2] = w_events_captured
        summary[3] = b_coalescence_time         
        summary[4] = w_coalescence_time
        summary[5] = self.eco
        summary[6] = t_brk
            
        return summary, b_lineage, w_lineage
    
##########################################################

    def with_migration_alternative(self, t_obs):

        '''Run a Markov chain simulation with migration from a metapopulation'''
        # Here we assume metapopulation has constant population size ...
        # .. and that all lineages entering the metapopulation do so at bt = 0 
            
        b_lineage = np.zeros((self.t_sim, 10))
        w_lineage = np.zeros((self.t_sim, 10))
            
        self.lineage_index = {
            'axis 0':'bt',
            'axis 2':'state of two lineages',
             0: 'probability that lineages are in subpopulation and separated',
             1: 'probability that lineages are in subpopulation and cotransmitted',
             2: 'probability that lineages are in subpopulation and coalesced',
             3: 'probability that lineages are in metapopulation',
             4: 'blank',
             5: 'probability of coalescent event at bt',
             6: 'summation series for expectation of coalescence time',            
             7: 'probability of coalescent event in subpopulation at bt',
             8: 'probability of lineages entering metapopulation at bt',
             9: 'probability of coalescent event in metapopulation at bt' }
            
        # sample two alleles from two different hosts (beho)
        b_lineage[t_obs, 0:4] = [1, 0, 0, 0]
     
        # sample two alleles from the same host (wiho)
        w_lineage[t_obs, 0:4] = [0, 1, 0, 0]
    
        meta = self.metapopulation
        b_count = 0
        w_count = 0
        t_brk = self.t_his - 1
            
        '''Run a Markov chain simulation'''
            
        for bt in range(t_obs + 1, self.t_his):
 
            N = self.parameters[bt, 0]
            Q = self.parameters[bt, 1]
            X = self.parameters[bt, 2]
            Nm = self.parameters[bt, 3]
                
            M = Nm / N
        
            # create a matrix of transition probabilities
      
            transition_matrix = np.ones((4,4)) 
           
            transition_matrix[0,:] = [
                ((N - 1) * (1 - M) ** 2) / N,
                (Q - 1) * (1 - M) / (N * Q),
                1 / (N * Q),
                (M * (Q - 1) + Q * (N - 1) * (2 * M - M ** 2)) / (N * Q) ]
               
            transition_matrix[1,:] = [
                (Q * X * (1 - M) ** 2) / (2 * Q - 1),
                (Q - 1) * (1 - M) * (2 * Q - Q * X - 1) / (Q * (2 * Q - 1)),
                (2 * Q - Q * X - 1) / (Q * (2 * Q - 1)),
                (Q * M * (2 * Q + Q * X + X - Q * X * M - 3) + M) / (Q * (2 * Q - 1)) ]

            transition_matrix[2,:] = [
                0,
                0,
                1,
                0 ]

            transition_matrix[3,:] = [
                0,
                0,
                0,
                1 ]
        
            # perform matrix multiplication
        
            b_lineage[bt, 0:4] = np.matmul(b_lineage[bt - 1, 0:4], transition_matrix[:, :])
            b_lineage[bt, 7] = b_lineage[bt, 2] - b_lineage[bt - 1, 2] # prob of coalescent event in subpop at bt
            b_lineage[bt, 8] = b_lineage[bt, 3] - b_lineage[bt - 1, 3] # prob of entering metapop at bt
                
            w_lineage[bt, 0:4] = np.matmul(w_lineage[bt - 1, 0:4], transition_matrix[:,:])
            w_lineage[bt, 7] = w_lineage[bt, 2] - w_lineage[bt - 1, 2] # prob of coalescent event in subpop at bt
            w_lineage[bt, 8] = w_lineage[bt, 3] - w_lineage[bt - 1, 3] # prob of entering metapop at bt
            
            if self.eco != False and (w_lineage[bt, 2] + w_lineage[bt, 3]) > self.eco and b_count == 0:
                b_count = 1
                    
            if self.eco != False and (b_lineage[bt, 2] + b_lineage[bt, 3]) > self.eco and w_count == 0:
                w_count = 1
                    
            if b_count == 1 and w_count == 1:
                t_brk = bt
                break

######### THIS IS THE SECTION THAT IS DIFFERENT
        # We iterate over lineage[i, 8] instead of using the cumulative sum at lineage[i, 3]

        for i in range(t_obs, meta.t_brk):
            
            b_meta = b_lineage[i, 8]
            w_meta = w_lineage[i, 8]
            
            b_lineage[i:self.t_sim, 9] += b_meta * meta.coalescent[0, 0: self.t_sim - i, 0]
            w_lineage[i:self.t_sim, 9] += w_meta * meta.coalescent[0, 0: self.t_sim - i, 0] 
                
########
        
        # overall probability of coalescent event at bt
        
        b_lineage[t_obs:self.t_sim, 5] = b_lineage[t_obs:self.t_sim, 7] + b_lineage[t_obs:self.t_sim, 9]
        w_lineage[t_obs:self.t_sim, 5] = w_lineage[t_obs:self.t_sim, 7] + w_lineage[t_obs:self.t_sim, 9]
                
        for bt in range(t_obs + 1, self.t_sim):
            b_lineage[bt, 6] = b_lineage[bt, 5] * (bt - t_obs)
            w_lineage[bt, 6] = w_lineage[bt, 5] * (bt - t_obs)
                
        # Percentage of coalescent events captured
           
        b_events_captured = sum(b_lineage[t_obs:self.t_sim, 5]) * 100
        w_events_captured = sum(w_lineage[t_obs:self.t_sim, 5]) * 100   
        
        # Expectation of coalescence time
            
        b_coalescence_time = sum(b_lineage[t_obs:self.t_sim, 6])
        w_coalescence_time = sum(w_lineage[t_obs:self.t_sim, 6])
    
        # Write to output array
        
        summary = np.zeros((7))        
            
        summary[0] = t_obs
        summary[1] = b_events_captured         
        summary[2] = w_events_captured
        summary[3] = b_coalescence_time         
        summary[4] = w_coalescence_time
        summary[5] = self.eco
        summary[6] = t_brk
            
        return summary, b_lineage, w_lineage     

##########################################################

    def get_diversity(self, show = True):
        
        if self.metapopulation !=False and not hasattr(self.metapopulation, 'diversity'):
            print('Please get_diversity() for the metapopulation and try again.')
        
        self.check_uptodate = (self.mu, self.r, self.v, self.parameters) # used by good_to_go function
        
        # Prepare an array for heterozygosity
        
        self.diversity = np.zeros((len(self.observation_times), 7))
    
        self.diversity_index = {
            'axis 0': 'observation time[i]',
            'axis 1': 'SNP heterozygosity and haplotype homozygosity',
             0: 't_obs',
             1: 'beho_snp_het',
             2: 'wiho_snp_het',
             3: 'beho_hap_hom',
             4: 'wiho_hap_hom',
             5: 'fws',
             6: 'fst' }
    
        # Prepare a printout of the results
        
        if show == True or show == "all":
            
            print("Observation time.    SNP heterozygosity.   Haplotype homozygosity at {0:.1f} kb locus".format(self.locus_kb))
            print("                      beho       wiho           beho       wiho")
            
        if show == "snp":
            
            print("Observation time.    SNP heterozygosity.")
            print("                      beho       wiho")            

        # Iterate through observation times starting with the earliest timepoint
        
        phi = self.phi_seed
        
        for i in reversed(range(len(self.observation_times))):
        
            t_obs = self.observation_times[i]

            # Create a backwards time series array to calculate expected heterozygosity of
            # .. alleles sampled from different hosts (beho: between-host variation)
            # .. alleles sampled from the same host (wiho: within-host variation)
            
            summate = np.zeros((self.t_sim, 6))
    
            self.summate_index = {
                'axis 0': 'bt',
                'axis 1': 'values',
                 0: 'expected SNP heterozygosity for this coalescence time',
                 1: 'b: summation series for expectation of SNP heterozygosity',
                 2: 'w: summation series for expectation of SNP heterozygosity',
                 3: 'expected haplotype homozygosity for this coalescence time',
                 4: 'b: summation series for expectation of haplotype homozygosity',
                 5: 'w: summation series for expectation of haplotype homozygosity' }

            haplotype_decay_product = 1
            
            for bt in range(t_obs + 1, self.t_sim):
                
                # SNP heterozygosity
                
                summate[bt, 0] = 1 - (1 - self.mu) ** (2 * (bt - t_obs))
                
                summate[bt, 1] = self.coalescent[i, bt, 0] * summate[bt, 0]
                summate[bt, 2] = self.coalescent[i, bt, 1] * summate[bt, 0]
                
                # haplotype homozygosity
                    
                haplotype_decay = 1 - (phi * self.r * self.locus_kb) - (self.v * self.locus_kb)
        
                if haplotype_decay < 0:
                    haplotype_decay = 0

                # summate[bt, 3] = haplotype_decay ** (2 * (bt - t_obs))
                # summate[bt, 4] = self.coalescent[i, bt, 0] * summate[bt, 3]
                # summate[bt, 5] = self.coalescent[i, bt, 1] * summate[bt, 3]
                
                summate[bt, 3] = haplotype_decay ** 2
                
                haplotype_decay_product = haplotype_decay_product * (haplotype_decay ** 2)
                
                summate[bt, 4] = self.coalescent[i, bt, 0] * haplotype_decay_product
                summate[bt, 5] = self.coalescent[i, bt, 1] * haplotype_decay_product
                    
                #summate[bt, 4] = self.coalescent[i, bt, 0] * np.prod(summate[t_obs + 1: bt + 1, 3])
                #summate[bt, 5] = self.coalescent[i, bt, 1] * np.prod(summate[t_obs + 1: bt + 1, 3])
                
                # check that the above two methods give roughly the same results

            beho_snp_het = sum(summate[t_obs+1:,1])
            wiho_snp_het = sum(summate[t_obs+1:,2])
            beho_hap_hom = sum(summate[t_obs+1:,4])
            wiho_hap_hom = sum(summate[t_obs+1:,5])
            fws = 1 - (wiho_snp_het / beho_snp_het)
            
            self.diversity[i, 0] = t_obs
            self.diversity[i, 1] = beho_snp_het
            self.diversity[i, 2] = wiho_snp_het
            self.diversity[i, 3] = beho_hap_hom
            self.diversity[i, 4] = wiho_hap_hom
            self.diversity[i, 5] = fws
            
            if self.metapopulation != False:
                meta_snp_het = self.metapopulation.diversity[0, 1]
                fst = 1 - (beho_snp_het / meta_snp_het)
                self.diversity[i, 6] = fst
            
            phi = 1 - wiho_hap_hom
            
            if show == True or show == "all":
            
                print("{0:9d}{1:19.2e}{2:11.2e}{3:15.2e}{4:11.2e}".format(
                    t_obs,
                    beho_snp_het,
                    wiho_snp_het,
                    beho_hap_hom,
                    wiho_hap_hom))
                
            if show == "snp":
            
                print("{0:9d}{1:19.2e}{2:11.2e}".format(
                    t_obs,
                    beho_snp_het,
                    wiho_snp_het ))                
                
##########################################################
                
    def get_segments(self, show = True):
        
        self.shared_segments = np.zeros((len(self.observation_times), 7))
        
        self.shared_segments_index = {
            'axis 0':'observation_times[i]',
            'axis 1':'shared segment analysis',
             0:'t_obs',
             1:'b_omega',
             2:'b_sigma',
             3:'b_segment_length',
             4:'w_omega',
             5:'w_sigma',
             6:'w_segment_length'}
        
        if show != False:

            print("Shared haplotype segments of {0:d} units or more, in a chromosome of length {1:d} units".format(
                self.minimum_length,
                self.chromosome_length))
            
            print("where one unit is {0:.2f} kilobases".format(
                self.unit_length))
            
            print("\nObservation time         Between host                   Within host")
            print("                       omega  sigma   mean kb         omega  sigma   mean kb")

        phi_list = self.phi_seed * np.ones(self.chromosome_length + 1)
        
        for j in reversed(range(len(self.observation_times))):
        
            t_obs = self.observation_times[j]

            segments = np.zeros((self.chromosome_length + 1, 16))
        
            self.segments_index = {
                'axis 0':'i in range(chromosome_length + 1)',
                'axis 1':'shared segment analysis',
                 0:'t_obs',
                 1:'haplotype_units',
                 2:'haplotype_kb',
                 3:'beho_haphom',
                 4:'wiho_haphom',
                 5:'beho_gi',
                 6:'wiho_gi',
                 7:'nzi',
                 8:'i * beho_gi * nzi',
                 9:'i * wiho_gi * nzi',
                 10:'chromosome_length * beho_gi * nzi',
                 11:'chromosome_length * wiho_gi * nzi',
                 12:'beho_gi * nzi',
                 13:'wiho_gi * nzi',
                 14:'beho_prob_L',
                 15:'wiho_prob_L' }
            
            for i in range(self.chromosome_length):
                
                haplotype_kb = i * self.unit_length
                
                haplotype_decay_product = 1
                
                shs_sums = np.zeros((self.t_sim, 2)) 
                # [bt, 0] beho: summation series for expectation of haplotype homozygosity
                # [bt, 1] wiho: summation series for expectation of haplotype homozygosity
                
                for bt in range(t_obs + 1, self.t_sim):
    
                    phi = phi_list[i]
                    
                    haplotype_decay = 1 - (phi * self.r * haplotype_kb) - (self.v * haplotype_kb)
        
                    if haplotype_decay < 0:
                        haplotype_decay = 0
                    
                    haplotype_decay_product *= haplotype_decay ** 2
        
                    shs_sums[bt, 0] = self.coalescent[j, bt, 0] * haplotype_decay_product
                    shs_sums[bt, 1] = self.coalescent[j, bt, 1] * haplotype_decay_product        

                nzi = self.chromosome_length - i
                
                if i != self.chromosome_length:
                    
                    beho_hap_hom = sum(shs_sums[t_obs + 1:, 0])
                    wiho_hap_hom = sum(shs_sums[t_obs + 1:, 1])
                    
                    segments[i, 3] = beho_hap_hom
                    segments[i, 4] = wiho_hap_hom
                
                if i == 0:

                    beho_gi = 1 - beho_hap_hom
                    wiho_gi = 1 - wiho_hap_hom
        
                else:

                    beho_gi = segments[i - 1, 3] - beho_hap_hom
                    wiho_gi = segments[i - 1, 4] - wiho_hap_hom
        
                segments[i, 0] = t_obs 
                segments[i, 1] = i 
                segments[i, 2] = haplotype_kb    
                segments[i, 5] = beho_gi
                segments[i, 6] = wiho_gi
                segments[i, 7] = nzi
                segments[i, 8] = i * beho_gi * nzi 
                segments[i, 9] = i * wiho_gi * nzi
                segments[i, 10] = self.chromosome_length * beho_gi * nzi
                segments[i, 11] = self.chromosome_length * wiho_gi * nzi
                segments[i, 12] = beho_gi * nzi
                segments[i, 13] = wiho_gi * nzi

                phi_list[i] = 1 - wiho_hap_hom
                
            i = self.chromosome_length
            
            nzi = 1
            beho_gi = segments[i - 1, 3]
            wiho_gi = segments[i - 1, 4]
            
            segments[i, 0] = t_obs 
            segments[i, 1] = i
            segments[i, 2] = haplotype_kb    
            segments[i, 5] = beho_gi
            segments[i, 6] = wiho_gi
            segments[i, 7] = nzi
            segments[i, 8] = i * beho_gi * nzi 
            segments[i, 9] = i * wiho_gi * nzi
            segments[i, 10] = self.chromosome_length * beho_gi * nzi
            segments[i, 11] = self.chromosome_length * wiho_gi * nzi
            segments[i, 12] = beho_gi * nzi
            segments[i, 13] = wiho_gi * nzi
            
            phi_list[i] = 1 - wiho_hap_hom
        
            beho_omega = np.sum(segments[self.minimum_length:,10]) / np.sum(segments[:,8])
            wiho_omega = np.sum(segments[self.minimum_length:,11]) / np.sum(segments[:,9]) 
    
            beho_sigma = np.sum(segments[self.minimum_length:,8]) / np.sum(segments[:,8])
            wiho_sigma = np.sum(segments[self.minimum_length:,9]) / np.sum(segments[:,9])

            beho_segment_length = self.chromosome_length * self.unit_length * beho_sigma / beho_omega
            wiho_segment_length = self.chromosome_length * self.unit_length * wiho_sigma / wiho_omega

            segments[:,14] = segments[:,12] / np.sum(segments[:,12]) 
            # Probability distribution of beho_L
            
            segments[:,15] = segments[:,13] / np.sum(segments[:,13])
            # Probability distribution of wiho_L
            
            if t_obs == 0:
                
                self.segments = segments
            
            self.shared_segments[j,0] = t_obs
            self.shared_segments[j,1] = beho_omega
            self.shared_segments[j,2] = beho_sigma
            self.shared_segments[j,3] = beho_segment_length
            self.shared_segments[j,4] = wiho_omega
            self.shared_segments[j,5] = wiho_sigma
            self.shared_segments[j,6] = wiho_segment_length

            if show != False:
                
                print("{0:9d}{1:19.2f}{2:7.2f}{3:10.2f}{4:14.2f}{5:7.2f}{6:10.2f}".format(
                    t_obs,
                    beho_omega,
                    beho_sigma,
                    beho_segment_length,
                    wiho_omega,
                    wiho_sigma,
                    wiho_segment_length))
                
##########################################################                
                
    def good_to_go(self):
        
        if self.check_uptodate == (self.mu, self.r, self.v, self.parameters):
            
            print('Simulation has been run on latest set of transmission & mutation parameters.')
            
        else:
            
            print('Please run simulation on latest set of transmission & mutation parameters.')
            
##########################################################  

    def show_settings(self):
        
        print("mu {0:.3e}, \nr {1:.3e}, \nv {2:.3e}, \nlocus_kb {3:.2f}".format(
            self.mu,
            self.r,
            self.v,
            self.locus_kb))
        
        print("unit_length {0:.2f}, \nchromosome_length {1:d}, \nminimum_length {2:d}".format(
            self.unit_length,
            self.chromosome_length,
            self.minimum_length))
        
##########################################################   
        
    def restore_settings(self):
        
        self.mu = Population.mu
        self.r = Population.r
        self.v = Population.v
        self.locus_kb = Population.locus_kb
        self.phi_seed = Population.phi_seed
        self.unit_length = Population.unit_length
        self.chromosome_length = Population.chromosome_length
        self.minimum_length = Population.minimum_length
        
##########################################################        
        
    def plot_history(self, metrics = ("N", "Q", "X")):
        
        '''Plot transmission parameters as a forward time series'''
                     
        if type(metrics) is not tuple:
            metrics = (metrics, )             
    
        time_axis = [-x for x in range(self.t_his)]
        
        width = 10
        depth = 3 * len(metrics)

        fig, ax = plt.subplots(
            nrows = len(metrics),
            ncols = 1,
            figsize=(width, depth),
            sharex = True )

        if len(metrics) == 1:
            ax = [ax]
            
        for i in range(len(metrics)):
            
            if metrics[i] == "N" or metrics[i] == "Nh":
            
                ax[i].plot(time_axis, self.parameters[:, 0], marker='', color='blue', linewidth=2, label="Nh")
                ax[i].legend(title="Nh", frameon=True, fontsize = 12) 
                ax[i].set_ylabel("Nh", fontsize=12)
                ax[i].grid(b=True, which='both', color='0.65', linestyle='-')
                     
            elif metrics[i] == "Q":
            
                ax[i].plot(time_axis, self.parameters[:, 1], marker='', color='blue', linewidth=2, label="Q")
                ax[i].legend(title="Q", frameon=True, fontsize = 12) 
                ax[i].set_ylabel("Q", fontsize=12)
                ax[i].grid(b=True, which='both', color='0.65', linestyle='-')         
            
            elif metrics[i] == "X" or metrics[i] == "chi":
            
                ax[i].plot(time_axis, self.parameters[:, 2], marker='', color='blue', linewidth=2, label="\u03C7")
                ax[i].legend(title="\u03C7", frameon=True, fontsize = 12) 
                ax[i].set_ylabel("\u03C7", fontsize=12)
                ax[i].grid(b=True, which='both', color='0.65', linestyle='-')
                     
#            elif metrics[i] == "M":
            
#                ax[i].plot(time_axis, paramtr_ft[:, 3], marker='', color='blue', linewidth=2, label="M")
#                ax[i].legend(title="M", frameon=True, fontsize = 12) 
#                ax[i].set_ylabel("M", fontsize=12)
#                ax[i].grid(b=True, which='both', color='0.65', linestyle='-')                      
                
            else:
                
                print('Sorry,' + ' "' + metrics[i] + '" ' + 'is not recognised as a metric for this plot.') 
            
        ax[i].set_xlabel("Generations", fontsize=12)
        #ax[i].set_xlim(-go_back_in_time, 0)    
    
        plt.show()

##########################################################        
        
    def plot_observations(self, metrics = ("snp_het", "hap_hom", "fws", "fst", "N", "X")):

        '''Plot simulation results and transmission parameters over the observation times'''
    
        if type(metrics) is not tuple:
            metrics = (metrics, )
        
        go_back_in_time = int(np.amax(self.diversity[:,0]))  # how many generations we go back in time
        time_axis = [-x for x in range(go_back_in_time)]
        
        obs_times = - self.diversity[:,0]
    
        width = 10
        depth = 3 * len(metrics)
        
        fig, ax = plt.subplots(
            nrows = len(metrics),
            ncols = 1,
            figsize=(width, depth),
            sharex = True)

        if len(metrics) == 1:
            ax = [ax]
            
        for i in range(len(metrics)):
            
            if metrics[i] == "snp_het" or metrics[i] == "snphet" or metrics[i] == "pi":
            
                ax[i].plot(obs_times, self.diversity[:,1], marker='', color='blue', linewidth=2, label="between-host")
                ax[i].plot(obs_times, self.diversity[:,2], marker='', color='red', linewidth=2, label="within-host")
                ax[i].legend(title="Sample", frameon=False, fontsize = 12) 
                ax[i].set_ylabel("Nucleotide diversity", fontsize=12)
                ax[i].grid(b=True, which='both', color='0.65', linestyle='-')
            
            elif metrics[i] == "hap_hom" or metrics[i] == "haphom":
            
                ax[i].plot(obs_times, self.diversity[:,3], marker='', color='blue', linewidth=2, label="between-host")
                ax[i].plot(obs_times, self.diversity[:,4], marker='', color='red', linewidth=2, label="within-host")
                ax[i].legend(title="Sample", frameon=False, fontsize = 12) 
                ax[i].set_ylabel("Haplotype homozygosity", fontsize=12)
                ax[i].grid(b=True, which='both', color='0.65', linestyle='-')
            
            elif metrics[i] == "fws" or metrics[i] == "Fws":
            
                ax[i].plot(obs_times, self.diversity[:,5], marker='', color='blue', linewidth=2)
                ax[i].set_ylabel("Fws", fontsize=12)
                # ax[i].set_ylim(0,1)
                ax[i].grid(b=True, which='both', color='0.65', linestyle='-')
            
            elif metrics[i] == "fst" or metrics[i] == "Fst":
            
                ax[i].plot(obs_times, self.diversity[:,6], marker='', color='blue', linewidth=2)
                ax[i].set_ylabel("Fst", fontsize=12)
                # ax[i].set_ylim(0,1)
                ax[i].grid(b=True, which='both', color='0.65', linestyle='-')
                
            elif metrics[i] == "omega":
            
                ax[i].plot(obs_times, self.shared_segments[:, 1], marker='', color='blue', linewidth=2, label="between-host")
                ax[i].plot(obs_times, self.shared_segments[:, 4], marker='', color='red', linewidth=2, label="within-host")
                ax[i].set_ylabel("omega", fontsize=12)
                ax[i].grid(b=True, which='both', color='0.65', linestyle='-')                
                
            elif metrics[i] == "sigma":
            
                ax[i].plot(obs_times, self.shared_segments[:, 2], marker='', color='blue', linewidth=2, label="between-host")
                ax[i].plot(obs_times, self.shared_segments[:, 5], marker='', color='red', linewidth=2, label="within-host")
                ax[i].set_ylabel("sigma", fontsize=12)
                ax[i].grid(b=True, which='both', color='0.65', linestyle='-')             
                
            elif metrics[i] == "segment_length":
            
                ax[i].plot(obs_times, self.shared_segments[:, 3], marker='', color='blue', linewidth=2, label="between-host")
                ax[i].plot(obs_times, self.shared_segments[:, 6], marker='', color='red', linewidth=2, label="within-host")
                ax[i].set_ylabel("Fst", fontsize=12)
                ax[i].grid(b=True, which='both', color='0.65', linestyle='-')              

            elif metrics[i] == "N" or metrics[i] == "Nh":
            
                ax[i].plot(time_axis, self.parameters[:go_back_in_time, 0], marker='', color='blue', linewidth=2, label="Nh")
                ax[i].legend(title="Nh", frameon=True, fontsize = 12) 
                ax[i].set_ylabel("Nh", fontsize=12)
                ax[i].grid(b=True, which='both', color='0.65', linestyle='-')
                
            elif metrics[i] == "Q":
            
                ax[i].plot(time_axis, self.parameters[:go_back_in_time, 1], marker='', color='blue', linewidth=2, label="Nh")
                ax[i].legend(title="Q", frameon=True, fontsize = 12) 
                ax[i].set_ylabel("Q", fontsize=12)
                ax[i].grid(b=True, which='both', color='0.65', linestyle='-')                
            
            elif metrics[i] == "X" or metrics[i] == "chi":
            
                ax[i].plot(time_axis, self.parameters[:go_back_in_time, 2], marker='', color='blue', linewidth=2, label="\u03C7")
                ax[i].legend(title="\u03C7", frameon=True, fontsize = 12) 
                ax[i].set_ylabel("\u03C7", fontsize=12)
                ax[i].grid(b=True, which='both', color='0.65', linestyle='-')
                
            else:
                
                print('Sorry,' + ' "' + metrics[i] + '" ' + 'is not recognised as a metric by this plot.') 
                
        ax[i].set_xlabel("Generations", fontsize=12)
        ax[i].set_xlim(-go_back_in_time, 0)    
    
        plt.show()
        
##########################################################

def species(
    founder_duration = 1000,
    founder_N = 10,
    founder_Q = 3,
    founder_X = 0.2,
    
    R0 = 1.0005,
    expansion_Q = 3,
    expansion_X = 0.2,
    
    plateau_duration = 40000,
    plateau_N = 14660,
    plateau_Q = 3,
    plateau_X = 0.2 ):
       
    expansion = []
    N = founder_N
    ln_R0 = np.log(R0)
    
    while N < plateau_N - 1:
        
        x = (plateau_N - N) / (plateau_N - founder_N)
        
        N = N * np.exp(x * ln_R0)
 
        expansion.append(N)
        
    expansion_duration = len(expansion)

    duration = founder_duration + expansion_duration + plateau_duration
    
    parameters_ft = np.zeros((duration, 4))
    
    for ft in range(founder_duration):
        
        parameters_ft[ft, 0] = founder_N
        parameters_ft[ft, 1] = founder_Q
        parameters_ft[ft, 2] = founder_X
        
    N = founder_N
        
    for ft in range(expansion_duration):
        
        parameters_ft[ft + founder_duration, 0] = expansion[ft]
        parameters_ft[ft + founder_duration, 1] = expansion_Q
        parameters_ft[ft + founder_duration, 2] = expansion_X
        
    for ft in range(plateau_duration):
        
        parameters_ft[ft + founder_duration + expansion_duration, 0] = plateau_N
        parameters_ft[ft + founder_duration + expansion_duration, 1] = plateau_Q
        parameters_ft[ft + founder_duration + expansion_duration, 2] = plateau_X
        
    species = Population(parameters_ft)    
        
    return species   

#########################################################

def region(
    metapopulation = False,
    duration = 10000,
    Nh = 1000,
    Q = 3,
    X = 0.1,
    Nm = 10 ):
    
    history = [[duration, Nh, Q, X, Nm]]
       
    region = Population(history, metapopulation)    
        
    return region
        
