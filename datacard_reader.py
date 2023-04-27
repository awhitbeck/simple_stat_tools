#reads text file as space separated values
from first import multibin_likelihood, likelihood, naive_limit
import csv
import matplotlib.pyplot as plt

class datacard:
    def __init__(self, filename):
        self.filename = filename

        self.bin_name = ""
        self.obs = []
        self.process_name = []
        self.process_index = []
        self.rate = []
        self.systematic = []
        self.systematic_name = []
        self.systematic_type = []
        self.raw = self.read_datacard(filename)

        #check if rate, process_name, process_index are the same length
        if len(self.rate) != len(self.process_name) or len(self.rate) != len(self.process_index) :
            print("ERROR: rate, process_name, process_index are not the same length")

    def read_datacard(self,filename):
        with open(filename) as f:
            reader = csv.reader(f, delimiter=' ')
            #delete element whose first element is '#'
            reader = [row for row in reader if row[0] != '#']
            #delete empty elements
            reader = [row for row in reader if row != []]
            #delete elements that contains '---'
            reader = [row for row in reader if not '---' in row[0] ]
            #delete elements that start with 'jmax','*'
            reader = [row for row in reader if len(row)>1 and row[0] != 'jmax' and row[1] != '*' or len(row)<=1 ]
            #delete elements that start with 'imax','*'
            reader = [row for row in reader if len(row)>1 and row[0] != 'imax' and row[1] != '*' or len(row)<=1 ]
            #delete elements that start with 'kmax','*'
            reader = [row for row in reader if len(row)>1 and row[0] != 'kmax' and row[1] != '*' or len(row)<=1 ]

            #extract bin name from row where first element is 'bin'
            bin_name=[]
            bin_name.append([row for row in reader if 'bin' in row[0]])
            self.bin_name = bin_name[0][0][0].split('\t')[1]

            #extract observation from row where first element is 'observation'
            observation = [row for row in reader if row[0] == 'observation']

            #delete elements of observation that are ''
            observation = [col for col in observation[0] if not col in ['observation','']]

            #extract observation values
            if len(observation) != 1 :
                print("ERROR: observation malformed")
            else :
                self.obs = int(observation[0])

            process = []
            process =  [ row for row in reader if 'process' in row[0] ]
            process[0] = process[0][0].split('\t')
            process[1] = process[1][0].split('\t')
            #extract process name from first row where first element is 'process'
            self.process_name = [col for col in process[0] if not col in ['process','']]

            #extract process name from second row where first element is 'process'
            self.process_index = [int(col) for col in process[1] if not col in ['process','']]

            #extract rates from row where first element is 'rate'
            rate = [ row for row in reader if 'rate' in row[0] ]
            #detele empty elements and elemets that are 'rate'

            rate = rate[0][0].split('\t')
            self.rate = [float(col) for col in rate if not col in ['rate','']]

            #extract systematics from rows contianing 'lnN'
            self.systematic = [ row for row in reader if 'lnN' in row[0] ]
            self.systematic = [ row[0].split('\t') for row in self.systematic ]
            self.systematic_name = [ row[0] for row in self.systematic ]
            self.systematic_type = [ row[1] for row in self.systematic ]
            self.systematic = [ row[2:] for row in self.systematic ]

            return list(reader)

    def build_likelihood(self):
        #build likelihood function
        self.signal_rate = [self.rate[i] for i in range(len(self.rate)) if self.process_index[i] <= 0]
        self.background_rate = [self.rate[i] for i in range(len(self.rate)) if self.process_index[i] > 0]
        print('signal_rate: ',self.signal_rate)
        print('background_rate: ',self.background_rate)

        llh = likelihood(int(sum(self.background_rate)),self.background_rate,self.signal_rate)
        llh.nuis_name = self.systematic_name
        llh.nuis_type = self.systematic_type
        llh.nuis_unc = self.systematic

    def print_datacard(self):
        print('filename: ',self.filename)
        print('raw: ',self.raw)
        print('bin_name: ',self.bin_name)
        print('obs: ',self.obs)
        print('process_name: ',self.process_name)
        print('process_index: ',self.process_index)
        print('rate: ',self.rate)
        print('systematic: ',self.systematic)
        print('systematic_name: ',self.systematic_name)
        print('systematic_type: ',self.systematic_type)

### read in datacards
d11 = datacard('data/CN500_WH_BB_EvtCat_11.txt')
d11.print_datacard()
d10 = datacard('data/CN500_WH_BB_EvtCat_10.txt')
#d10.print_datacard()
d9 = datacard('data/CN500_WH_BB_EvtCat_9.txt')
#d9.print_datacard()
d8 = datacard('data/CN500_WH_BB_EvtCat_8.txt')
#d8.print_datacard()

### build likelihood function
llh = multibin_likelihood(4)
llh.bin.append(d11.build_likelihood())
llh.bin.append(d10.build_likelihood())
llh.bin.append(d9.build_likelihood())
llh.bin.append(d8.build_likelihood())

### compute observed and expected limit
limit = naive_limit(llh)
limit.plot_likelihood(0.0,10,1000)
#print('limit: ',limit.compute_limit())
#l = limit.run_toy_experiments(0.0,300)
### plot distributions of limits from toys
#plt.hist(l,bins=10)
#plt.show()