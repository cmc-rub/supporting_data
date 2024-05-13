import numpy as np                                                                                                    
import pyblock                                                                                                        
import sys

cell = []
with open(sys.argv[1], "r") as logfile:
    count = 0
    for line in logfile:
        splitline = line.split()
        if splitline == []:
            continue
        if splitline[0] == "Step":
            count += 1
        if count == 3:
            for line in logfile:
                splitline = line.split()
                if splitline == []:
                    continue
                if splitline[0].isdigit():
                    cell.append([float(i) for i in splitline[16:19]])

cell = np.array(cell)                                                                                                 
cell_with_mean = np.zeros((cell.shape[0],cell.shape[1]+1))                                                            
cell_with_mean[:,:3]=cell[:]                                                                                   
cell_with_mean[:,3]=0.5*(cell[:,0]+cell[:,1])                                                                         
reblock_cell = pyblock.blocking.reblock(cell_with_mean.T)                                                             
opt_cell = pyblock.blocking.find_optimal_block(cell.shape[0], reblock_cell)                                           
block_err_cell=reblock_cell[opt_cell[0]].std_err                                                                      
mean = reblock_cell[opt_cell[0]].mean                                                                                 
print(mean, block_err_cell)
