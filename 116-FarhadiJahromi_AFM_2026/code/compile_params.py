import os
import sys
import numpy as np

with open("fit_keys.out", "r") as keyfile:
    # Source - https://stackoverflow.com/a/54278929
    # Posted by Eugene Yarmash, modified by community. See post 'Timeline' for change history
    # Retrieved 2026-01-28, License - CC BY-SA 4.0
    # Modified by Babak Farhadi Jahromi
    with open(sys.argv[1], "rb") as parfile:
        keyline = keyfile.readline().split()
        # get contents of last line in CMAES output file
        try:  # catch OSError in case of a one line file
            parfile.seek(-2, os.SEEK_END)
            while parfile.read(1) != b'\n':
                parfile.seek(-2, os.SEEK_CUR)
        except OSError:
            parfile.seek(0)
        parline = parfile.readline().decode().split()[5:]
        # set up parameter dicitionary
        npars = len(parline)
        pardict = {}
        idict = {"sigma":0, "Jii":1, "EN":2}
        for i in range(0, npars):
            par = parline[i]
            ptype = keyline[2*i]
            at = keyline[2*i+1]
            if at not in pardict.keys():
                pardict[at] = np.zeros((3))
            pardict[at][idict[ptype]] = par

# write parameter file
with open("topoqeq.par", "w") as outfile:
    for i in sorted(pardict.keys()):
        params = pardict[i]
        outfile.write(f"{i:16} {0.001*params[0]:12.6f} {params[1]:12.6f} {params[2]:12.6f}\n")
