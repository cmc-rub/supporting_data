import os

systems = [x for x in os.listdir('.') if os.path.isdir(x) == True]


for system in systems:
    os.chdir(system)
    dirs = os.listdir('.')
    for d in dirs:
        os.chdir(d)
        name = [x for x in os.listdir('.') if x.rsplit('.',1)[-1] == 'mfpx'][0].rsplit('.',1)[0]
        os.system('mv %s.ric %s.ric' % (name,system))
        os.system('mv %s.par %s.par' % (name,system))
        os.system('mv %s.mfpx %s.mfpx' % (name,system))
        os.chdir('../')
    os.chdir('../')
