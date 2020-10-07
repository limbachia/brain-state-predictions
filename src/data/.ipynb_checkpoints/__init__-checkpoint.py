import os
import subprocess

def download_raw():
    '''
    
    Downloads approah-retreat fMRI segment pkl files into /data/raw for different
    brain regions along with their masks from bswift 
    
    '''
    os.system('src/data/sync-down.sh')
    
    
def download(subj,regs_fancy=True):
    '''
    Download functional data for subj whose baseline has been filtered out.
    Also, downloads stim file folder of the the subj
    '''
    
    def run_process(cmd):
        process = subprocess.Popen(cmd,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE,
                                   universal_newlines=True)

        stdout, stderr = process.communicate()
        print(stdout+'\n')
        print(stderr+'\n')
    
    if not os.path.isdir('data/external/{}'.format(subj)):
        os.mkdir('data/external/{}'.format(subj))
        
    cmd = ["rsync",
           "-rav","-e",
           "ssh","--include","*/",
           "climbach@login.bswift.umd.edu:/data/bswift-1/Pessoa_Lab/eCON/dataset/results_ShockCensored/{0}/uncontrollable/parametric/{0}_EP_TR_MNI_2mm_SI_denoised_NoBaseline.nii.gz".format(subj),
           "/home/climbach/approach-retreat/data/external/{}".format(subj)]
    
    run_process(cmd)
    
    if regs_fancy:
        cmd = ["rsync",
               "-rav","-e",
               "ssh","--include","*/",
               "climbach@login.bswift.umd.edu:/data/bswift-1/Pessoa_Lab/eCON/dataset/preproc2/{0}/regs_fancy".format(subj),
               "/home/climbach/approach-retreat/data/external/{}".format(subj)]
        
        run_process(cmd)
