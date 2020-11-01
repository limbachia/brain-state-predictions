import os
import subprocess
import pandas as pd

## Replaced by download_all()
#def download_raw():
#    '''
#    Download all participants preprocessed and filtered fMRI data
#    '''
#    os.system('src/data/sync-down.sh')

def run_process(cmd):
    process = subprocess.Popen(cmd,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE,
                               universal_newlines=True)

    stdout, stderr = process.communicate()
    print(stdout+'\n')
    print(stderr+'\n')
    
def download(subj,regs_fancy=True):
    '''
    Parameters
    ----------
    subj: participants's ID <string>
    regs_fancy: boolean
    
    Returns
    -------
    Downloads preprocessed, filtered functional data of a participant whose baseline.
    If regs_fancy = True, downloads stim file folder of the participant.
    '''
    if not os.path.isdir('data/raw/{}'.format(subj)):
        os.mkdir('data/raw/{}'.format(subj))
        
    cmd = ["rsync",
           "-rav",
           "-e",
           "ssh",
           "--include","*/",
           "climbach@login.bswift.umd.edu:/data/bswift-1/Pessoa_Lab/eCON/dataset/results_ShockCensored/{0}/uncontrollable/parametric/{0}_EP_TR_MNI_2mm_SI_denoised_NoBaseline.nii.gz".format(subj),
           "/home/climbach/approach-retreat/data/raw/{}".format(subj)]
    
    print("Downloading {}'s func data and its regressor files".format(subj))
    
    run_process(cmd)
    
    if regs_fancy:
        cmd = ["rsync",
               "-rav","-e",
               "ssh","--include","*_fancy/","--include","*_fancy/*","--exclude","*",
               "climbach@login.bswift.umd.edu:/data/bswift-1/Pessoa_Lab/eCON/dataset/preproc2/{0}/".format(subj),
               "/home/climbach/approach-retreat/data/raw/{}".format(subj)]
        
        run_process(cmd)
    
    
def download_all():
    """
    Downloads preprocessed, filtered functional data of all participant. 
    If regs_fancy = True, downloads stim file folder of the participant.
    """
    
    subjs = pd.read_excel('/home/climbach/approach-retreat/data/raw/CON_yoked_table.xlsx')
    subjs = subjs.query("use==1")['uncontrol'].values
    
    for subj in subjs:
        print("Downloading {}'s func data and its regressor files".format(subj))
        download(subj)
