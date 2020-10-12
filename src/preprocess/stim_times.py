import numpy as np

TR_ref = np.arange(0,360*1.25,1.25)

def getSelectedStims(line):
    """
    Parameter
    ---------
    line: single line of strings that consist of multiple stim onsets and durations.
            For example: "2.20:8.97 15.50:7.25 30.00:6.36\n"
    
    Returns
    -------
    tuple of numpy.1darray
        sel_onsets, sel_durs: those onsets which have duration between 7.5-8.75 seconds.
    """
    onsets = np.array([float(onset) for onset,_ in [pair.split(':') for pair in line.split()]])
    durs = np.array([float(dur) for _,dur in [pair.split(':') for pair in line.split()]])
    sel_onsets = onsets[np.logical_and(durs>=7.5, durs<=8.75)]
    sel_durs = durs[np.logical_and(durs>=7.5, durs<=8.75)]
    return sel_onsets, sel_durs

def getApprRetrCorrespondingIdx(peak,re_on):
    """
    Parameters
    ----------
    peak: peak times of approah (onset + duration)
    re_on: retreat onsets 
    
    Returns
    -------
    ap_indx: numpy.1darray of approach onset indices that
        match with retreat onset indices
    """
    ap_idx = []; re_idx = []
    for i,p in enumerate(peak):
        arr = np.array([abs(p - num) for num in re_on])
        if (arr.round(0) == 0).any():
            re_idx.append(np.where(arr.round(0)==0)[0][0])
            ap_idx.append(i)
            
    #np.testing.assert_array_equal(np.array(ap_idx),np.array(re_idx))
    assert len(ap_idx) == len(re_idx)
    
    return ap_idx


def return_closest_TR(timings):
    """
    Parameter
    ---------
    timings: numpy.1darray of stim onsets
    
    Returns
    -------
    tuple of numpy.1darray,
        TR_time, TRs = TR times that closesly match with the stim onsets, TR number
    """
    TR_time = []; TRs = []
    for n in timings:
        time = min(TR_ref, key=lambda x:abs(x-n))
        TR_time.append(time)
        TRs.append(np.where(TR_ref==time)[0][0])
    return np.array(TR_time), np.array(TRs)