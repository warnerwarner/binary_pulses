# binary_pulses

Classes and functions for fitting and analyising data from binary experiments

## Data accessibility
This repository also contains processed single cell data used in the manuscript. Inside the units_usrt folder are three data structures, which contain the individual cell responses to each of the 3 odours used in this study. They are labelled as unit_usrt1, unit_usrt3, and unit_usrt5 respectively. Each object can be opened using numpy.load("unit_usrtX.npy", allow_pickle=True). Each object is a 4 dimensional tensor of the responses of all 'good' units extracted from the Kilosort unit extraction analysis. The objects are indexed by units x stimuli type x repeatition x timepoint. So unit_usrt[0,1,2,3] would contain the firing rate of unit 0 in response to the 3rd repeatition of the second stimulus pattern, at the third timepoint after stimulus onset (note, python objects are indexed from 0). 

The unit_usrt objects are not equal in size along all dimensions and as such when they are opened if you check the shape (e.g. unit_usrt.shape) will only return the shape of the first two dimensions (units and stimulus type). This is because each stimulus patterns were presented a different number of times across the units. The data can still be indexed as normal.

Any questions about data accessibility please open an issue.

