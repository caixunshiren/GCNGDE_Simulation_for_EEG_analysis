# This script converts the "All Seizures" html tables of EU metadata into a csvs file (one for clinical and one
# for subclinical seizures
import pandas as pd
import os
import numpy as np
import sys
import re

if len(sys.argv)==1:
    print('Usage: python szr_html_tables2csv.py patient#')
    print('For example: python szr_html_tables2csv.py 256')
    exit()
if len(sys.argv)!=2:
    raise Exception('Error: szr_html_tables2csv.py requires exactly 1 argument: patient#')

sub=sys.argv[1]

# Import table from html file which lists onset channels and subclinical seizures
#sub='253'
#sub='264'
#sub='1146'
#metadata_dir='/Users/davidgroppe/PycharmProjects/SZR_ANT/EU_METADATA'
#htable=pd.read_html(os.path.join(metadata_dir,'all_szrs_FR_'+sub+'.html'))
in_fname=os.path.joing('/Users/davidgroppe/PycharmProjects/SZR_ANT/EU_METADATA/ALL_SZR_HTML',
                       'all_szrs_FR_'+str(sub)+'.html')
print('Reading %s' % in_fname)
htable=pd.read_html(in_fname)
print('%d tables read' % len(htable))

# First table is CLINICAL SEIZURES<---USE ME!!!!
#htable[0].head()
## Second table is CLINICAL SEIZURES with slightly different info
#htable[1].head()
# 3rd table is SUBclinical SEIZURES
#htable[2].head()
# 4th table is IIDs
#htable[3].head()

#htable[0].to_csv(os.path.join(metadata_dir,'FR_'+sub+'_clinical_szrs.csv'))
#htable[2].to_csv(os.path.join(metadata_dir,'FR_'+sub+'_subclinical_szrs.csv'))
clin_out_fname='pat_FR_'+str(sub)+'_clinical_szrs.csv'
print('Creating file %s' % clin_out_fname)
htable[0].to_csv(clin_out_fname)
subclin_out_fname='pat_FR_'+str(sub)+'_subclinical_szrs.csv'
print('Creating file %s' % subclin_out_fname)
htable[2].to_csv(subclin_out_fname)



# Print Electrode Recommendation
print "Recommended Electrodes: "

onset_elec = []
onset_elec = []
early_elec = []
with open(clin_out_fname) as f:
    for line in f:
        m_onset = re.search(r"origin: (.*)  early:", line)
        m_early = re.search(r"early: (.*)  late:", line)
        if (m_onset):
        	onset_el = m_onset.group(1)
        	for elec in onset_el.split(","):
        		onset_elec.append(elec)

        if (m_early):
        	onset_el = m_early.group(1)
        	for elec in onset_el.split(","):
        		early_elec.append(elec)


onset_elec = sorted(set(onset_elec))

early_elec = sorted(set(early_elec))

print "Onset Electrodes:"
for el in onset_elec:
	print "'"+str(el)+"'; ..."

print "Early Electrodes:"
for el in early_elec:
	print "'"+str(el)+"'; ..."


# Handle Subclinical Seizure Formatting
new_content = []
with open(subclin_out_fname) as f:
    for line in f:
        regex = "^[\d]+,(\d+):,([\d]+).([\d.]+).'([\d.]+) (.*),(.*)$"
        test = re.search(regex,line);
        if (test):
            (num, day, mon, yr, strt, end) = [t(s) for t,s in zip((int,int,int,int,str, str),re.search(regex,line).groups())]
            newline = ("%i-%i-%i %s,%i-%i-%i %s" % ( (yr+2000), mon, day, strt, (yr+2000), mon, day, end) );
            new_content.append(newline)

f = open(subclin_out_fname, 'w')
for l in new_content:
  f.write("%s\n" % l)


# Handle Seizure Formatting
new_content = []
with open(clin_out_fname) as f:
    for line in f:
        regex = "[\d+],(\d+):.*eeg: ([\d]+).([\d]+).'([\d.]+) (.*),eeg: ([\d]+).([\d]+).'([\d.]+) (.*?),"
        test = re.search(regex,line);
        if (test):
            (num, day, mon, yr, strt, day2, mon2, yr2, end) = [t(s) for t,s in zip((int,int,int,int,str, int,int,int,str),re.search(regex,line).groups())]
            newline = ("%i-%i-%i %s,%i-%i-%i %s" % ( (yr+2000), mon, day, strt, (yr2+2000), mon2, day2, end) );
            new_content.append(newline)

f = open(clin_out_fname, 'w')
for l in new_content:
  f.write("%s\n" % l)  