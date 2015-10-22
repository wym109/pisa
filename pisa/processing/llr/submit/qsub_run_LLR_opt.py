#! /usr/bin/env python
#
# Initializes job files to be run on a computing cluster.
#
# NOTE: This script is highly cluster-specific, and will need to be
# modified for clusters different than the MSU GPU cluster.
#
#

import os
from argparse import ArgumentParser,ArgumentDefaultsHelpFormatter

def create_job_file_preamble():
    """
    Creates the first lines of the job pbs submission file that is
    common to all jobs, and stores it in a list to be written to the
    file later.
    """
   
    preamble = []
    preamble.append('#PBS -l walltime='+str(args.time)+':00:00\n')
    preamble.append('#PBS -l mem='+str(args.mem_size)+'gb,vmem=16gb\n')
    preamble.append('#PBS -l nodes=1:ppn=1:gpus=1\n')
    preamble.append("#PBS -l feature='gpgpu:intel14'\n")
    preamble.append('#\n')
    preamble.append('#PBS -m a\n')
    # Your email here if desired:
    #preamble.append('#PBS -M tca3+PBS@psu.edu\n')
    
    return preamble

parser = ArgumentParser(
    'Runs the LLR analysis on a number files, submitted to the rcc cluster. 
NOTE: log directory requires ABSOLUTE path!',
    formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('template_settings',type=str,
                    help="Settings file to use for template making.")
parser.add_argument('bfgs_settings',type=str,
                    help="minimizer settings file.")
parser.add_argument('trials_per_file',type=str,
                    help="Number of trials per file.")
parser.add_argument('num_files',type=int,help="Number of files to run.")
parser.add_argument('outdir',type=str,help="location to store the output data.")
parser.add_argument('logfile_dir',type=str,
                    help="logfile to write output (both stderr/stdin) to")
parser.add_argument('--no_alt_fit',action='store_true',default=False,
                    help="No alt hierarchy fit.")
parser.add_argument('--single_octant',action='store_true',default=False,
                    help="single octant in llh only.")
parser.add_argument('--file_begin_tag',type=int,default=1,
                    help="file number to begin tagging files.")
parser.add_argument('--job_file_dir',type=str,
                    default=os.path.expandvars('$HOME/work/pisa_related/pisa_analysis/gpu_analysis/job_files/'),
                    help="directory of job files.")
parser.add_argument('--time',type=int,default=4,
                    help="walltime in hours of job.")
parser.add_argument('--mem_size',type=int,default=7,
                    help="size of memory to request in GB.")
args = parser.parse_args()


scriptname = os.path.join(
    os.path.expandvars("$PISA"),
    "/pisa/analysis/llr/LLROptimizerAnalysis.py")

# step 1: Create job file preamble
job_file_common = []
job_file_common = create_job_file_preamble()
job_name_base = "pbs_run_LLR_opt.in"

# step 2: loop over num_files and create job files to be given to qsub_wrapper.sh
for ifile in xrange(args.num_files):
    file_tag = args.file_begin_tag + ifile
    jobFile = os.path.join(args.job_file_dir,job_name_base+"_%03d"%file_tag)
    
    outfilename = os.path.join(args.outdir,"llh_data_%d"%file_tag)
    
    print "  Creating file: "+jobFile+"."
    fh = open(jobFile,"w")
    for line in job_file_common: fh.write(line)
    logfile = os.path.join(args.logfile_dir,'log_'+str(file_tag))
    os.system('touch '+logfile)

    fh.write('#PBS -j oe\n')
    fh.write('#PBS -o '+logfile+'\n')
    fh.write('\n')
    fh.write('module load cuda\n')
    fh.write('module list\n')
    fh.write('\n')
    fh.write('cd $PBS_O_WORKDIR\n')
    fh.write('PATH=/mnt/home/tarlen/software/anaconda/bin:$PATH\n')
    fh.write('\n')
    
    # This is to test the gpu usage and properties especially useful
    # when you have a heterogenous cluster of GPUs.
    fh.write('\n'+os.path.expandvars("$PISA/pisa/processing/submit/")+'ping_gpu.py\n\n')
    fh.write('nvidia-smi\n\n')

    command = scriptname+" -t "+args.template_settings+" -m "+args.bfgs_settings+" -n "+str(args.trials_per_file)+" --outfile="+outfilename
    if args.single_octant: command +=" --single_octant"
    if args.no_alt_fit: command+=" --no_alt_fit"
    
    fh.write(command+'\n')
    fh.close()

print "using script: ",scriptname    
print "DONE! Finished creating files in: "+args.job_file_dir
