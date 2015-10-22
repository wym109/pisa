#
# author: Timothy C. Arlen
#         tca3@psu.edu
#
# date:   18 September 2015
#
# Utility functions that handle the processing of very unstructured
# log text files.
#
# WARNING: If something about the formatting of the log files change,
# some of these utility functions must also change.
#

from __future__ import division, print_function
import re


def getTimeStamp(iline, logfile_lines):
    """
    Searches through beginning of logfile until it finds the timestamp
    in the format: Day MMM DD HH:MM:SS YYYY
      ex: Sun Aug 30 04:16:16 2015

    \Params:
      * iline - line number in the logfile
      * logfile_lines - all lines of logfile in a list, one line per element

    \Returns:
      * iline - updated to right past the line of the time stamp
      * time_stamp - time stamp as a string

    """

    # matches expressions of the form Wkday Month Day HH:MM::SS YYYY
    expr1 = '[A-Z][a-z][a-z] [0-3][0-9] [0-2][0-9]:[0-5][0-9]:[0-5][0-9] \d\d\d\d'
    expr2 = '[A-Z][a-z][a-z] [ ][0-9] [0-2][0-9]:[0-5][0-9]:[0-5][0-9] \d\d\d\d'

    # find date, then break
    time_stamp = ""
    while iline < len(logfile_lines):
        line = logfile_lines[iline].rstrip()
        line_split = line.split()
        if re.search(expr1, line):
            time_stamp = line
            break
        elif re.search(expr2, line):
            time_stamp = line
            break
        iline+=1

    if time_stamp == "":
        raise ValueError("file was searched and no timestamp was found!")

    return time_stamp, iline


def getTrialStart(iline, all_lines, llr_type):
    """
    Finds the current values of data_h, hypo_h, itrial at this line.
    Also increments iline by one, and returns it.

    Assumes the lines will be like the following format:
    -----------------------------------------
    [    INFO] start trial 1
    [    INFO] Finding best fit for true_NMH under hypo_NMH assumption
    -----------------------------------------

    \Params:
      * iline - line number in the logfile
      * all_lines - all lines of a logfile in a list, one per element.

    \Returns:
      * data_h - Normal or Inverted pseudo data ('true_NMH' or 'true_IMH')
      * hypo_h - Normal or Inverted hypothesis ('hypo_NMH' or 'hypo_IMH')
      * itrial - current trial number in this log file
      * iline - new line nmber after finding the start of the trials.
    """

    itrial = int(all_lines[iline].split()[-1])
    iline+=1

    line_split = all_lines[iline].split()
    data_h, hypo_h = line_split[-4], line_split[-2]
    iline+=1

    #print("llr_type: ",llr_type," trial: ",itrial)

    return data_h, hypo_h, itrial, iline


def collectRunInfo(logging_info, all_lines, iline, mean_template_time):
    """
    Collects optimizer run information, as defined in log_dict (within
    this function). Puts this information into logging_dict.

    \Modifies:
      * logging_info - path in output data logging dict that the run log
        info will be written to.

    \Returns:
      * iline - incremented by the number of lines read in.
    """

    log_dict = {"optimizer_time": 0.0,
                "mean_template_time": mean_template_time,
                "warnflag": 0, "task": "", "nit": "", "funcalls": ""}
    log_dict["warnflag"] = int(all_lines[iline].split()[-1])
    iline += 1
    log_dict["task"] = "".join(all_lines[iline].split()[4:])
    iline+=2
    log_dict["nit"] = int(all_lines[iline].split()[-1])
    iline+=1
    log_dict["funcalls"] = int(all_lines[iline].split()[-1])
    iline+=1
    log_dict["optimizer_time"] = float(all_lines[iline].split()[-2])
    iline+=1

    for k,v in log_dict.items(): logging_info[k].append(v)

    return iline


def processLogFile(iline, logfile_lines, output_data):
    """
    Process the logfile for all logging information for all trials run during
    this partial run.

    \Params:
      * iline - line number in the logfile
      * logfile_lines - all lines of a logfile in a list, one per element.

    \Modifies:
      * output_data - all logging information gets added here only into the
        leaf nodes.

    \Returns:
      * None
    """

    itrial = 0
    data_h = ""
    hypo_h = ""
    llr_type = 'true_h_fiducial'  # then switch to 'false_h_best_fit'
                                  # when the trials are finished...

    avg_template_time = 0.0
    n_template_calls = 0

    #print("total number of lines to process: ",len(logfile_lines))
    while iline < len(logfile_lines):
        
        # Until we get nonzero itrial, keep advancing in the file
        if itrial == 0:
            if "[    INFO] start trial" in logfile_lines[iline]:
                data_h, hypo_h, itrial, iline = getTrialStart(
                    iline, logfile_lines, llr_type)
            else:
                iline += 1
                continue

        else:
            # When in main loop, only two things can occur:
            #   1) get template completion time
            #   2) end of optimizer
            if ("[    INFO] ==> elapsed time for template maker:"
                in logfile_lines[iline]):
                avg_template_time += float(logfile_lines[iline].split()[-2])
                n_template_calls += 1
                iline+=1

            if ("]  warnflag :" in logfile_lines[iline]):

                # optimizer run has ended! Collect all info now:
                iline = collectRunInfo(output_data[data_h][llr_type][hypo_h],
                                       logfile_lines, iline,
                                       (avg_template_time/n_template_calls))


                #---------------------------------------------------------
                # Now one of three things happens:
                # (If logfile format changes, this will break...)
                #---------------------------------------------------------
                # 1) next line is 'Finding best fit...' and it is same trial but
                #    different hypothesis
                # 2) next line is 'stop trial...' and the new pseudo data set is
                #    formed for same llr_type
                # 3) 'start trial...' immediately ensues, signifying a change in
                #    llr_type
                if ("[    INFO] Finding best fit for" in logfile_lines[iline]):
                    line_split = logfile_lines[iline].split()
                    data_h, hypo_h = line_split[-4], line_split[-2]
                    iline+=1
                elif ("[    INFO] stop trial" in logfile_lines[iline]):
                    iline+=1

                    # First check if end of file:
                    if iline >= len(logfile_lines):
                        #print("Exiting...")
                        break

                    # If next line is the start of a new trial, then
                    # proceed. If we instead get a ValueError, then this
                    # means that the llr_type ended and it's time to move
                    # down to the next time it starts a new trial:
                    try:
                        data_h, hypo_h, itrial, iline = getTrialStart(
                            iline, logfile_lines, llr_type)
                    except ValueError:
                        # Skip over the lines until you get to the
                        # starting trial
                        while True:
                            if ("[    INFO] start trial 1" in
                                logfile_lines[iline]):

                                if 'true_h' in llr_type:
                                    llr_type = 'false_h_best_fit'
                                else:
                                    llr_type = 'true_h_fiducial'

                                data_h, hypo_h, itrial, iline = getTrialStart(
                                    iline, logfile_lines, llr_type)
                                iline+=1
                                break
                            iline+=1
                elif ("[    INFO] start trial 1" in
                      logfile_lines[iline]):
                    
                    if 'true_h' in llr_type:
                        llr_type = 'false_h_best_fit'
                    else:
                        llr_type = 'true_h_fiducial'
                        
                    data_h, hypo_h, itrial, iline = getTrialStart(
                        iline, logfile_lines, llr_type)
                    iline+=1
                    
                else:
                    print ("\n\nLine Number:", iline )
                    print (logfile_lines[iline-2].rstrip())
                    print (logfile_lines[iline-1].rstrip())
                    print (logfile_lines[iline].rstrip())
                    print (logfile_lines[iline+1].rstrip())
                    raise Exception(
                        "Failed to find correct output after optimization!")
        iline+=1

    return
