#!/usr/bin/env python
# pylint: disable=exec-used, eval-used


"""
Find and run PISA unit test functions
"""


from os import walk
from os.path import abspath, dirname, expanduser, expandvars, isfile, join, relpath
import sys

from pisa.utils.fileio import nsort_key_func
from pisa.utils.log import Levels, logging, set_verbosity


PISA_PATH = abspath(join(dirname(dirname(__file__)), "pisa"))


def find_test_funcs_in_file(filepath):
    """Find test functions defined by "def test_*" within a file at `filepath`

    Parameters
    ----------
    filepath : str
         Path to python file

    Returns
    -------
    tests : list of str

    """
    tests = []
    with open(filepath, "r") as f:
        for line in f.readlines():
            tokens = line.split()
            if tokens and tokens[0] == "def" and tokens[1].startswith("test_"):
                funcname = tokens[1].split("(")[0].strip()
                tests.append(funcname)
    return tests


def find_tests(path):
    """Find tests to run within `path` (within the file at `path` or recusively
    within all .py files within the directory at `path`).

    Parameters
    ----------
    path : str
        Path to a file or directory

    Returns
    -------
    tests : dict
        Each key is a relative path to the .py file containing "test_*"
        functions and each value is a list of the "test_*" function names
        within that file.

    """
    path = expanduser(expandvars(path))

    tests = {}
    if isfile(path):
        return find_test_funcs_in_file(path)

    for dirpath, dirs, files in walk(path, followlinks=True):
        files.sort(key=nsort_key_func)
        dirs.sort(key=nsort_key_func)

        for filename in files:
            if not filename.endswith(".py"):
                continue
            filepath = join(dirpath, filename)
            these_tests = find_test_funcs_in_file(filepath)
            if these_tests:
                tests[relpath(filepath, start=PISA_PATH)] = these_tests

    return tests


def run_tests(path):
    """Run all tests found at `path` (or recursively below if `path` is a
    directory).

    Each module is imported and each test function is run initially with
    `set_verbosity(Levels.WARN)`, but if an exception is caught, the module is
    re-imported / functio is re-run with `set_verbosity(Levels.TRACE)`, then
    the traceback from the (original) exception emitted is displayed.

    Parameters
    ----------
    path : str
        Path to file or directory

    """
    path = expanduser(expandvars(path))

    tests = find_tests(path)

    module_pypaths_succeeded = []
    module_pypaths_failed = []
    test_pypaths_succeeded = []
    test_pypaths_failed = []

    for rel_file_path, test_func_names in tests.items():
        pypath = ["pisa"] + rel_file_path[:-3].split("/")
        parent_pypath = ".".join(pypath[:-1])
        module = pypath[-1].replace(".", "_")
        module_pypath = f"{parent_pypath}.{module}"

        # Don't output anything unless we encounter an error
        set_verbosity(Levels.WARN)
        try:
            exec(f"from {parent_pypath} import {module}")
        except Exception as err:
            module_pypaths_failed.append(module_pypath)

            sys.stdout.write(f"<< FAILURE IMPORTING : {module_pypath}\n\n")

            # Reproduce the failure with full output
            set_verbosity(Levels.TRACE)
            try:
                exec(f"from {parent_pypath} import {module}")
            except Exception:
                pass

            # Print the exception that occurred
            logging.exception(err)

            sys.stdout.write(f"\n   FAILURE IMPORTING : {module_pypath} >>\n\n\n\n")
            continue

        module_pypaths_succeeded.append(module_pypath)

        for test_func_name in test_func_names:
            test_pypath = f"{module_pypath}.{test_func_name}"

            # Don't output anything unless we encounter an error
            set_verbosity(Levels.WARN)
            try:
                test_func = eval(f"{module}.{test_func_name}")
                test_func()

            except Exception as err:
                test_pypaths_failed.append(test_pypath)

                sys.stdout.write(f"<< FAILURE RUNNING : {test_pypath}\n\n")

                # Reproduce the error with full output
                set_verbosity(Levels.TRACE)
                try:
                    test_func = eval(f"{module}.{test_func_name}")
                    test_func()
                except Exception:
                    pass

                # Print the exception that occurred
                logging.exception(err)

                sys.stdout.write(f"\n   FAILURE RUNNING : {test_pypath} >>\n\n\n\n")

            else:
                test_pypaths_succeeded.append(test_pypath)

    # Summarize results

    modules_loaded = len(module_pypaths_succeeded)
    modules_failed_to_import = len(module_pypaths_failed)
    tests_succeeded = len(test_pypaths_succeeded)
    tests_failed = len(test_pypaths_failed)

    set_verbosity(Levels.INFO)
    logging.info(
        f"PISA TESTS : {modules_loaded} modules loaded,"
        f" {modules_failed_to_import} modules failed to load"
    )
    logging.info(
        f"PISA TESTS : {tests_succeeded} tests succeeded,"
        f" {tests_failed} tests failed"
    )

    # Exit with error if any failures (import or test failure)

    if modules_failed_to_import or tests_failed:
        raise Exception(
            f"{modules_failed_to_import} import errors and {tests_failed} test failures"
        )


if __name__ == "__main__":
    run_tests(PISA_PATH)
