import glob
import sys
import os

test_directory = os.path.dirname(os.path.abspath(__file__)) + os.path.sep + os.path.join('..', '**')
def test_print(filesToCheck=test_directory):
    """
    Tests for bad print statements that will fail in python 3.

    Tests the entire pyklip directory for bad print statements in
    python files that would break if run on python 3.
    If there is ever a "print " as you would normally print in python 2, it will throw
    a syntax error with all the files and lines that the bad print statements were found in.
    Files should be written using python 3's print() function.

    Args:
        filesToCheck: String of location and type of files to check.
        Defaults to python files in pyklip.

    Raises:
        SyntaxError: Bad print statements in:
            File: File
                Lines: Lines
    """

    #gathers all python files in pyklip directory recursively
    if sys.version_info > (3,4):
        #python 3.5+ behavior for recursive globbing
        filesToCheck = filesToCheck + os.path.sep + '*.py'
        files = glob.iglob(filesToCheck, recursive=True)    
    else:
        #python 2.2->3.4 behavior for recursive globbing
        import fnmatch
        files = []
        for root, dirname, filenames in os.walk('..'):
            for filename in fnmatch.filter(filenames, '*.py'):
                files.append(os.path.join(root, filename))
    
    #dictionary to hold all bad prints. (Key, Value) = ((string) File, (list) Line)
    bad_prints = {}
    for file_name in files:
        with open(file_name) as f:
            content = f.readlines()
        linecount = 1
        #used to check multiline comments such as doc strings.
        multiline_comment = False
        for line in content:
            #Checking for multiline comments before skipping.
            if '\"\"\"' in line:
                multiline_comment = not multiline_comment
            if multiline_comment:
                linecount += 1
                continue
            #splits line by spaces and ignores trailing whitespace
            split_line = line.strip().split(" ")
            if split_line[0] == "print" or "print\"" in split_line[0][:6] or "print\'" in split_line[0][:6]:
                #initializes list in dictionary if it doesn't exist. 
                bad_prints.setdefault(file_name, []).append(linecount)
            linecount += 1

    #if anything exists in the bad_print dictionary raise a Syntax Error.
    if any(bad_prints):
        error_message = ''
        for key in bad_prints.keys():
            error_message = error_message + '\tFile: ' + str(key) + '\n' + '\t\tLines: ' + str(bad_prints[key]) + '\n'
        raise SyntaxError('Bad print statements in:' + '\n' + error_message)

if __name__ == "__main__":
    test_print()
