"""Implementation of tools to do with system administration made as platform independent as possible.


"""

import sys
import os
import urllib
import urllib2
import getpass
import tarfile
import md5


def log_to_file(filename, s, verbose=False):
    """Log string to file name
    """

    fid = open(filename, 'a')
    if verbose: print s
    fid.write(s + '\n')
    fid.close()


def get_user_name():
    """Get user name provide by operating system
    """

    if sys.platform == 'win32':
        #user = os.getenv('USERPROFILE')
        user = os.getenv('USERNAME')
    else:
        user = os.getenv('LOGNAME')


    return user    

def get_host_name():
    """Get host name provide by operating system
    """

    if sys.platform == 'win32':
        host = os.getenv('COMPUTERNAME')
    else:
        host = os.uname()[1]


    return host    

def get_revision_number():
    """Get the version number of this repository copy.

    Try getting data from stored_version_info.py first, otherwise
    try using SubWCRev.exe (Windows) or svnversion (linux), otherwise
    try reading file .svn/entries for version information, otherwise
    throw an exception.

    NOTE: This requires that the command svn is on the system PATH
    (simply aliasing svn to the binary will not work)
    """

    def get_revision_from_svn_entries():
        '''Get a subversion revision number from the .svn/entires file.'''

        msg = '''
No version info stored and command 'svn' is not recognised on the system PATH.

If ANUGA has been installed from a distribution e.g. as obtained from SourceForge,
the version info should be available in the automatically generated file
'stored_version_info.py' in the anuga root directory.

If run from a Subversion sandpit, ANUGA will try to obtain the version info by
using the command 'svn info'.  In this case, make sure the command line client
'svn' is accessible on the system path.  Simply aliasing 'svn' to the binary will
not work.

If you are using Windows, you have to install the file svn.exe which can be
obtained from http://www.collab.net/downloads/subversion.

Good luck!
'''

        try:
            fd = open(os.path.join('.svn', 'entries'))
        except:
            raise Exception, msg

        line = fd.readlines()[3]
        fd.close()
        try:
            revision_number = int(line)
        except:
            msg = ".svn/entries, line 4 was '%s'?" % line.strip()
            raise Exception, msg

        return revision_number

    def get_revision_from_svn_client():
        '''Get a subversion revision number from an svn client.'''

        if sys.platform[0:3] == 'win':
            try:
                fid = os.popen(r'C:\Program Files\TortoiseSVN\bin\SubWCRev.exe')
            except:
                return get_revision_from_svn_entries()
            else:
                version_info = fid.read()
                if version_info == '':
                    return get_revision_from_svn_entries()

            # split revision number from data
            for line in version_info.split('\n'):
                if line.startswith('Updated to revision '):
                    break

            fields = line.split(' ')
            msg = 'Keyword "Revision" was not found anywhere in text: %s' % version_info
            assert fields[0].startswith('Updated'), msg

            try:
                revision_number = int(fields[3])
            except:
                msg = ("Revision number must be an integer. I got '%s' from "
                       "'SubWCRev.exe'." % fields[3])
                raise Exception, msg
        else:                   # assume Linux
            try:
                fid = os.popen('svn info 2>/dev/null')
            except:
                return get_revision_from_svn_entries()
            else:
                version_info = fid.read()
                if version_info == '':
                    return get_revision_from_svn_entries()

            # split revision number from data
            for line in version_info.split('\n'):
                if line.startswith('Revision:'):
                    break

            fields = line.split(':')
            msg = 'Keyword "Revision" was not found anywhere in text: %s' % version_info
            assert fields[0].startswith('Revision'), msg

            try:
                revision_number = int(fields[1])
            except:
                msg = ("Revision number must be an integer. I got '%s' from "
                       "'svn'." % fields[1])
                raise Exception, msg

        return revision_number

    # try to get revision information from stored_version_info.py
    try:
        from anuga.stored_version_info import version_info
    except:
        return get_revision_from_svn_client()

    # split revision number from data
    for line in version_info.split('\n'):
        if line.startswith('Revision:'):
            break

    fields = line.split(':')
    msg = 'Keyword "Revision" was not found anywhere in text: %s' % version_info
    assert fields[0].startswith('Revision'), msg

    try:
        revision_number = int(fields[1])
    except:
        msg = ("Revision number must be an integer. I got '%s'.\n"
               'Check that the command svn is on the system path.'
               % fields[1])
        raise Exception, msg

    return revision_number


def store_version_info(destination_path='.', verbose=False):
    """Obtain current version from Subversion and store it.
    
    Title: store_version_info()

    Author: Ole Nielsen (Ole.Nielsen@ga.gov.au)

    CreationDate: January 2006

    Description:
        This function obtains current version from Subversion and stores it
        is a Python file named 'stored_version_info.py' for use with
        get_version_info()

        If svn is not available on the system PATH, an Exception is thrown
    """

    # Note (Ole): This function should not be unit tested as it will only
    # work when running out of the sandpit. End users downloading the
    # ANUGA distribution would see a failure.
    #
    # FIXME: This function should really only be used by developers (
    # (e.g. for creating new ANUGA releases), so maybe it should move
    # to somewhere else.
    
    import config

    try:
        fid = os.popen('svn info')
    except:
        msg = 'Command "svn" is not recognised on the system PATH'
        raise Exception(msg)
    else:    
        txt = fid.read()
        fid.close()


        # Determine absolute filename
        if destination_path[-1] != os.sep:
            destination_path += os.sep
            
        filename = destination_path + config.version_filename

        fid = open(filename, 'w')

        docstring = 'Stored version info.\n\n'
        docstring += 'This file provides the version for distributions '
        docstring += 'that are not accessing Subversion directly.\n'
        docstring += 'The file is automatically generated and should not '
        docstring += 'be modified manually.\n'
        fid.write('"""%s"""\n\n' %docstring)
        
        fid.write('version_info = """\n%s"""' %txt)
        fid.close()


        if verbose is True:
            print 'Version info stored to %s' %filename

def safe_crc(string):
    """64 bit safe crc computation.

       See Guido's 64 bit fix at http://bugs.python.org/issue1202            
    """

    from zlib import crc32
    import os

    x = crc32(string)
        
    if os.name == 'posix' and os.uname()[4] in ['x86_64', 'ia64']:
        crcval = x - ((x & 0x80000000) << 1)
    else:
        crcval = x
        
    return crcval


def compute_checksum(filename, max_length=2**20):
    """Compute the CRC32 checksum for specified file

    Optional parameter max_length sets the maximum number
    of bytes used to limit time used with large files.
    Default = 2**20 (1MB)
    """

    fid = open(filename, 'rb') # Use binary for portability
    crcval = safe_crc(fid.read(max_length))
    fid.close()

    return crcval

def get_pathname_from_package(package):
    """Get pathname of given package (provided as string)

    This is useful for reading files residing in the same directory as
    a particular module. Typically, this is required in unit tests depending
    on external files.

    The given module must start from a directory on the pythonpath
    and be importable using the import statement.

    Example
    path = get_pathname_from_package('anuga.utilities')

    """

    exec('import %s as x' %package)

    path = x.__path__[0]
    
    return path

    # Alternative approach that has been used at times
    #try:
    #    # When unit test is run from current dir
    #    p1 = read_polygon('mainland_only.csv')
    #except: 
    #    # When unit test is run from ANUGA root dir
    #    from os.path import join, split
    #    dir, tail = split(__file__)
    #    path = join(dir, 'mainland_only.csv')
    #    p1 = read_polygon(path)
        
            
##
# @brief Get list of variable names in an expression string.
# @param source A string containing a python expression.
# @return A list of variable name strings.
# @note Throws SyntaxError exception if not a valid expression.
def get_vars_in_expression(source):
    '''Get list of variable names in a python expression.'''

    import compiler
    from compiler.ast import Node

    ##
    # @brief Internal recursive function.
    # @param node An AST parse Node.
    # @param var_list Input list of variables.
    # @return An updated list of variables.
    def get_vars_body(node, var_list=[]):
        if isinstance(node, Node):
            if node.__class__.__name__ == 'Name':
                for child in node.getChildren():
                    if child not in var_list:
                        var_list.append(child)
            for child in node.getChildren():
                if isinstance(child, Node):
                    for child in node.getChildren():
                        var_list = get_vars_body(child, var_list)
                    break

        return var_list

    return get_vars_body(compiler.parse(source))


##
# @brief Get a file from the web.
# @param file_url URL of the file to fetch.
# @param file_name Path to file to create in the filesystem.
# @param auth Auth tuple (httpproxy, proxyuser, proxypass).
# @param blocksize Read file in this block size.
# @return 'auth' tuple for subsequent calls, if successful, else False.
# @note If 'auth' not supplied, will prompt user.
# @note Will try using environment variable HTTP_PROXY for proxy server.
# @note Will try using environment variable PROXY_USERNAME for proxy username.
# @note Will try using environment variable PROXY_PASSWORD for proxy password.
def get_web_file(file_url, file_name, auth=None, blocksize=1024*1024):
    '''Get a file from the web (HTTP).

    file_url:  The URL of the file to get
    file_name: Local path to save loaded file in
    auth:      A tuple (httpproxy, proxyuser, proxypass)
    blocksize: Block size of file reads
    
    Will try simple load through urllib first.  Drop down to urllib2
    if there is a proxy and it requires authentication.

    Environment variable HTTP_PROXY can be used to supply proxy information.
    PROXY_USERNAME is used to supply the authentication username.
    PROXY_PASSWORD supplies the password, if you dare!
    '''

    # Simple fetch, if fails, check for proxy error
    try:
        urllib.urlretrieve(file_url, file_name)
        return None     # no proxy, no auth required
    except IOError, e:
        if e[1] == 407:     # proxy error
            pass
        elif e[1][0] == 113:  # no route to host
            print 'No route to host for %s' % file_url
            return False    # return False
        else:
            print 'Unknown connection error to %s' % file_url
            return False

    # We get here if there was a proxy error, get file through the proxy
    # unpack auth info
    try:
        (httpproxy, proxyuser, proxypass) = auth
    except:
        (httpproxy, proxyuser, proxypass) = (None, None, None)

    # fill in any gaps from the environment
    if httpproxy is None:
        httpproxy = os.getenv('HTTP_PROXY')
    if proxyuser is None:
        proxyuser = os.getenv('PROXY_USERNAME')
    if proxypass is None:
        proxypass = os.getenv('PROXY_PASSWORD')

    # Get auth info from user if still not supplied
    if httpproxy is None or proxyuser is None or proxypass is None:
        print '-'*52
        print ('You need to supply proxy authentication information.')
        if httpproxy is None:
            httpproxy = raw_input('                    proxy server: ')
        else:
            print '         HTTP proxy was supplied: %s' % httpproxy
        if proxyuser is None:
            proxyuser = raw_input('                  proxy username: ') 
        else:
            print 'HTTP proxy username was supplied: %s' % proxyuser
        if proxypass is None:
            proxypass = getpass.getpass('                  proxy password: ')
        else:
            print 'HTTP proxy password was supplied: %s' % '*'*len(proxyuser)
        print '-'*52

    # the proxy URL cannot start with 'http://', we add that later
    httpproxy = httpproxy.lower()
    if httpproxy.startswith('http://'):
        httpproxy = httpproxy.replace('http://', '', 1)

    # open remote file
    proxy = urllib2.ProxyHandler({'http': 'http://' + proxyuser
                                              + ':' + proxypass
                                              + '@' + httpproxy})
    authinfo = urllib2.HTTPBasicAuthHandler()
    opener = urllib2.build_opener(proxy, authinfo, urllib2.HTTPHandler)
    urllib2.install_opener(opener)
    try:
        webget = urllib2.urlopen(file_url)
    except urllib2.HTTPError:
        return False

    # transfer file to local filesystem
    fd = open(file_name, 'wb')
    while True:
        data = webget.read(blocksize)
        if len(data) == 0:
            break
        fd.write(data)
    fd.close
    webget.close()

    # return successful auth info
    return (httpproxy, proxyuser, proxypass)


##
# @brief Tar a file (or directory) into a tarfile.
# @param files A list of files (or directories) to tar.
# @param tarfile The created tarfile name.
# @note We use gzip compression.
def tar_file(files, tarname):
    '''Compress a file or directory into a tar file.'''

    o = tarfile.open(tarname, 'w:gz')
    for file in files:
        o.add(file)
    o.close()


##
# @brief Untar a file into an optional target directory.
# @param tarname Name of the file to untar.
# @param target_dir Directory to untar into.
def untar_file(tarname, target_dir='.'):
    '''Uncompress a tar file.'''

    o = tarfile.open(tarname, 'r:gz')
    members = o.getmembers()
    for member in members:
        o.extract(member, target_dir)
    o.close()


##
# @brief Return a hex digest (MD5) of a given file.
# @param filename Path to the file of interest.
# @param blocksize Size of data blocks to read.
# @return A hex digest string (16 bytes).
# @note Uses MD5 digest.
def get_file_hexdigest(filename, blocksize=1024*1024*10):
    '''Get a hex digest of a file.'''
    
    m = md5.new()
    fd = open(filename, 'r')
            
    while True:
        data = fd.read(blocksize)
        if len(data) == 0:
            break
        m.update(data)
                                                                
    fd.close()
    return m.hexdigest()

    fd = open(filename, 'r')


##
# @brief Create a file containing a hexdigest string of a data file.
# @param data_file Path to the file to get the hexdigest from.
# @param digest_file Path to hexdigest file to create.
# @note Uses MD5 digest.
def make_digest_file(data_file, digest_file):
    '''Create a file containing the hex digest string of a data file.'''
    
    hexdigest = get_file_hexdigest(data_file)
    fd = open(digest_file, 'w')
    fd.write(hexdigest)
    fd.close()

##
# @brief Function to return the length of a file.
# @param in_file Path to file to get length of.
# @return Number of lines in file.
# @note Doesn't count '\n' characters.
# @note Zero byte file, returns 0.
# @note No \n in file at all, but >0 chars, returns 1.
def file_length(in_file):
    '''Function to return the length of a file.'''

    fid = open(in_file)
    data = fid.readlines()
    fid.close()
    return len(data)


