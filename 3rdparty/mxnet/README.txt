Binary distribution of mxnet(https://github.com/dmlc/mxnet) on Win64.

This is a binary distribution of mxnet. It comes with precompiled
mxnet library and python packages. You can use it to install the python
package and to create C++ projects with 
MxNet.cpp (https://github.com/hjk41/MxNet.cpp).

Installing and setting up environment
===============
1. Download CUDNN v3 (https://developer.nvidia.com/cudnn) and unpack 
into %MXNET_HOME%\3rdparty\cudnn, so that there is 
%MXNET_HOME%\3rdparty\cudnn\cudnn64_70.dll.

2. Run setupenv.cmd by double clicking the file. This script will setup 
the necessary environmental variables for you. You can also set the
environment up yourselves, refer to setupenv.cmd for more information.

Installing Python package
===============
To install Python package of Mxnet:
    cd python
    python setup.py install

The setup script will download dependent packages and install mxnet.
If everything goes well, you should be able to use mxnet like this:
    C:\> python
    >>> import mxnet as mx
    >>> a=mx.nd.zeros((2,3))
    >>> print(a.asnumpy())
    [[0. 0. 0.]
     [0. 0. 0.]]


Building C++ applications with Visual Studio
===============
Copy libmxnet.dll and libmxnet.lib to MxNet.cpp/lib/windows/
We have provided a sample C++ solution in 
MxNet.cpp/windows/vs/MxnetTestApp/MxnetTestApp.sln.
You can use it as a template to create your own projects.

Note that you need to use setupenv.cmd before you can execute the
executable, or you may encounter errors saying missing dll files.
The shell script just adds the necessary dll files to your %PATH%
to avoid the problem. You can follow the shell script and add the 
dll files to your environmental variables permenantly.