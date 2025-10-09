Dependency Installation for Mac OS
==================================

The minimum required version for installation is **macOS 14 (Sonoma)** or newer.

Install Python 3
----------------

The UML .FRI tool is written using the Python 3 language. For the application to function correctly, you must install
the latest version of Python from the official website.

- Open your web browser and navigate to the official Python downloads page: https://www.python.org/downloads/
- Download the latest macOS installer (a .pkg file).
- Run the downloaded installer package and complete the installation according to the prompts.

Obtaining GIT
-------------

GIT version control is used to store the source code of the UML .FRI application.

On macOS, Git is usually installed automatically the first time you use a Git command in the Terminal, or you can
install it using The Command Line Developer Tools for Xcode, which are significantly lighter than the full Xcode
application. You can obtain them by running the following command in the Terminal:

    xcode-select --install

This command will install the essential developer tools, including Git.

Downloading Sources using GIT
-----------------------------

With Git installed, you can download your copy of UML .FRI for the first time by using the following command:

    git clone https://github.com/umlfri/umlfri2.git

If you only want to update existing source code to the newest version, use the following command inside
the working copy:

    git pull

Install Other Dependencies
--------------------------

UML .FRI needs the pyparsing library for parsing ufl expressions, the lxml library for reading and writing XML, and
several other dependencies, including PyQt which handles the Graphical User Interface (GUI). These libraries
are installed using pip.

Just run the following command inside the UML .FRI working directory:

    pip3 install -r requirements.txt

Download the Icon Pack
----------------------

You may continue by starting the application if you want. However, it will look suboptimal without the installed
icon pack. You can download and install it automatically by starting the download_icons script:

    ./tools/icons/download_icons.py

Starting the Application
------------------------

After that, you can start UML .FRI by executing this inside the working directory:

    ./main.py

And now, it is completely up to you what you want to do and what you will do.
