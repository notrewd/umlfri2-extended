----------------------------------------------------------------

If there’s an error on macOS, open the source directory in terminal, and use:

find . -name ".DS_Store" -delete

----------------------------------------------------------------

# Automatic Java Import

UML .FRI now understands how to build a full class diagram directly from Java
source files:

1. Start the desktop application and open the **File** menu.
2. Choose **Import Java Sources**.
3. Select either a folder tree or a single `.java` file that should be
	analysed.
4. Enter a project name when prompted. A new solution is created and populated
	with the imported classes, inheritance links, implemented interfaces and
	basic associations derived from fields.

Warnings reported by the importer (for example syntax errors in individual
files) are displayed after the import finishes so you can fix the sources and
run the import again if necessary.